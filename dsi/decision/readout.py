from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dsi.decision.neutral import _evaluate_decision, binary_auc_score, continuous_auc_score, regression_metrics
from dsi.utils.logging_utils import setup_logging


class OrdinalDecisionDataset(Dataset):
    def __init__(self, features: np.ndarray, target_time: np.ndarray, rank_label: np.ndarray) -> None:
        super().__init__()
        logger = setup_logging(None)
        logger.info("Initializing OrdinalDecisionDataset. rows=%s", int(features.shape[0]))

        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.target_time = torch.as_tensor(target_time, dtype=torch.float32)
        self.rank_label = torch.as_tensor(rank_label, dtype=torch.float32)

        logger.info("OrdinalDecisionDataset initialized.")

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[index], self.target_time[index], self.rank_label[index]


def _iter_decision_batches(
    features: np.ndarray,
    target: np.ndarray,
    rank_label: np.ndarray,
    batch_size: int,
    device: torch.device,
    shuffle_blocks: bool,
    seed: int,
    id_features: np.ndarray | None = None,
) -> Any:

    row_count = int(features.shape[0])
    step = int(batch_size)
    starts = np.arange(0, row_count, step, dtype=np.int64)

    if bool(shuffle_blocks) and starts.size > 1:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(starts)

    for start_value in starts:
        start = int(start_value)
        end = min(start + step, row_count)
        batch_features = torch.as_tensor(features[start:end], dtype=torch.float32).to(device, non_blocking=True)
        batch_target = torch.as_tensor(target[start:end], dtype=torch.float32).to(device, non_blocking=True)
        batch_rank = torch.as_tensor(rank_label[start:end], dtype=torch.float32).to(device, non_blocking=True)
        batch_ids = None
        if id_features is not None:
            batch_ids = torch.as_tensor(id_features[start:end], dtype=torch.long).to(device, non_blocking=True)
        yield start, end, batch_features, batch_target, batch_rank, batch_ids


class MLPBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int, dropout: float) -> None:
        super().__init__()
        logger = setup_logging(None)
        logger.info("Initializing MLPBackbone. input_dim=%s hidden=%s layers=%s", int(input_dim), int(hidden_dim), int(hidden_layers))

        layers: list[nn.Module] = []
        current = int(input_dim)
        for _ in range(max(int(hidden_layers), 1)):
            layers.extend(
                [
                    nn.Linear(current, int(hidden_dim)),
                    nn.LayerNorm(int(hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            current = int(hidden_dim)
        self.net = nn.Sequential(*layers)
        self.output_dim = int(hidden_dim)

        logger.info("MLPBackbone initialized. output_dim=%s", self.output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class CrossNetwork(nn.Module):
    def __init__(self, input_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        logger = setup_logging(None)
        logger.info("Initializing CrossNetwork. input_dim=%s layers=%s", int(input_dim), int(layers))

        self.layers = nn.ModuleList([nn.Linear(int(input_dim), int(input_dim)) for _ in range(max(int(layers), 1))])
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(int(input_dim))

        logger.info("CrossNetwork initialized.")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x0 = features
        cross = features
        for layer in self.layers:
            cross = cross + x0 * self.dropout(layer(cross))
        return self.norm(cross)


class DCNBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int, dropout: float, cross_layers: int) -> None:
        super().__init__()
        logger = setup_logging(None)
        logger.info("Initializing DCNBackbone. input_dim=%s hidden=%s", int(input_dim), int(hidden_dim))

        self.cross = CrossNetwork(int(input_dim), int(cross_layers), float(dropout))
        self.deep = MLPBackbone(int(input_dim), int(hidden_dim), int(hidden_layers), float(dropout))
        self.output_dim = int(input_dim) + int(hidden_dim)

        logger.info("DCNBackbone initialized. output_dim=%s", self.output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.cross(features), self.deep(features)], dim=-1)


class IDAwareBackbone(nn.Module):
    def __init__(
        self,
        base_backbone: nn.Module,
        cardinalities: list[int],
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        logger = setup_logging(None)
        logger.info("Initializing IDAwareBackbone. fields=%s embed_dim=%s", len(cardinalities), int(embedding_dim))

        self.base_backbone = base_backbone
        self.embeddings = nn.ModuleList(
            nn.Embedding(max(int(cardinality), 1), int(embedding_dim))
            for cardinality in cardinalities
        )
        self.dropout = nn.Dropout(float(dropout))
        self.output_dim = int(base_backbone.output_dim) + len(self.embeddings) * int(embedding_dim)

        logger.info("IDAwareBackbone initialized. output_dim=%s", self.output_dim)

    def forward(self, features: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.base_backbone(features)
        if ids is None or ids.shape[1] == 0 or not self.embeddings:
            return hidden
        parts = [hidden]
        for idx, embedding in enumerate(self.embeddings):
            value = ids[:, idx].long().clamp_min(0).clamp_max(embedding.num_embeddings - 1)
            parts.append(self.dropout(embedding(value)))
        return torch.cat(parts, dim=1)


class IDInputBackbone(nn.Module):
    def __init__(
        self,
        scorer_name: str,
        input_dim: int,
        cardinalities: list[int],
        embedding_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        dropout: float,
        cross_layers: int,
    ) -> None:
        super().__init__()
        logger = setup_logging(None)
        logger.info("Initializing IDInputBackbone. scorer=%s fields=%s embed_dim=%s", scorer_name, len(cardinalities), int(embedding_dim))

        self.embeddings = nn.ModuleList(
            nn.Embedding(max(int(cardinality), 1), int(embedding_dim))
            for cardinality in cardinalities
        )
        self.dropout = nn.Dropout(float(dropout))

        expanded_dim = int(input_dim) + len(self.embeddings) * int(embedding_dim)
        if str(scorer_name) == "mlp":
            self.base_backbone = MLPBackbone(expanded_dim, int(hidden_dim), int(hidden_layers), float(dropout))
        elif str(scorer_name) == "dcn":
            self.base_backbone = DCNBackbone(expanded_dim, int(hidden_dim), int(hidden_layers), float(dropout), int(cross_layers))
        else:
            raise ValueError(f"Unsupported scorer={scorer_name}")
        self.output_dim = int(self.base_backbone.output_dim)

        logger.info("IDInputBackbone initialized. input_dim=%s output_dim=%s", expanded_dim, self.output_dim)

    def forward(self, features: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        if ids is None or ids.shape[1] == 0 or not self.embeddings:
            return self.base_backbone(features)
        parts = [features]
        for idx, embedding in enumerate(self.embeddings):
            value = ids[:, idx].long().clamp_min(0).clamp_max(embedding.num_embeddings - 1)
            parts.append(self.dropout(embedding(value)))
        return self.base_backbone(torch.cat(parts, dim=1))


def _forward_backbone(backbone: nn.Module, features: torch.Tensor, ids: torch.Tensor | None) -> torch.Tensor:
    if isinstance(backbone, (IDAwareBackbone, IDInputBackbone)):
        return backbone(features, ids)
    return backbone(features)


class OrdinalDecisionModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        split_nodes: np.ndarray,
    ) -> None:
        super().__init__()
        logger = setup_logging(None)
        logger.info("Initializing OrdinalDecisionModel. bins=%s mode=integral", len(split_nodes))

        self.backbone = backbone
        split_tensor = torch.as_tensor(split_nodes, dtype=torch.float32).view(-1)
        self.register_buffer("split_nodes", split_tensor)

        self.ordinal_head = nn.Linear(int(backbone.output_dim), int(split_tensor.numel()))
        self.rank_head = nn.Linear(int(backbone.output_dim), 1)

        logger.info("OrdinalDecisionModel initialized.")

    def restored_time(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        left_nodes = torch.cat([self.split_nodes.new_tensor([0.0]), self.split_nodes[:-1]])
        bucket_width = (self.split_nodes - left_nodes).clamp(min=0.0)
        return torch.sum(probs * bucket_width.view(1, -1), dim=1)

    def forward(self, features: torch.Tensor, ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = _forward_backbone(self.backbone, features, ids)
        ordinal_logits = self.ordinal_head(hidden)
        pred_time = self.restored_time(ordinal_logits)
        rank_logit = self.rank_head(hidden).squeeze(-1)
        return pred_time, rank_logit, ordinal_logits


def _standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:

    train_float = np.asarray(train, dtype=np.float32)
    mean = train_float.mean(axis=0, keepdims=True)
    std = train_float.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    outputs = [((train_float - mean) / std).astype(np.float32, copy=False)]
    for value in others:
        outputs.append(((np.asarray(value, dtype=np.float32) - mean) / std).astype(np.float32, copy=False))
    return tuple(outputs)


def _build_backbone(args: argparse.Namespace, scorer_name: str, input_dim: int) -> nn.Module:
    logger = setup_logging(None)
    logger.info("Building the decision backbone. scorer=%s input_dim=%s", scorer_name, int(input_dim))

    cardinalities = [int(value) for value in getattr(args, "decision_sparse_cardinalities", [])]
    id_mode = str(getattr(args, "decision_id_mode", "append"))
    if bool(getattr(args, "decision_id_features", False)) and cardinalities and id_mode == "input":
        model = IDInputBackbone(
            str(scorer_name),
            int(input_dim),
            cardinalities,
            int(getattr(args, "decision_id_embedding_dim", 16)),
            int(args.scorer_hidden_dim),
            int(args.scorer_hidden_layers),
            float(args.scorer_dropout),
            int(args.scorer_cross_layers),
        )
        logger.info("Decision backbone built. scorer=%s output_dim=%s", scorer_name, int(model.output_dim))
        return model

    if str(scorer_name) == "mlp":
        model: nn.Module = MLPBackbone(int(input_dim), int(args.scorer_hidden_dim), int(args.scorer_hidden_layers), float(args.scorer_dropout))
    elif str(scorer_name) == "dcn":
        model = DCNBackbone(
            int(input_dim),
            int(args.scorer_hidden_dim),
            int(args.scorer_hidden_layers),
            float(args.scorer_dropout),
            int(args.scorer_cross_layers),
        )
    else:
        raise ValueError(f"Unsupported scorer={scorer_name}")

    if bool(getattr(args, "decision_id_features", False)) and cardinalities:
        model = IDAwareBackbone(model, cardinalities, int(getattr(args, "decision_id_embedding_dim", 16)), float(args.scorer_dropout))

    logger.info("Decision backbone built. scorer=%s output_dim=%s", scorer_name, int(model.output_dim))
    return model


def _quantile_from_sorted(sorted_values: np.ndarray, probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs.astype(np.float64, copy=False), 0.0, 1.0)
    positions = clipped * float(len(sorted_values) - 1)
    left = np.floor(positions).astype(np.int64)
    right = np.ceil(positions).astype(np.int64)
    weight = positions - left.astype(np.float64)
    return (sorted_values[left] * (1.0 - weight) + sorted_values[right] * weight).astype(np.float32)


def _search_ordinal_split_nodes(labels: np.ndarray, bin_count: int, beta: float, logger) -> np.ndarray:
    logger.info("Searching for ordinal thresholds. bins=%s beta=%.6f", int(bin_count), float(beta))

    flat_labels = labels.reshape(-1).astype(np.float32, copy=False)
    sorted_labels = np.sort(flat_labels)
    z = np.arange(1, int(bin_count) + 1, dtype=np.float64) / float(bin_count)
    alpha_space = np.concatenate([np.array([0.0], dtype=np.float64), np.arange(0.001, 5.0, 0.1, dtype=np.float64)])

    best_loss = float("inf")
    best_alpha = 0.0
    best_split = _quantile_from_sorted(sorted_labels, z)
    for alpha in alpha_space:
        gamma = z if abs(float(alpha)) < 1e-12 else (1.0 - np.exp(-alpha * z)) / max(1.0 - np.exp(-alpha), 1e-12)
        split_nodes = _quantile_from_sorted(sorted_labels, gamma)
        cdf_left = np.concatenate([np.array([0.0], dtype=np.float64), gamma[:-1]])
        cdf_right = gamma
        cdf_delta = np.clip(cdf_right - cdf_left, 1e-12, None)
        node_left = np.concatenate([np.array([0.0], dtype=np.float32), split_nodes[:-1]]).astype(np.float64)
        node_delta = split_nodes.astype(np.float64) - node_left
        cdf_square_sum = float(np.sum(np.square(cdf_delta)))
        learning_bound = cdf_square_sum * float(np.sum(np.square(node_delta) / cdf_delta))
        restore_bound = cdf_square_sum * float(np.sum(np.square(node_delta)))
        objective = learning_bound + float(beta) * restore_bound
        if objective < best_loss:
            best_loss = float(objective)
            best_alpha = float(alpha)
            best_split = split_nodes.astype(np.float32, copy=True)

    logger.info("Ordinal threshold search complete. alpha=%.6f objective=%.6f", best_alpha, best_loss)
    return best_split


def _ordinal_curve_losses(
    ordinal_logits: torch.Tensor,
    target_time: torch.Tensor,
    threshold_batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    ordinal_target = (target_time.view(-1, 1) > threshold_batch).float()

    ordinal_ce = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_target, reduction="none").sum(dim=1).mean()

    ordinal_prob = torch.sigmoid(ordinal_logits)
    ordinal_prior = F.relu(ordinal_prob[:, 1:] - ordinal_prob[:, :-1]).sum(dim=1).mean()
    return ordinal_ce, ordinal_prior


def _fit_ordinal_decision(
    scorer_name: str,
    train_x: np.ndarray,
    train_time: np.ndarray,
    train_rank: np.ndarray,
    val_x: np.ndarray,
    val_time: np.ndarray,
    val_rank: np.ndarray,
    train_ids: np.ndarray | None,
    val_ids: np.ndarray | None,
    device: torch.device,
    args: argparse.Namespace,
    logger,
) -> OrdinalDecisionModel:
    logger.info("Training the ordinal decision readout. scorer=%s rows=%s dim=%s", scorer_name, train_x.shape[0], train_x.shape[1])

    backbone = _build_backbone(args, scorer_name, train_x.shape[1])
    split_nodes = _search_ordinal_split_nodes(train_time, int(args.ordinal_bins), float(args.ordinal_beta), logger)
    model = OrdinalDecisionModel(backbone, split_nodes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.scorer_lr), weight_decay=float(args.scorer_weight_decay))

    batch_size = int(args.batch_size)
    ordinal_ce_weight = float(getattr(args, "ordinal_ce_weight", 0.0))
    ordinal_prior_weight = float(getattr(args, "ordinal_prior_weight", 0.0))
    use_sample_shuffle = (
        train_ids is None
        and ordinal_ce_weight <= 0.0
        and ordinal_prior_weight <= 0.0
    )
    if use_sample_shuffle:
        logger.info("Ordinal decision uses sample-level shuffling to reproduce the compact ablation C/E training protocol.")
        train_loader = DataLoader(
            OrdinalDecisionDataset(train_x, train_time, train_rank),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = None

    best_value = float("inf")
    best_state = None
    for epoch in range(1, int(args.scorer_epochs) + 1):
        model.train()
        losses: list[float] = []
        if use_sample_shuffle and train_loader is not None:
            for features, target_time, rank_label in train_loader:

                features = features.to(device, non_blocking=True)
                target_time = target_time.to(device, non_blocking=True)
                rank_label = rank_label.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                pred_time, rank_logit, _ = model(features)
                restore_loss = F.smooth_l1_loss(pred_time, target_time)
                rank_loss = F.binary_cross_entropy_with_logits(rank_logit, rank_label)
                loss = float(args.ordinal_restore_weight) * restore_loss + float(args.rank_loss_weight) * rank_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.scorer_clip_grad_norm))
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        else:
            for start, end, features, target_time, rank_label, batch_ids in _iter_decision_batches(
                train_x,
                train_time,
                train_rank,
                batch_size,
                device,
                True,
                int(args.seed) + epoch,
                train_ids,
            ):
                optimizer.zero_grad(set_to_none=True)
                pred_time, rank_logit, ordinal_logits = model(features, batch_ids)
                threshold_batch = model.split_nodes.view(1, -1)
                restore_loss = F.smooth_l1_loss(pred_time, target_time)
                rank_loss = F.binary_cross_entropy_with_logits(rank_logit, rank_label)
                loss = float(args.ordinal_restore_weight) * restore_loss + float(args.rank_loss_weight) * rank_loss

                if ordinal_ce_weight > 0.0 or ordinal_prior_weight > 0.0:
                    ordinal_ce, ordinal_prior = _ordinal_curve_losses(ordinal_logits, target_time, threshold_batch)
                if ordinal_ce_weight > 0.0:
                    loss = loss + ordinal_ce_weight * ordinal_ce
                if ordinal_prior_weight > 0.0:
                    loss = loss + ordinal_prior_weight * ordinal_prior
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.scorer_clip_grad_norm))
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        val_pred, val_score = _predict_ordinal_decision(model, val_x, device, int(args.batch_size), val_ids)
        select_value = _validation_select_value(args.validation_select, val_time, val_pred, val_rank, val_score)
        logger.info("Ordinal-decision epoch complete. scorer=%s epoch=%s loss=%.6f select=%.6f", scorer_name, epoch, float(np.mean(losses)), select_value)
        if select_value < best_value:
            best_value = float(select_value)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("Ordinal decision readout training complete. scorer=%s best_select=%.6f", scorer_name, best_value)
    return model


@torch.no_grad()


def _predict_ordinal_decision(
    model: OrdinalDecisionModel,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
    id_features: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:

    model.eval()
    pred_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    tensor = torch.as_tensor(features, dtype=torch.float32)
    for start in range(0, features.shape[0], int(batch_size)):
        end = min(start + int(batch_size), features.shape[0])
        batch = tensor[start:end].to(device)
        batch_ids = torch.as_tensor(id_features[start:end], dtype=torch.long, device=device) if id_features is not None else None
        pred_time, rank_score, _ = model(batch, batch_ids)
        pred_parts.append(pred_time.detach().cpu().numpy().astype(np.float32, copy=False))
        score_parts.append(rank_score.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(pred_parts, axis=0), np.concatenate(score_parts, axis=0)


def _validation_select_value(select_name: str, true_time: np.ndarray, pred_time: np.ndarray, rank_label: np.ndarray, rank_score: np.ndarray) -> float:

    if str(select_name) == "mae":
        return float(regression_metrics(pred_time, true_time)["mae_sec"])
    if str(select_name) == "xauc":
        return -float(continuous_auc_score(true_time, pred_time))
    if str(select_name) == "rank_auc":
        return -float(binary_auc_score(rank_label, rank_score))
    raise ValueError(f"Unsupported validation_select={select_name}")


def _seed_offset(scorer_name: str, feature_set_name: str) -> int:

    scorer_offsets = {"mlp": 101, "dcn": 211}
    feature_offsets = {
        "raw": 1009,
        "raw_padded": 2003,
        "posterior": 3001,
        "raw_equivalent": 4001,
        "posterior_equivalent": 4001,
        "fused": 4001,
    }
    return int(scorer_offsets.get(str(scorer_name), 997) + feature_offsets.get(str(feature_set_name), 7919))


def _run_feature_set(
    scorer_name: str,
    feature_set_name: str,
    matrices: dict[str, np.ndarray],
    labels: dict[str, dict[str, np.ndarray]],
    play_time: dict[str, np.ndarray],
    id_features: dict[str, np.ndarray] | None,
    eval_arrays: dict[str, np.ndarray],
    device: torch.device,
    args: argparse.Namespace,
    logger,
) -> dict[str, Any]:
    logger.info("Running the decision feature set. scorer=%s feature_set=%s", scorer_name, feature_set_name)

    train_x, val_x, test_x = _standardize(matrices["train"], matrices["val"], matrices["test"])
    rank_target = str(args.rank_target)
    train_rank = labels[rank_target]["train"]
    val_rank = labels[rank_target]["val"]
    train_ids = id_features["train"] if bool(getattr(args, "decision_id_features", False)) and id_features is not None else None
    val_ids = id_features["val"] if bool(getattr(args, "decision_id_features", False)) and id_features is not None else None
    test_ids = id_features["test"] if bool(getattr(args, "decision_id_features", False)) and id_features is not None else None

    model = _fit_ordinal_decision(
        scorer_name,
        train_x,
        play_time["train"],
        train_rank,
        val_x,
        play_time["val"],
        val_rank,
        train_ids,
        val_ids,
        device,
        args,
        logger,
    )
    val_pred, _ = _predict_ordinal_decision(model, val_x, device, int(args.batch_size), val_ids)
    test_pred, test_score = _predict_ordinal_decision(model, test_x, device, int(args.batch_size), test_ids)

    save_predictions = getattr(args, "save_predictions", None)
    if save_predictions:
        logger.info("Saving decision test predictions. scorer=%s feature_set=%s", scorer_name, feature_set_name)

        output_path = Path(str(save_predictions)).expanduser()
        if output_path.suffix:
            output_path = output_path.with_name(f"{output_path.stem}_{scorer_name}_{feature_set_name}{output_path.suffix}")
        else:
            output_path = output_path / f"{scorer_name}_{feature_set_name}.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_path,
            pred_time=np.asarray(test_pred, dtype=np.float32),
            rank_score=np.asarray(test_score, dtype=np.float32),
            play_time=np.asarray(eval_arrays["play_time"], dtype=np.float32),
            duration_sec=np.asarray(eval_arrays["duration_sec"], dtype=np.float32),
            watch_event=np.asarray(eval_arrays["watch_event"], dtype=np.int64),
            session_key=np.asarray(eval_arrays["session_key"], dtype=np.int64),
            long_label=np.asarray(eval_arrays["long_label"], dtype=np.float32),
            overplay_label=np.asarray(eval_arrays["overplay_label"], dtype=np.float32),
        )
        logger.info("Decision test predictions saved. path=%s", output_path)

    metrics = _evaluate_decision(
        test_pred,
        test_score,
        eval_arrays,
        [int(value) for value in args.topks],
        f"{scorer_name}:{feature_set_name}",
        bool(args.fast_eval),
        int(args.fast_eval_rows),
        int(args.seed),
        logger,
    )
    logger.info("Decision feature-set run complete. scorer=%s feature_set=%s", scorer_name, feature_set_name)
    return metrics
