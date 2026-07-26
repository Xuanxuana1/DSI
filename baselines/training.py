#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.data import DSIBaselineDataLoaders
from baselines.adapter import build_adapter, build_gr_numeric_vocab, list_supported_models
from baselines.model import CWM, Cread, D2Q, DIFL, EGMN, GR, TPM, WideAndDeep
from baselines.utils import (
    build_quantile_table,
    duration_quantile_edges,
    eval_binary_auc,
    eval_mae,
    eval_rmse,
    eval_xauc,
    get_playtime_percentiles_range,
)
from dsi.commands import load_config, log_path, report_path
from dsi.decision.neutral import session_ndcg_metrics
from dsi.utils.bridge import portableize_serialized_paths

logger = logging.getLogger(__name__)


def setup_logger(path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def set_seed(seed: int) -> None:

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _quantile_from_sorted(sorted_values: np.ndarray, probs: np.ndarray) -> np.ndarray:
    logger.debug("Reading ranking quantiles.")

    clipped = np.clip(probs.astype(np.float64, copy=False), 0.0, 1.0)
    positions = clipped * float(len(sorted_values) - 1)
    left = np.floor(positions).astype(np.int64)
    right = np.ceil(positions).astype(np.int64)
    weight = positions - left.astype(np.float64)

    values = sorted_values[left] * (1.0 - weight) + sorted_values[right] * weight
    logger.debug("Ranking quantiles loaded.")
    return values.astype(np.float32)


def search_cread_split_nodes(labels: np.ndarray, head_num: int, beta: float) -> tuple[np.ndarray, dict[str, float]]:
    logger.info("Searching for CREAD thresholds. heads=%d beta=%.6f", int(head_num), float(beta))

    flat_labels = labels.reshape(-1).astype(np.float32, copy=False)
    sorted_labels = np.sort(flat_labels)
    z = np.arange(1, int(head_num) + 1, dtype=np.float64) / float(head_num)
    alpha_space = np.arange(0.001, 5.0, 0.1, dtype=np.float64)
    beta_value = float(beta)

    best_loss = float("inf")
    best_alpha = float(alpha_space[0])
    best_split = np.zeros(int(head_num), dtype=np.float32)
    for alpha in alpha_space:
        gamma = z if abs(float(alpha)) < 1e-12 else (1.0 - np.exp(-alpha * z)) / max(1.0 - np.exp(-alpha), 1e-12)
        split_nodes = _quantile_from_sorted(sorted_labels, gamma)
        cdf_left = np.concatenate([np.array([0.0], dtype=np.float64), gamma[:-1]])
        cdf_right = gamma
        node_left = np.concatenate([np.array([0.0], dtype=np.float32), split_nodes[:-1]]).astype(np.float64)
        node_right = split_nodes.astype(np.float64)
        cdf_delta = np.clip(cdf_right - cdf_left, 1e-12, None)
        node_delta = node_right - node_left
        cdf_square_sum = float(np.sum(np.square(cdf_delta)))
        a_w = cdf_square_sum * float(np.sum(np.square(node_delta) / cdf_delta))
        a_b = cdf_square_sum * float(np.sum(np.square(node_delta)))
        a_loss = a_w + beta_value * a_b
        if a_loss < best_loss:
            best_loss = float(a_loss)
            best_alpha = float(alpha)
            best_split = split_nodes.astype(np.float32, copy=True)

    logger.info("CREAD threshold search complete. best_alpha=%.6f best_loss=%.6f", best_alpha, best_loss)
    return best_split, {
        "cread_split_search_alpha": best_alpha,
        "cread_split_search_beta": beta_value,
        "cread_split_search_loss": best_loss,
    }


def build_ratio_quantile_table(labels: np.ndarray, durations: np.ndarray, bucket_num: int, quantile_bins: int) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Building the DIFL ratio-quantile table. bucket_num=%d quantile_bins=%d", int(bucket_num), int(quantile_bins))

    edges = duration_quantile_edges(durations, int(bucket_num))
    bucket_ids = np.searchsorted(edges, durations.reshape(-1), side="left").astype(np.int64)
    bucket_count = int(edges.size + 1)
    ratio = (labels.astype(np.float64, copy=False) / np.maximum(durations.astype(np.float64, copy=False), 1e-6)).clip(0.0, 10.0)
    probs = np.linspace(0.0, 1.0, int(quantile_bins), dtype=np.float64)
    table = np.zeros((bucket_count, int(quantile_bins)), dtype=np.float32)
    fallback = np.quantile(ratio, probs).astype(np.float32)

    for bucket_id in range(bucket_count):
        mask = bucket_ids == bucket_id
        if np.any(mask):
            table[bucket_id] = np.quantile(ratio[mask], probs).astype(np.float32)
        else:
            table[bucket_id] = fallback
    logger.info("DIFL ratio-quantile table built. actual_buckets=%d", bucket_count)
    return edges.astype(np.float32, copy=False), table


def build_baseline(name: str, description: list[tuple[str, int, str]], args: argparse.Namespace) -> torch.nn.Module:
    logger.info("Building baseline model. name=%s", name)

    if name in {"vr", "wlr", "wd"}:
        model = WideAndDeep(
            description,
            embed_dim=int(args.embed_dim),
            mlp_dims=tuple(int(v) for v in args.mlp_dims),
            dropout=float(args.dropout),
        )
        logger.info("Baseline model built. name=%s", name)
        return model

    if name == "cwm":
        model = CWM(
            description,
            embed_dim=int(args.cwm_embed_dim),
        )
        logger.info("Baseline model built. name=%s", name)
        return model

    if name == "egmn":
        model = EGMN(
            description,
            embed_dim=int(args.embed_dim),
            share_mlp_dims=tuple(int(v) for v in args.share_mlp_dims),
            dropout=float(args.egmn_dropout),
            components=int(args.egmn_components),
        )
        logger.info("Baseline model built. name=%s", name)
        return model

    if name == "tpm":
        model = TPM(
            description,
            class_num=int(args.tpm_bins) - 1,
            embed_dim=int(args.embed_dim),
            mlp_dims=tuple(int(v) for v in args.mlp_dims),
            dropout=float(args.dropout),
        )
        logger.info("Baseline model built. name=%s", name)
        return model

    if name == "d2q":
        model = D2Q(
            description,
            embed_dim=int(args.embed_dim),
            mlp_dims=tuple(int(v) for v in args.mlp_dims),
            dropout=float(args.dropout),
        )
        logger.info("Baseline model built. name=%s", name)
        return model

    if name == "cread":
        model = Cread(
            description,
            embed_dim=int(args.embed_dim),
            share_mlp_dims=tuple(int(v) for v in args.cread_share_mlp_dims),
            output_mlp_dims=tuple(int(v) for v in args.cread_output_dims),
            head_num=int(args.cread_heads),
            dropout=float(args.dropout),
        )
        logger.info("Baseline model built. name=%s", name)
        return model

    if name == "gr":
        model = GR(
            description,
            vocab_size=int(args.gr_vocab_size),
            duration_buckets=int(args.duration_buckets),
            feat_dim=int(args.gr_feat_dim),
            hidden_dim=int(args.gr_hidden_dim),
            n_head=int(args.gr_heads),
            decoder_layers=int(args.gr_decoder_layers),
            ffn_dim=int(args.gr_ffn_dim),
            dropout=float(args.gr_dropout),
            use_embedding_mixup=bool(args.gr_embedding_mixup),
        )
        logger.info("Baseline model built. name=%s", name)
        return model

    if name == "difl":
        model = DIFL(
            description,
            duration_buckets=int(args.duration_buckets),
            feat_dim=int(args.difl_feat_dim),
            dropout=float(args.difl_dropout),
        )
        logger.info("Baseline model built. name=%s", name)
        return model

    raise ValueError(f"Unsupported baseline: {name}")


def apply_baseline_defaults(args: argparse.Namespace) -> None:
    logger.info("Applying shared baseline defaults. baseline=%s", args.baseline)

    if args.lr is None:
        if str(args.baseline) == "gr":

            args.lr = 0.005
        else:
            args.lr = 0.01

    if args.epochs is None:
        args.epochs = 5

    logger.info("Shared baseline defaults applied. epochs=%d lr=%.6f", int(args.epochs), float(args.lr))


def prepare_gr_vocab_size(args: argparse.Namespace, loaders: DSIBaselineDataLoaders) -> dict[str, Any]:
    logger.info("Prebuilding the GR vocabulary.")

    train_dataset = loaders["train"].dataset
    labels = train_dataset.play_time.astype(np.float32, copy=False)
    durations = train_dataset.duration.astype(np.float32, copy=False)

    max_numeric_tokens = None
    if int(args.gr_max_vocab_size) > 0:
        max_numeric_tokens = max(int(args.gr_max_vocab_size) - 3, 1)
    numeric_vocab = build_gr_numeric_vocab(
        labels,
        durations,
        float(args.gr_q_start),
        float(args.gr_q_end),
        float(args.gr_q_decay_rate),
        float(args.gr_epsilon),
        max_numeric_tokens=max_numeric_tokens,
    )

    args.gr_vocab_size = int(len(numeric_vocab) + 3)
    args.gr_numeric_vocab = [int(value) for value in numeric_vocab]
    logger.info("GR vocabulary prebuilt. vocab_size=%d numeric_tokens=%d", int(args.gr_vocab_size), int(len(numeric_vocab)))
    return {"gr_vocab_size": int(args.gr_vocab_size), "gr_numeric_tokens": int(len(numeric_vocab))}


def bind_references(adapter, name: str, loaders: DSIBaselineDataLoaders, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    logger.info("Preparing baseline auxiliary tables. name=%s", name)

    train_dataset = loaders["train"].dataset
    labels = train_dataset.play_time.astype(np.float32, copy=False)
    durations = train_dataset.duration.astype(np.float32, copy=False)
    info: dict[str, Any] = {}

    if name == "wlr":
        label_scale = max(float(np.max(labels)), 1e-6)
        adapter.set_label_scale(label_scale)
        info["wlr_label_space"] = "normalized_by_train_max"
        info["wlr_label_scale_sec"] = label_scale

    if name == "egmn":
        label_scale = max(float(np.max(labels)), 1e-6)
        adapter.set_label_scale(label_scale)
        info["egmn_label_space"] = "normalized_by_train_max"
        info["egmn_label_scale_sec"] = label_scale

    if name == "cwm":
        adapter.set_params(float(args.cwm_c_inv), float(args.cwm_sigma))
        info["cwm_label_space"] = "play_time_truncate"
        info["cwm_c_inv"] = float(args.cwm_c_inv)
        info["cwm_sigma"] = float(args.cwm_sigma)
        info["cwm_backbone"] = "fm"

    if name == "tpm":
        begins, ends = get_playtime_percentiles_range(labels, int(args.tpm_bins), device)
        adapter.set_buckets(begins, ends, int(args.tpm_bins))
        info["tpm_bins"] = int(args.tpm_bins)

    if name == "d2q":
        edges, quantiles = build_quantile_table(labels, durations, int(args.duration_buckets), int(args.d2q_quantile_bins))
        adapter.set_quantiles(
            torch.as_tensor(edges, dtype=torch.float32, device=device),
            torch.as_tensor(quantiles, dtype=torch.float32, device=device),
        )
        info["duration_buckets"] = int(args.duration_buckets)
        info["d2q_quantile_bins"] = int(args.d2q_quantile_bins)

    if name == "cread":
        label_scale = max(float(np.max(labels)), 1e-6)
        search_labels = (labels / label_scale).astype(np.float32, copy=False)
        split_nodes, search_info = search_cread_split_nodes(search_labels, int(args.cread_heads), float(args.cread_beta))
        adapter.set_split_nodes(
            torch.as_tensor(split_nodes, dtype=torch.float32, device=device),
            label_scale=label_scale,
            lambda_ce=float(args.cread_lambda_ce),
            lambda_restore=float(args.cread_lambda_restore),
            lambda_ord=float(args.cread_lambda_ord),
        )
        info["cread_heads"] = int(args.cread_heads)
        info["cread_label_space"] = "normalized"
        info["cread_label_scale_sec"] = label_scale
        info["cread_split_node_mode"] = "paper_ead_alpha_search"
        info["cread_lambda_ce"] = float(args.cread_lambda_ce)
        info["cread_lambda_restore"] = float(args.cread_lambda_restore)
        info["cread_lambda_ord"] = float(args.cread_lambda_ord)
        info.update(search_info)

    if name == "gr":
        edges = duration_quantile_edges(durations, int(args.duration_buckets))
        adapter.set_references(
            labels,
            durations,
            torch.as_tensor(edges, dtype=torch.float32, device=device),
            float(args.gr_q_start),
            float(args.gr_q_end),
            float(args.gr_q_decay_rate),
            float(args.gr_epsilon),
            int(args.gr_max_len),
            int(args.gr_window_size),
            float(args.gr_cls_weight),
            float(args.gr_huber_weight),
            float(args.gr_teacher_force_ratio),
            getattr(args, "gr_numeric_vocab", None),
        )
        info["gr_vocab_size"] = int(args.gr_vocab_size)
        info["gr_numeric_tokens"] = int(max(int(args.gr_vocab_size) - 3, 0))
        info["gr_max_len"] = int(args.gr_max_len)
        info["duration_buckets"] = int(edges.size + 1)

    if name == "difl":
        edges, quantiles = build_ratio_quantile_table(labels, durations, int(args.duration_buckets), int(args.difl_quantile_bins))
        adapter.set_references(
            labels,
            durations,
            torch.as_tensor(edges, dtype=torch.float32, device=device),
            torch.as_tensor(quantiles, dtype=torch.float32, device=device),
            int(args.difl_micro_bs),
        )
        info["duration_buckets"] = int(edges.size + 1)
        info["difl_quantile_bins"] = int(args.difl_quantile_bins)
        info["difl_micro_bs"] = int(args.difl_micro_bs)

    logger.info("Baseline auxiliary tables ready. name=%s", name)
    return info


def evaluate_light(adapter, loaders: DSIBaselineDataLoaders, split: str, device: torch.device) -> dict[str, float]:
    logger.info("Running lightweight baseline evaluation. split=%s", split)

    adapter.eval()
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    row_count = 0
    with torch.no_grad():
        for features, label in loaders[split]:
            _, pred = adapter.get_base_pred(features)
            error = (pred.reshape(-1) - label.reshape(-1)).detach()
            absolute_error_sum += float(torch.abs(error).sum().cpu().item())
            squared_error_sum += float(torch.square(error).sum().cpu().item())
            row_count += int(error.numel())

    mae = absolute_error_sum / max(row_count, 1)
    rmse = float(np.sqrt(squared_error_sum / max(row_count, 1)))
    metrics = {"mae": float(mae), "rmse": float(rmse), "rows": float(row_count)}
    logger.info("Lightweight baseline evaluation complete. split=%s mae=%.6f rmse=%.6f", split, metrics["mae"], metrics["rmse"])
    return metrics


def train_base(adapter, loaders: DSIBaselineDataLoaders, args: argparse.Namespace, device: torch.device) -> list[dict[str, float]]:
    logger.info("Training the base baseline. epochs=%d lr=%.6f", int(args.epochs), float(args.lr))

    baseline_name = str(getattr(args, "baseline", "")).lower()
    if baseline_name in {"d2q", "egmn"}:
        optimizer = torch.optim.Adagrad(adapter.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        optimizer_name = "Adagrad"
    else:
        optimizer = torch.optim.Adam(adapter.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        optimizer_name = "Adam"
    logger.info("Base baseline optimizer selected. baseline=%s optimizer=%s", baseline_name, optimizer_name)
    history: list[dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        adapter.train()
        total_loss = 0.0
        total_batches = 0
        for features, label in loaders["train"]:
            loss = adapter.stage1_loss(features, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(adapter.parameters()), max_norm=float(args.clip_grad_norm))
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            total_batches += 1

        train_loss = total_loss / max(total_batches, 1)
        val_metrics = evaluate_light(adapter, loaders, "val", device)
        row = {"epoch": float(epoch), "train_loss": float(train_loss), **{f"val_{k}": float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))}}
        history.append(row)
        logger.info("Base training epoch complete. epoch=%d train_loss=%.6f val_mae=%.6f", epoch, train_loss, val_metrics["mae"])

    logger.info("Base baseline training complete.")
    return history


def topk_session_value(score: np.ndarray, reward: np.ndarray, session_key: np.ndarray, topk: int) -> dict[str, float]:

    session = np.asarray(session_key, dtype=np.int64).reshape(-1)
    pred = np.asarray(score, dtype=np.float64).reshape(-1)
    rel = np.asarray(reward, dtype=np.float64).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(rel)
    session = session[valid]
    pred = pred[valid]
    rel = rel[valid]
    order = np.argsort(session, kind="mergesort")
    session = session[order]
    pred = pred[order]
    rel = rel[order]

    starts = np.flatnonzero(np.r_[True, session[1:] != session[:-1]]) if session.size else np.asarray([], dtype=np.int64)
    ends = np.r_[starts[1:], session.size] if session.size else np.asarray([], dtype=np.int64)
    selected_sum = 0.0
    selected_count = 0
    usable_sessions = 0
    for start, end in zip(starts, ends):
        n = int(end - start)
        if n <= 0:
            continue
        k = min(int(topk), n)
        idx = np.argsort(-pred[start:end], kind="mergesort")[:k] + start
        selected_sum += float(np.sum(rel[idx]))
        selected_count += int(k)
        usable_sessions += 1

    return {
        "value": float(selected_sum / selected_count) if selected_count else float("nan"),
        "sessions": int(usable_sessions),
        "selected": int(selected_count),
    }


def evaluate(adapter, loaders: DSIBaselineDataLoaders, split: str, device: torch.device) -> dict[str, float]:
    logger.info("Evaluating the baseline. split=%s", split)

    adapter.eval()

    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    durations: list[np.ndarray] = []
    overplay_labels: list[np.ndarray] = []
    events: list[np.ndarray] = []
    sessions: list[np.ndarray] = []
    with torch.no_grad():
        for features, label in loaders[split]:
            _, pred = adapter.get_base_pred(features)
            labels.append(label.detach().cpu().numpy().reshape(-1))
            scores.append(pred.detach().cpu().numpy().reshape(-1))
            durations.append(features["duration"].detach().cpu().numpy().reshape(-1))
            event = features["event"].detach().cpu().numpy().reshape(-1)
            events.append(event)
            sessions.append(features["session_key"].detach().cpu().numpy().reshape(-1))
            overplay_labels.append((event == 3).astype(np.float32))

    y = np.concatenate(labels)
    pred = np.concatenate(scores)
    d = np.concatenate(durations)
    y_trunc = np.minimum(y, d)
    pred_trunc = np.minimum(np.maximum(pred, 0.0), d)
    event = np.concatenate(events)
    session = np.concatenate(sessions)
    op = np.concatenate(overplay_labels)
    long_label = np.isin(event, np.asarray([2, 3], dtype=np.int64)).astype(np.float64)
    watch_time_log = np.log1p(np.asarray(y, dtype=np.float64).clip(min=0.0))
    y_sec = np.asarray(y, dtype=np.float64).clip(min=0.0)
    op_label = op.astype(np.float64, copy=False)
    metrics = {
        "mae": eval_mae(y, pred),
        "rmse": eval_rmse(y, pred),
        "xauc": eval_xauc(y, pred),
        "mae_trunc": eval_mae(y_trunc, pred_trunc),
        "rmse_trunc": eval_rmse(y_trunc, pred_trunc),
        "xauc_trunc": eval_xauc(y_trunc, pred_trunc),
        "op_auc": eval_binary_auc(op, pred),
        "wt_ndcg@5": session_ndcg_metrics(session, pred, watch_time_log, ks=(5,))["ndcg@5"],
        "c_ndcg@5": session_ndcg_metrics(session, pred, long_label, ks=(5,))["ndcg@5"],
        "op_ndcg@5": session_ndcg_metrics(session, pred, op_label, ks=(5,))["ndcg@5"],
        "wt_top5_sec": topk_session_value(pred, y_sec, session, 5)["value"],
        "op_top5_rate": topk_session_value(pred, op_label, session, 5)["value"],
        "rows": float(len(y)),
    }
    logger.info("Baseline evaluation complete. split=%s mae=%.6f xauc=%.6f op_auc=%.6f", split, metrics["mae"], metrics["xauc"], metrics["op_auc"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train external baselines on DSI processed data.")
    parser.add_argument("--config", default="configs/kuairec.yaml")
    parser.add_argument("--baseline", choices=list_supported_models(), default="wlr")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-eval-rows", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split-prefix", choices=["standard", "random"], default="standard")
    parser.add_argument("--feature-protocol", choices=["clean", "full"], default="clean")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--clip-grad-norm", type=float, default=10.0)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--mlp-dims", nargs="+", type=int, default=[256, 128, 64])
    parser.add_argument("--share-mlp-dims", nargs="+", type=int, default=[256, 128, 64])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--egmn-components", type=int, default=10)
    parser.add_argument("--egmn-dropout", type=float, default=0.2)
    parser.add_argument("--tpm-bins", type=int, default=32)
    parser.add_argument("--duration-buckets", type=int, default=30)
    parser.add_argument("--d2q-quantile-bins", type=int, default=101)
    parser.add_argument("--cwm-embed-dim", type=int, default=10)
    parser.add_argument("--cwm-c-inv", type=float, default=40.0)
    parser.add_argument("--cwm-sigma", type=float, default=2.0)
    parser.add_argument("--cread-heads", type=int, default=50)
    parser.add_argument("--cread-beta", type=float, default=50.0)
    parser.add_argument("--cread-lambda-ce", type=float, default=1.0)
    parser.add_argument("--cread-lambda-restore", type=float, default=1.0)
    parser.add_argument("--cread-lambda-ord", type=float, default=0.00002)
    parser.add_argument("--cread-share-mlp-dims", nargs="+", type=int, default=[256, 128, 64])
    parser.add_argument("--cread-output-dims", nargs="+", type=int, default=[32])
    parser.add_argument("--gr-max-vocab-size", type=int, default=0)
    parser.add_argument("--gr-vocab-size", type=int, default=0)
    parser.add_argument("--gr-feat-dim", type=int, default=128)
    parser.add_argument("--gr-hidden-dim", type=int, default=128)
    parser.add_argument("--gr-heads", type=int, default=8)
    parser.add_argument("--gr-decoder-layers", type=int, default=3)
    parser.add_argument("--gr-ffn-dim", type=int, default=256)
    parser.add_argument("--gr-dropout", type=float, default=0.1)
    parser.add_argument("--gr-window-size", type=int, default=20)
    parser.add_argument("--gr-max-len", type=int, default=20)
    parser.add_argument("--gr-cls-weight", type=float, default=10.0)
    parser.add_argument("--gr-huber-weight", type=float, default=1.0)
    parser.add_argument("--gr-q-start", type=float, default=0.9999)
    parser.add_argument("--gr-q-end", type=float, default=0.9)
    parser.add_argument("--gr-q-decay-rate", type=float, default=0.99)
    parser.add_argument("--gr-epsilon", type=float, default=1e-6)
    parser.add_argument("--gr-teacher-force-ratio", type=float, default=0.5)
    parser.add_argument("--gr-embedding-mixup", action="store_true")
    parser.add_argument("--difl-feat-dim", type=int, default=32)
    parser.add_argument("--difl-dropout", type=float, default=0.0)
    parser.add_argument("--difl-quantile-bins", type=int, default=101)
    parser.add_argument("--difl-micro-bs", type=int, default=128)
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    output_path = report_path(config_path, f"baseline_{args.baseline}", int(args.seed), args.output)
    baseline_log_path = log_path(config_path, f"baseline_{args.baseline}", int(args.seed), args.log_path)
    setup_logger(baseline_log_path)
    apply_baseline_defaults(args)
    set_seed(int(args.seed))

    device = torch.device(args.device)
    loaders = DSIBaselineDataLoaders(
        config["data"]["processed_dir"],
        device,
        int(args.batch_size),
        max_train_rows=args.max_train_rows,
        max_eval_rows=args.max_eval_rows,
        num_workers=int(args.num_workers),
        split_prefix=str(args.split_prefix),
        include_duration_bucket=args.baseline in {"d2q", "gr", "difl"},
        duration_bucket_count=int(args.duration_buckets),
        include_raw_features=str(args.feature_protocol) == "full",
    )
    pre_reference_info: dict[str, Any] = {}
    if str(args.baseline) == "gr":
        pre_reference_info = prepare_gr_vocab_size(args, loaders)
    model = build_baseline(args.baseline, loaders.description, args).to(device)
    adapter = build_adapter(args.baseline, model).to(device)
    reference_info = bind_references(adapter, args.baseline, loaders, device, args)
    reference_info.update(pre_reference_info)

    base_history = train_base(adapter, loaders, args, device)
    base_test = evaluate(adapter, loaders, "test", device)

    report = {
        "method": "baseline",
        "baseline": args.baseline,
        "seed": int(args.seed),
        "config": str(config_path),
        "processed_dir": str(config["data"]["processed_dir"]),
        "split_prefix": str(args.split_prefix),
        "feature_protocol": str(args.feature_protocol),
        "model_continuous_scale": float(getattr(loaders, "model_continuous_scale", 1.0)),
        "reference_info": reference_info,
        "base_history": base_history,
        "base_test": base_test,
    }
    serialized_report = portableize_serialized_paths(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(serialized_report, handle, indent=2, ensure_ascii=False)
    logger.info("Baseline report written. output=%s", output_path)


if __name__ == "__main__":
    main()
