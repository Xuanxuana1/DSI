#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from baselines.readout_probe import (
    _baseline_namespace,
    _baseline_raw_context,
    _labels,
    _load_or_build_native_interface,
    _pad_readout_matrix,
)
from baselines.training import (
    apply_baseline_defaults,
    set_seed,
    setup_logger,
    topk_session_value,
)
from dsi.commands import load_config
from dsi.decision.neutral import binary_auc_score, session_ndcg_metrics
from dsi.decision.features import FeatureSource, _load_content_arrays, _model_features
from dsi.decision.readout import _standardize
from dsi.dataset import load_manifest
from dsi.training import _build_model
from dsi.utils.bridge import portableize_serialized_paths, resolve_path

logger = logging.getLogger(__name__)


def _source_args(source: str) -> argparse.Namespace:
    logger.info("Building the transfer-baseline configuration. source=%s", source)

    namespace = argparse.Namespace(
        baseline=str(source),
        baseline_lr=None,
        baseline_epochs=None,
        weight_decay=1e-6,
        clip_grad_norm=10.0,
        embed_dim=16,
        mlp_dims=[256, 128, 64],
        share_mlp_dims=[256, 128, 64],
        dropout=0.0,
        egmn_components=10,
        egmn_dropout=0.2,
        tpm_bins=32,
        duration_buckets=30,
        d2q_quantile_bins=101,
        cread_heads=50,
        cread_beta=50.0,
        cread_lambda_ce=1.0,
        cread_lambda_restore=1.0,
        cread_lambda_ord=0.00002,
        cread_share_mlp_dims=[256, 128, 64],
        cread_output_dims=[32],
    )
    result = _baseline_namespace(namespace, str(source))
    apply_baseline_defaults(result)
    logger.info("Transfer-baseline configuration built. source=%s", source)
    return result


def _label_payload_from_npz(npz: np.lib.npyio.NpzFile, ratio_threshold: float) -> dict[str, np.ndarray]:
    logger.info("Building transfer labels.")

    event = np.asarray(npz["watch_event"], dtype=np.int64)
    play_time = np.asarray(npz["play_time_sec"], dtype=np.float32).clip(min=0.0)
    duration = np.asarray(npz["duration_sec"], dtype=np.float32).clip(min=1e-6)
    result = {
        "play_time": play_time,
        "duration_sec": duration,
        "completion": (event == 2).astype(np.float32),
        "long": np.isin(event, np.asarray([2, 3], dtype=np.int64)).astype(np.float32),
        "overplay": (event == 3).astype(np.float32),
        "ratio_threshold": ((play_time / duration) > float(ratio_threshold)).astype(np.float32),
        "engagement": np.asarray(npz["engagement_mark"], dtype=np.float32),
        "session_key": np.asarray(npz["session_key"], dtype=np.int64),
    }
    logger.info("Transfer labels built. rows=%d", int(event.shape[0]))
    return result


def _load_dsi_source(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    logger.info("Loading DSI transfer representations.")

    report = json.loads(resolve_path(args.dsi_report).read_text(encoding="utf-8"))
    cache_path = resolve_path(report["metadata"]["feature_cache"]["path"])
    with np.load(cache_path, allow_pickle=False) as cache:
        raw = {split: np.asarray(cache[f"feature_raw_{split}"], dtype=np.float32) for split in ("train", "val", "test")}
        posterior = {split: np.asarray(cache[f"feature_posterior_{split}"], dtype=np.float32) for split in ("train", "val", "test")}

    readout_posterior = posterior
    if bool(args.dsi_drop_direct_marginals):
        readout_posterior = {split: value[:, 10:].astype(np.float32, copy=False) for split, value in posterior.items()}

    matrices = {
        split: _pad_readout_matrix(
            np.concatenate([raw[split], readout_posterior[split]], axis=1).astype(np.float32, copy=False),
            int(args.input_dim),
        )
        for split in ("train", "val", "test")
    }

    zero_scores = {
        "completion": posterior["test"][:, 2].astype(np.float32, copy=False),
        "overplay": posterior["test"][:, 3].astype(np.float32, copy=False),
        "long": posterior["test"][:, 8].astype(np.float32, copy=False),
    }

    config, _ = load_config(args.config)
    processed = Path(config["data"]["processed_dir"])
    labels: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        with np.load(processed / f"standard_{split}.npz", allow_pickle=False) as npz:
            labels[split] = _label_payload_from_npz(npz, float(args.ratio_threshold))
    logger.info("DSI transfer representations loaded. dim=%d", int(matrices["train"].shape[1]))
    return matrices, labels, zero_scores


def _load_raw_source(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], None]:
    logger.info("Loading raw-only transfer representations.")

    report = json.loads(resolve_path(args.dsi_report).read_text(encoding="utf-8"))
    cache_path = resolve_path(report["metadata"]["feature_cache"]["path"])
    with np.load(cache_path, allow_pickle=False) as cache:
        raw = {
            split: np.asarray(cache[f"feature_raw_{split}"], dtype=np.float32)
            for split in ("train", "val", "test")
        }

    matrices = {
        split: _pad_readout_matrix(raw[split], int(args.input_dim))
        for split in ("train", "val", "test")
    }

    config, _ = load_config(args.config)
    processed = Path(config["data"]["processed_dir"])
    labels: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        with np.load(processed / f"standard_{split}.npz", allow_pickle=False) as npz:
            labels[split] = _label_payload_from_npz(npz, float(args.ratio_threshold))

    logger.info("Raw-only transfer representations loaded. raw_dim=%d input_dim=%d", int(raw["train"].shape[1]), int(matrices["train"].shape[1]))
    return matrices, labels, None


def _load_random_dsi_source(args: argparse.Namespace, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], None]:
    logger.info("Loading random-provider transfer representations. seed=%d", int(args.seed))

    report = json.loads(resolve_path(args.dsi_report).read_text(encoding="utf-8"))
    cache_path = resolve_path(report["metadata"]["feature_cache"]["path"])
    with np.load(cache_path, allow_pickle=False) as cache:
        raw = {
            split: np.asarray(cache[f"feature_raw_{split}"], dtype=np.float32)
            for split in ("train", "val", "test")
        }

    config, config_path = load_config(args.config)
    config["training"]["seed"] = int(args.seed)
    processed = Path(config["data"]["processed_dir"])
    manifest = load_manifest(processed)
    content_arrays = _load_content_arrays(processed)
    set_seed(int(args.seed))
    model = _build_model("obs_set_posterior", manifest, config).to(device)
    model.eval()
    source = FeatureSource(
        name="random_dsi",
        model_type="obs_set_posterior",
        config_path=Path(config_path),
        checkpoint_root=Path(config["outputs"]["root_dir"]),
    )

    posterior = {
        split: _model_features(
            source,
            config,
            model,
            manifest,
            f"standard_{split}",
            None,
            int(args.feature_batch_size),
            device,
            content_arrays,
            "compact_27",
            logger,
        )
        for split in ("train", "val", "test")
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    matrices = {
        split: _pad_readout_matrix(
            np.concatenate([raw[split], posterior[split]], axis=1).astype(np.float32, copy=False),
            int(args.input_dim),
        )
        for split in ("train", "val", "test")
    }

    labels: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        with np.load(processed / f"standard_{split}.npz", allow_pickle=False) as npz:
            labels[split] = _label_payload_from_npz(npz, float(args.ratio_threshold))

    logger.info("Random-provider transfer representations loaded. posterior_dim=%d input_dim=%d", int(posterior["train"].shape[1]), int(matrices["train"].shape[1]))
    return matrices, labels, None


def _load_baseline_source(args: argparse.Namespace, source: str, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    logger.info("Loading baseline transfer representations. source=%s", source)

    config, _ = load_config(args.config)
    baseline_args = _source_args(source)
    loaders, representation, zero_scores, _ = _load_or_build_native_interface(
        config,
        args,
        source,
        device,
        baseline_args,
    )

    raw = {split: _baseline_raw_context(loaders[split].dataset, None) for split in ("train", "val", "test")}
    matrices = {
        split: _pad_readout_matrix(
            np.concatenate([raw[split], representation[split]], axis=1).astype(np.float32, copy=False),
            int(args.input_dim),
        )
        for split in ("train", "val", "test")
    }

    labels = {split: _labels(loaders[split].dataset) for split in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        event = labels[split]["watch_event"]
        labels[split]["completion"] = (event == 2).astype(np.float32)
        labels[split]["long"] = np.isin(event, np.asarray([2, 3], dtype=np.int64)).astype(np.float32)
        labels[split]["overplay"] = (event == 3).astype(np.float32)
        labels[split]["ratio_threshold"] = (
            labels[split]["play_time"] / np.maximum(labels[split]["duration_sec"], 1e-6) > float(args.ratio_threshold)
        ).astype(np.float32)
    logger.info("Baseline transfer representations loaded. source=%s dim=%d", source, int(matrices["train"].shape[1]))
    return matrices, labels, zero_scores


class LinearTransferHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.output = nn.Linear(int(input_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(features).squeeze(-1)


class MLPTransferHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(previous_dim, int(hidden_dim)), nn.ReLU()])
            previous_dim = int(hidden_dim)
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _fraction_indices(row_count: int, fraction: float, seed: int) -> np.ndarray | None:
    logger.info("Building the low-sample label set. fraction=%.4f", float(fraction))

    if float(fraction) >= 1.0:
        logger.info("Low-sample label set built. rows=%d", int(row_count))
        return None

    rows = max(1, int(round(float(row_count) * float(fraction))))
    rng = np.random.default_rng(int(seed))
    result = np.sort(rng.choice(int(row_count), size=rows, replace=False).astype(np.int64))
    logger.info("Low-sample label set built. rows=%d", int(rows))
    return result


def _iter_batches(features: np.ndarray, target: np.ndarray, index: np.ndarray | None, batch_size: int, seed: int):
    row_count = int(features.shape[0]) if index is None else int(index.shape[0])
    starts = np.arange(0, row_count, int(batch_size), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(starts)
    for start_value in starts:
        start = int(start_value)
        end = min(start + int(batch_size), row_count)
        lookup = slice(start, end) if index is None else index[start:end]
        yield features[lookup], target[lookup]


@torch.no_grad()


def _predict(model: nn.Module, features: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    for start in range(0, int(features.shape[0]), int(batch_size)):
        batch = torch.as_tensor(features[start : start + int(batch_size)], dtype=torch.float32, device=device)
        parts.append(model(batch).detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(parts, axis=0)


def _fit_transfer_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    index: np.ndarray | None,
    task: str,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    logger.info("Training the transfer head. task=%s head=%s", task, str(args.head))

    set_seed(int(seed))
    if str(args.head) == "linear":
        model = LinearTransferHead(int(train_x.shape[1])).to(device)
    elif str(args.head) == "mlp":
        model = MLPTransferHead(int(train_x.shape[1]), [int(value) for value in args.mlp_hidden_dims]).to(device)
    else:
        raise ValueError(f"Unsupported transfer head: {args.head}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_value = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    trace: list[dict[str, float]] = []

    step = 0
    pass_index = 0
    interval_loss = 0.0
    interval_steps = 0
    while step < int(args.probe_steps):
        pass_index += 1
        for batch_x, batch_y in _iter_batches(train_x, train_y, index, int(args.batch_size), int(seed) + pass_index):
            features = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
            target = torch.as_tensor(batch_y, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            if task == "binary":
                loss = F.binary_cross_entropy_with_logits(prediction, target)
            elif task == "quantile":
                error = target - prediction
                loss = torch.maximum(float(args.quantile) * error, (float(args.quantile) - 1.0) * error).mean()
            else:
                raise ValueError(f"Unsupported transfer task: {task}")
            loss.backward()
            optimizer.step()
            step += 1
            interval_loss += float(loss.detach().cpu().item())
            interval_steps += 1

            should_evaluate = step % int(args.eval_every_steps) == 0 or step >= int(args.probe_steps)
            if should_evaluate:
                val_prediction = _predict(model, val_x, device, int(args.batch_size))
                if task == "binary":
                    val_metric = binary_auc_score(val_y, val_prediction)
                    value = -float(val_metric)
                    metric_name = "auc"
                else:
                    error = val_y - val_prediction
                    val_metric = float(np.maximum(float(args.quantile) * error, (float(args.quantile) - 1.0) * error).mean())
                    value = float(val_metric)
                    metric_name = "pinball_log"
                trace.append(
                    {
                        "step": float(step),
                        "train_loss": float(interval_loss / max(interval_steps, 1)),
                        f"val_{metric_name}": float(val_metric),
                    }
                )
                logger.info(
                    "Transfer-head convergence check. task=%s head=%s step=%d train_loss=%.6f val_%s=%.6f",
                    task,
                    str(args.head),
                    int(step),
                    float(interval_loss / max(interval_steps, 1)),
                    metric_name,
                    float(val_metric),
                )
                interval_loss = 0.0
                interval_steps = 0
                if value < best_value:
                    best_value = float(value)
                    best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            if step >= int(args.probe_steps):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("Transfer-head training complete. task=%s head=%s best=%.6f", task, str(args.head), best_value)
    return model, trace


def _zero_parameter_metrics(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> dict[str, Any]:
    logger.info("Evaluating zero-parameter decision scores.")

    result: dict[str, Any] = {}
    for task in ("long", "overplay"):
        result[task] = {
            "auc": binary_auc_score(labels[task], scores[task]),
            "ndcg@5": session_ndcg_metrics(labels["session_key"], scores[task], labels[task], ks=(5,))["ndcg@5"],
            "top5_positive_rate": topk_session_value(scores[task], labels[task], labels["session_key"], 5)["value"],
        }
    logger.info("Zero-parameter decision-score evaluation complete.")
    return result


def _run_transfer(args: argparse.Namespace, source: str, matrices: dict[str, np.ndarray], labels: dict[str, dict[str, np.ndarray]], zero_scores: dict[str, np.ndarray] | None, device: torch.device) -> dict[str, Any]:
    logger.info("Running the transfer experiment. source=%s", source)

    train_x, val_x, test_x = _standardize(matrices["train"], matrices["val"], matrices["test"])
    zero_result = None if zero_scores is None else _zero_parameter_metrics(zero_scores, labels["test"])
    fractions: dict[str, Any] = {}

    for fraction in [float(value) for value in args.fractions]:
        key = f"{fraction:.2f}"
        index = _fraction_indices(int(train_x.shape[0]), float(fraction), int(args.seed))
        fraction_result: dict[str, Any] = {
            "train_rows": int(train_x.shape[0] if index is None else index.shape[0]),
        }
        if bool(args.engagement_probe):
            engagement_model, engagement_trace = _fit_transfer_probe(
                train_x,
                labels["train"]["engagement"],
                val_x,
                labels["val"]["engagement"],
                index,
                "binary",
                args,
                device,
                int(args.seed) + 307,
            )
            engagement_score = _predict(engagement_model, test_x, device, int(args.batch_size))
            fraction_result["engagement"] = {
                "auc": binary_auc_score(labels["test"]["engagement"], engagement_score),
                "ndcg@5": session_ndcg_metrics(
                    labels["test"]["session_key"],
                    engagement_score,
                    labels["test"]["engagement"],
                    ks=(5,),
                )["ndcg@5"],
                "convergence": engagement_trace,
            }
        if bool(args.engagement_only):
            fractions[key] = fraction_result
            continue
        ratio_model, ratio_trace = _fit_transfer_probe(
            train_x,
            labels["train"]["ratio_threshold"],
            val_x,
            labels["val"]["ratio_threshold"],
            index,
            "binary",
            args,
            device,
            int(args.seed) + 101,
        )
        ratio_score = _predict(ratio_model, test_x, device, int(args.batch_size))

        train_log = np.log1p(labels["train"]["play_time"]).astype(np.float32, copy=False)
        val_log = np.log1p(labels["val"]["play_time"]).astype(np.float32, copy=False)
        quantile_model, quantile_trace = _fit_transfer_probe(
            train_x,
            train_log,
            val_x,
            val_log,
            index,
            "quantile",
            args,
            device,
            int(args.seed) + 211,
        )
        quantile_log = _predict(quantile_model, test_x, device, int(args.batch_size))
        target_log = np.log1p(labels["test"]["play_time"]).astype(np.float32, copy=False)
        quantile_error = target_log - quantile_log
        quantile_sec = np.expm1(np.clip(quantile_log, 0.0, 8.0)).astype(np.float32, copy=False)

        fraction_result.update({
            "ratio_threshold": {
                "threshold": float(args.ratio_threshold),
                "auc": binary_auc_score(labels["test"]["ratio_threshold"], ratio_score),
                "ndcg@5": session_ndcg_metrics(
                    labels["test"]["session_key"],
                    ratio_score,
                    labels["test"]["ratio_threshold"],
                    ks=(5,),
                )["ndcg@5"],
                "top5_positive_rate": topk_session_value(
                    ratio_score,
                    labels["test"]["ratio_threshold"],
                    labels["test"]["session_key"],
                    5,
                )["value"],
                "convergence": ratio_trace,
            },
            "quantile": {
                "q": float(args.quantile),
                "pinball_log": float(
                    np.maximum(
                        float(args.quantile) * quantile_error,
                        (float(args.quantile) - 1.0) * quantile_error,
                    ).mean()
                ),
                "mae_sec": float(np.abs(quantile_sec - labels["test"]["play_time"]).mean()),
                "convergence": quantile_trace,
            },
        })
        fractions[key] = fraction_result

    logger.info("Transfer experiment complete. source=%s", source)
    return {"source": source, "input_dim": int(train_x.shape[1]), "zero_parameter": zero_result, "fractions": fractions}


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-representation transfer probe.")
    parser.add_argument("--config", default="configs/kuairec.yaml")
    parser.add_argument("--source", choices=["raw", "random_dsi", "dsi", "d2q", "tpm", "cread", "egmn"], required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--dsi-report", default="artifacts/reports/kuairec_decision_seed2027.json")
    parser.add_argument("--dsi-drop-direct-marginals", action="store_true")
    parser.add_argument("--input-dim", type=int, default=82)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.10, 1.00])
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument("--ratio-threshold", type=float, default=0.5)
    parser.add_argument("--head", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--mlp-hidden-dims", nargs="+", type=int, default=[128])
    parser.add_argument("--probe-steps", type=int, default=6000)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--baseline-batch-size", type=int, default=4096)
    parser.add_argument("--feature-batch-size", type=int, default=8192)
    parser.add_argument("--baseline-train-rows", type=int, default=None)
    parser.add_argument("--baseline-eval-rows", type=int, default=None)
    parser.add_argument("--split-prefix", choices=["standard", "random"], default="standard")
    parser.add_argument("--feature-protocol", choices=["clean", "full"], default="clean")
    parser.add_argument("--duration-buckets", type=int, default=30)
    parser.add_argument("--interface-cache-dir", default="artifacts/cache/baseline_interfaces")
    parser.add_argument("--refresh-interface-cache", action="store_true")
    parser.add_argument("--engagement-probe", action="store_true")
    parser.add_argument("--engagement-only", action="store_true")
    args = parser.parse_args()

    if int(args.probe_steps) <= 0:
        raise ValueError("--probe-steps must be positive")
    if int(args.eval_every_steps) <= 0:
        raise ValueError("--eval-every-steps must be positive")
    if str(args.head) == "mlp" and (not args.mlp_hidden_dims or any(int(value) <= 0 for value in args.mlp_hidden_dims)):
        raise ValueError("--mlp-hidden-dims must contain positive dimensions")
    if bool(args.engagement_only) and not bool(args.engagement_probe):
        raise ValueError("--engagement-only requires --engagement-probe")

    setup_logger(None if args.log_path is None else resolve_path(args.log_path))
    logger.info("Running the transfer probe. source=%s seed=%d", args.source, int(args.seed))

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.source == "raw":
        matrices, labels, zero_scores = _load_raw_source(args)
    elif args.source == "random_dsi":
        matrices, labels, zero_scores = _load_random_dsi_source(args, device)
    elif args.source == "dsi":
        matrices, labels, zero_scores = _load_dsi_source(args)
    else:
        matrices, labels, zero_scores = _load_baseline_source(args, str(args.source), device)

    result = {
        "method": "frozen_representation_transfer_probe",
        "seed": int(args.seed),
        "config": str(resolve_path(args.config)),
        "protocol": {
            "fractions": [float(value) for value in args.fractions],
            "quantile": float(args.quantile),
            "ratio_threshold": float(args.ratio_threshold),
            "probe_steps": int(args.probe_steps),
            "eval_every_steps": int(args.eval_every_steps),
            "lr": float(args.lr),
            "input_dim": int(args.input_dim),
            "head": str(args.head),
            "mlp_hidden_dims": [int(value) for value in args.mlp_hidden_dims] if str(args.head) == "mlp" else None,
            "dsi_drop_direct_marginals": bool(args.dsi_drop_direct_marginals),
            "interface_cache_dir": str(args.interface_cache_dir),
        },
        "result": _run_transfer(args, str(args.source), matrices, labels, zero_scores, device),
    }
    serialized_result = portableize_serialized_paths(result)
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serialized_result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Transfer probe complete. output=%s", output_path)
    print(json.dumps(serialized_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
