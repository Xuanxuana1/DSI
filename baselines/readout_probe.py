#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from baselines.adapter import build_adapter
from baselines.data import DSIBaselineDataLoaders, DSIBaselineDataset
from baselines.training import (
    apply_baseline_defaults,
    bind_references,
    build_baseline,
    evaluate,
    set_seed,
    setup_logger,
    train_base,
)
from baselines.utils import duration_bucket, quantile_to_value
from dsi.commands import load_config, log_path, report_path
from dsi.dataset import batch_to_device, load_manifest
from dsi.decision.features import _raw_features, _sample_train_indices
from dsi.decision.readout import _run_feature_set, _seed_offset
from dsi.models import GenericLatentControlModel, build_shape
from dsi.training import _build_optimizers, _make_loaders
from dsi.utils.bridge import portableize_serialized_paths

logger = logging.getLogger(__name__)


def _add_baseline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baselines", nargs="+", choices=["d2q", "tpm", "cread", "egmn", "generic"], default=["d2q", "tpm", "cread", "egmn"])
    parser.add_argument("--baseline-batch-size", type=int, default=4096)
    parser.add_argument("--baseline-train-rows", type=int, default=None)
    parser.add_argument("--baseline-eval-rows", type=int, default=None)
    parser.add_argument("--baseline-epochs", type=int, default=None)
    parser.add_argument("--baseline-lr", type=float, default=None)
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
    parser.add_argument("--cread-heads", type=int, default=50)
    parser.add_argument("--cread-beta", type=float, default=50.0)
    parser.add_argument("--cread-lambda-ce", type=float, default=1.0)
    parser.add_argument("--cread-lambda-restore", type=float, default=1.0)
    parser.add_argument("--cread-lambda-ord", type=float, default=0.00002)
    parser.add_argument("--cread-share-mlp-dims", nargs="+", type=int, default=[256, 128, 64])
    parser.add_argument("--cread-output-dims", nargs="+", type=int, default=[32])
    parser.add_argument("--feature-protocol", choices=["clean", "full"], default="clean")
    parser.add_argument("--interface-cache-dir", default="artifacts/cache/baseline_interfaces")
    parser.add_argument("--refresh-interface-cache", action="store_true")


def _add_readout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-rows", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--feature-batch-size", type=int, default=8192)
    parser.add_argument("--scorer", choices=["mlp", "dcn"], default="dcn")
    parser.add_argument("--rank-target", choices=["long", "overplay"], default="overplay")
    parser.add_argument("--validation-select", choices=["mae", "xauc", "rank_auc"], default="mae")
    parser.add_argument("--rank-loss-weight", type=float, default=0.5)
    parser.add_argument("--ordinal-bins", type=int, default=30)
    parser.add_argument("--ordinal-beta", type=float, default=3.0)
    parser.add_argument("--ordinal-restore-weight", type=float, default=30.0)
    parser.add_argument("--ordinal-ce-weight", type=float, default=0.0)
    parser.add_argument("--ordinal-prior-weight", type=float, default=0.0)
    parser.add_argument("--scorer-hidden-dim", type=int, default=128)
    parser.add_argument("--scorer-hidden-layers", type=int, default=2)
    parser.add_argument("--scorer-cross-layers", type=int, default=2)
    parser.add_argument("--scorer-dropout", type=float, default=0.1)
    parser.add_argument("--scorer-epochs", type=int, default=5)
    parser.add_argument("--scorer-lr", type=float, default=0.001)
    parser.add_argument("--scorer-weight-decay", type=float, default=0.0001)
    parser.add_argument("--scorer-clip-grad-norm", type=float, default=10.0)
    parser.add_argument("--topks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--fast-eval", action="store_true")
    parser.add_argument("--fast-eval-rows", type=int, default=200000)
    parser.add_argument("--readout-input-dim", type=int, default=None)
    parser.add_argument("--representation-only", action="store_true")
    parser.add_argument("--representation-protocol", choices=["hidden_scalar", "native_distribution"], default="hidden_scalar")


def _baseline_namespace(args: argparse.Namespace, baseline: str) -> argparse.Namespace:
    return argparse.Namespace(
        baseline=str(baseline),
        lr=args.baseline_lr,
        epochs=args.baseline_epochs,
        weight_decay=float(args.weight_decay),
        clip_grad_norm=float(args.clip_grad_norm),
        embed_dim=int(args.embed_dim),
        mlp_dims=[int(value) for value in args.mlp_dims],
        share_mlp_dims=[int(value) for value in args.share_mlp_dims],
        dropout=float(args.dropout),
        egmn_components=int(args.egmn_components),
        egmn_dropout=float(args.egmn_dropout),
        tpm_bins=int(args.tpm_bins),
        duration_buckets=int(args.duration_buckets),
        d2q_quantile_bins=int(args.d2q_quantile_bins),
        cread_heads=int(args.cread_heads),
        cread_beta=float(args.cread_beta),
        cread_lambda_ce=float(args.cread_lambda_ce),
        cread_lambda_restore=float(args.cread_lambda_restore),
        cread_lambda_ord=float(args.cread_lambda_ord),
        cread_share_mlp_dims=[int(value) for value in args.cread_share_mlp_dims],
        cread_output_dims=[int(value) for value in args.cread_output_dims],
    )


def _slice_features(dataset: DSIBaselineDataset, index: slice | np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "user_id": torch.as_tensor(dataset.user_id[index], dtype=torch.long, device=device),
        "video_id": torch.as_tensor(dataset.video_id[index], dtype=torch.long, device=device),
        "tab": torch.as_tensor(dataset.tab[index], dtype=torch.long, device=device),
        "prev_event": torch.as_tensor(dataset.prev_event[index], dtype=torch.long, device=device),
        "duration": torch.as_tensor(dataset.duration[index], dtype=torch.float32, device=device),
        "duration_model": torch.as_tensor(dataset.duration_model[index], dtype=torch.float32, device=device),
        "duration_bucket": torch.as_tensor(dataset.duration_bucket[index], dtype=torch.long, device=device),
        "ctn_model": torch.as_tensor(dataset.ctn_model[index], dtype=torch.float32, device=device),
        "raw_sparse": torch.as_tensor(dataset.raw_sparse[index], dtype=torch.long, device=device),
        "raw_seq": torch.as_tensor(dataset.raw_seq[index], dtype=torch.long, device=device),
        "raw_seq_mask": torch.as_tensor(dataset.raw_seq_mask[index], dtype=torch.float32, device=device),
        "content_category": torch.as_tensor(dataset.content_category[index], dtype=torch.long, device=device),
        "event": torch.as_tensor(dataset.event[index], dtype=torch.long, device=device),
        "session_key": torch.as_tensor(dataset.session_key[index], dtype=torch.long, device=device),
    }


def _iter_index(index: np.ndarray | None, row_count: int, batch_size: int):
    if index is None:
        for start in range(0, int(row_count), int(batch_size)):
            yield slice(start, min(start + int(batch_size), int(row_count)))
        return
    for start in range(0, int(index.shape[0]), int(batch_size)):
        yield index[start : start + int(batch_size)]


def _labels(dataset: DSIBaselineDataset, index: np.ndarray | None = None) -> dict[str, np.ndarray]:
    lookup = slice(None) if index is None else index
    event = np.asarray(dataset.event[lookup], dtype=np.int64)
    return {
        "play_time": np.asarray(dataset.play_time[lookup], dtype=np.float32).clip(min=0.0),
        "duration_sec": np.asarray(dataset.duration[lookup], dtype=np.float32).clip(min=0.0),
        "watch_event": event,
        "session_key": np.asarray(dataset.session_key[lookup], dtype=np.int64),
        "engagement": np.asarray(dataset.engagement_mark[lookup], dtype=np.float32),
        "long_label": np.isin(event, np.asarray([2, 3], dtype=np.int64)).astype(np.float32),
        "overplay_label": (event == 3).astype(np.float32),
    }


def _baseline_raw_context(dataset: DSIBaselineDataset, index: np.ndarray | None) -> np.ndarray:
    logger.info("Building baseline raw context.")

    arrays = {
        "numeric": dataset.numeric,
        "session_numeric": dataset.session_numeric,
        "duration_sec": dataset.duration,
        "tab": dataset.tab,
        "prev_event": dataset.prev_event,
        "session_key": dataset.session_key,
    }

    raw_context, _ = _raw_features(arrays, index)
    logger.info("Baseline raw context built. dim=%d", int(raw_context.shape[1]))
    return raw_context


def _pad_readout_matrix(matrix: np.ndarray, target_dim: int | None) -> np.ndarray:
    logger.info("Aligning readout input width. current=%d target=%s", int(matrix.shape[1]), target_dim)

    if target_dim is None:
        logger.info("Readout input width already aligned.")
        return matrix

    target = int(target_dim)
    current = int(matrix.shape[1])
    if target < current:
        raise ValueError(f"--readout-input-dim={target} is smaller than feature dim {current}")

    if target == current:
        output = matrix
    else:
        padding = np.zeros((matrix.shape[0], target - current), dtype=np.float32)
        output = np.concatenate([matrix, padding], axis=1).astype(np.float32, copy=False)
    logger.info("Readout input width aligned. dim=%d", int(output.shape[1]))
    return output


@torch.no_grad()


def _collect_representation(
    adapter,
    dataset: DSIBaselineDataset,
    index: np.ndarray | None,
    batch_size: int,
    device: torch.device,
    name: str,
) -> np.ndarray:
    row_count = len(dataset) if index is None else int(index.shape[0])
    logger.info("Extracting baseline representations. name=%s rows=%d", name, row_count)
    adapter.eval()
    parts: list[np.ndarray] = []
    for batch_index in _iter_index(index, len(dataset), int(batch_size)):
        features = _slice_features(dataset, batch_index, device)
        p, proxy, hidden = adapter.get_base_pred_and_hidden(features)
        columns = [
            p.detach().cpu().numpy().astype(np.float32, copy=False),
            torch.log1p(proxy.clamp(min=0.0)).detach().cpu().numpy().astype(np.float32, copy=False),
        ]
        if hidden is not None:
            columns.append(hidden.detach().cpu().numpy().astype(np.float32, copy=False))
        parts.append(np.concatenate(columns, axis=1))
    output = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    logger.info("Baseline representations extracted. name=%s shape=%s", name, tuple(output.shape))
    return output


def _tpm_leaf_probabilities(node_prob: torch.Tensor, interval_count: int) -> torch.Tensor:
    height = int(np.log2(int(interval_count)))
    leaf_parts: list[torch.Tensor] = []

    for leaf_index in range(int(interval_count)):
        log_prob = torch.zeros(node_prob.shape[0], dtype=node_prob.dtype, device=node_prob.device)
        current_code = 2**height - 1 + int(leaf_index)
        for _ in range(height):
            branch = current_code % 2
            classifier_index = (current_code - 1) // 2
            probability = node_prob[:, classifier_index].clamp(1e-6, 1.0 - 1e-6)
            log_prob = log_prob + (torch.log1p(-probability) if branch == 1 else torch.log(probability))
            current_code = classifier_index
        leaf_parts.append(log_prob)
    return torch.exp(torch.stack(leaf_parts, dim=1))


def _cread_survival_at(adapter, ordinal_prob: torch.Tensor, threshold_sec: torch.Tensor) -> torch.Tensor:
    if adapter.split_nodes is None:
        raise RuntimeError("CREAD split nodes are not initialized")

    nodes = adapter.split_nodes.to(device=ordinal_prob.device, dtype=ordinal_prob.dtype) * float(adapter.label_scale)
    threshold = threshold_sec.reshape(-1).to(device=ordinal_prob.device, dtype=ordinal_prob.dtype)
    right = torch.searchsorted(nodes, threshold).clamp(min=0, max=nodes.numel() - 1)
    left = (right - 1).clamp(min=0)
    row = torch.arange(threshold.shape[0], device=ordinal_prob.device)

    left_node = nodes.index_select(0, left)
    right_node = nodes.index_select(0, right)
    left_prob = ordinal_prob[row, left]
    right_prob = ordinal_prob[row, right]
    weight = ((threshold - left_node) / (right_node - left_node).clamp_min(1e-6)).clamp(0.0, 1.0)
    interpolated = left_prob + weight * (right_prob - left_prob)
    return interpolated.clamp(0.0, 1.0)


def _egmn_survival_at(
    adapter,
    mixture_logits: torch.Tensor,
    exponential_rate: torch.Tensor,
    gaussian_mean: torch.Tensor,
    gaussian_scale: torch.Tensor,
    threshold_sec: torch.Tensor,
) -> torch.Tensor:

    threshold = threshold_sec.reshape(-1, 1).to(device=mixture_logits.device, dtype=mixture_logits.dtype)
    threshold = threshold / max(float(adapter.label_scale), 1e-6)
    mixture_prob = torch.softmax(mixture_logits, dim=1)

    exponential_survival = torch.exp(-exponential_rate * threshold).clamp(0.0, 1.0)
    normal = torch.distributions.Normal(gaussian_mean, gaussian_scale.clamp_min(1e-6))
    zero = torch.zeros_like(gaussian_mean)
    normalizer = (1.0 - normal.cdf(zero)).clamp_min(1e-6)
    gaussian_survival = ((1.0 - normal.cdf(threshold)) / normalizer).clamp(0.0, 1.0)
    component_survival = torch.cat([exponential_survival, gaussian_survival], dim=1)
    return (mixture_prob * component_survival).sum(dim=1).clamp(0.0, 1.0)


def _native_interface_batch(adapter, features: dict[str, torch.Tensor], name: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    duration = features["duration"].reshape(-1, 1).float().clamp_min(1e-3)
    long_threshold = duration.reshape(-1) * 0.9
    overplay_threshold = duration.reshape(-1) * 1.1

    if name == "d2q":
        if adapter.duration_edges is None or adapter.quantiles is None:
            raise RuntimeError("D2Q quantiles are not initialized")
        quantile_score, _ = adapter.model.forward_with_hidden(features)
        bucket = duration_bucket(features["duration"], adapter.duration_edges)
        proxy = quantile_to_value(quantile_score, bucket, adapter.quantiles).clamp_min(0.0)
        ratio_score = (proxy / duration).clamp(0.0, 20.0)
        representation = torch.cat([quantile_score.reshape(-1, 1), torch.log1p(proxy), ratio_score], dim=1)
        return representation, {"long": ratio_score.reshape(-1), "overplay": ratio_score.reshape(-1)}

    if name == "tpm":
        if adapter.bucket_begins is None or adapter.bucket_ends is None:
            raise RuntimeError("TPM buckets are not initialized")
        node_prob, _ = adapter.model.forward_with_hidden(features)
        leaf_prob = _tpm_leaf_probabilities(node_prob, int(adapter.bucket_num))
        leaf_value = ((adapter.bucket_begins + adapter.bucket_ends) / 2.0).to(device=leaf_prob.device, dtype=leaf_prob.dtype)
        proxy = (leaf_prob * leaf_value).sum(dim=1, keepdim=True).clamp_min(0.0)
        ratio_score = (proxy / duration).clamp(0.0, 20.0)
        representation = torch.cat([leaf_prob, torch.log1p(proxy), ratio_score], dim=1)
        return representation, {
            "long": (leaf_prob * (leaf_value > long_threshold.reshape(-1, 1)).to(leaf_prob.dtype)).sum(dim=1),
            "overplay": (leaf_prob * (leaf_value > overplay_threshold.reshape(-1, 1)).to(leaf_prob.dtype)).sum(dim=1),
        }

    if name == "cread":
        if adapter.split_nodes is None:
            raise RuntimeError("CREAD split nodes are not initialized")
        ordinal_prob, _ = adapter.model.forward_with_hidden(features)
        normalized_proxy = torch.sum(
            ordinal_prob
            * torch.diff(
                torch.cat([adapter.split_nodes.new_tensor([0.0]), adapter.split_nodes])
            ).reshape(1, -1),
            dim=1,
            keepdim=True,
        )
        proxy = (normalized_proxy * float(adapter.label_scale)).clamp_min(0.0)
        ratio_score = (proxy / duration).clamp(0.0, 20.0)
        representation = torch.cat([ordinal_prob, torch.log1p(proxy), ratio_score], dim=1)
        return representation, {
            "long": _cread_survival_at(adapter, ordinal_prob, long_threshold),
            "overplay": _cread_survival_at(adapter, ordinal_prob, overplay_threshold),
        }

    if name == "egmn":
        mixture_logits, exponential_rate, gaussian_mean, gaussian_scale, _ = adapter.model.forward_with_hidden(features)
        mixture_prob = torch.softmax(mixture_logits, dim=1)
        component_mean = torch.cat([1.0 / exponential_rate.clamp_min(1e-6), gaussian_mean], dim=1)
        normalized_proxy = (mixture_prob * component_mean).sum(dim=1, keepdim=True)
        proxy = (normalized_proxy * float(adapter.label_scale)).clamp_min(0.0)
        ratio_score = (proxy / duration).clamp(0.0, 20.0)
        representation = torch.cat(
            [
                mixture_prob,
                torch.log1p(exponential_rate),
                torch.log1p(gaussian_mean.clamp_min(0.0)),
                torch.log1p(gaussian_scale.clamp_min(0.0)),
                torch.log1p(proxy),
                ratio_score,
            ],
            dim=1,
        )
        return representation, {
            "long": _egmn_survival_at(adapter, mixture_logits, exponential_rate, gaussian_mean, gaussian_scale, long_threshold),
            "overplay": _egmn_survival_at(adapter, mixture_logits, exponential_rate, gaussian_mean, gaussian_scale, overplay_threshold),
        }

    raise ValueError(f"Native distribution interface is unsupported for baseline: {name}")


@torch.no_grad()


def _collect_native_interface(
    adapter,
    dataset: DSIBaselineDataset,
    batch_size: int,
    device: torch.device,
    name: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    logger.info("Extracting the baseline native interface. name=%s rows=%d", name, len(dataset))

    adapter.eval()
    representation_parts: list[np.ndarray] = []
    score_parts: dict[str, list[np.ndarray]] = {"long": [], "overplay": []}
    for batch_number, batch_index in enumerate(_iter_index(None, len(dataset), int(batch_size)), start=1):
        features = _slice_features(dataset, batch_index, device)
        representation, scores = _native_interface_batch(adapter, features, name)
        representation_parts.append(representation.detach().cpu().numpy().astype(np.float32, copy=False))
        for task in score_parts:
            score_parts[task].append(scores[task].detach().cpu().numpy().astype(np.float32, copy=False))
        if batch_number % 200 == 0:
            logger.info("Baseline native-interface extraction progress. name=%s batches=%d", name, batch_number)

    output = np.concatenate(representation_parts, axis=0).astype(np.float32, copy=False)
    scores = {task: np.concatenate(parts, axis=0).astype(np.float32, copy=False) for task, parts in score_parts.items()}
    logger.info("Baseline native interface extracted. name=%s shape=%s", name, tuple(output.shape))
    return output, scores


def _interface_cache_path(config: dict[str, Any], args: argparse.Namespace, baseline: str) -> Path:
    root = Path(str(args.interface_cache_dir)).expanduser()
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    processed_name = Path(str(config["data"]["processed_dir"])).name
    train_tag = "all" if args.baseline_train_rows is None else str(int(args.baseline_train_rows))
    eval_tag = "all" if args.baseline_eval_rows is None else str(int(args.baseline_eval_rows))
    epoch_tag = 5 if getattr(args, "baseline_epochs", None) is None else int(args.baseline_epochs)
    learning_rate = 0.01 if getattr(args, "baseline_lr", None) is None else float(args.baseline_lr)
    learning_rate_tag = f"{learning_rate:.6g}".replace(".", "p")
    return root / (
        f"{processed_name}_{baseline}_seed{int(args.seed)}_{args.feature_protocol}_"
        f"train{train_tag}_eval{eval_tag}_epoch{epoch_tag}_lr{learning_rate_tag}_native_v1"
    )


def _load_native_interface_cache(cache_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]] | None:
    logger.info("Reading the baseline native-interface cache. path=%s", cache_path)

    metadata_path = cache_path / "metadata.json"
    required = [cache_path / f"representation_{split}.npy" for split in ("train", "val", "test")]
    required.extend(cache_path / f"score_{task}_test.npy" for task in ("long", "overplay"))
    if not metadata_path.exists() or not all(path.exists() for path in required):
        logger.info("Baseline native-interface cache miss.")
        return None

    representation = {split: np.load(cache_path / f"representation_{split}.npy", mmap_mode="r") for split in ("train", "val", "test")}
    scores = {task: np.load(cache_path / f"score_{task}_test.npy", mmap_mode="r") for task in ("long", "overplay")}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    logger.info("Baseline native-interface cache loaded. dim=%d", int(representation["train"].shape[1]))
    return representation, scores, metadata


def _save_native_interface_cache(
    cache_path: Path,
    representation: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    logger.info("Saving the baseline native-interface cache. path=%s", cache_path)

    cache_path.mkdir(parents=True, exist_ok=True)
    for split, value in representation.items():
        np.save(cache_path / f"representation_{split}.npy", np.asarray(value, dtype=np.float32), allow_pickle=False)
    for task, value in scores.items():
        np.save(cache_path / f"score_{task}_test.npy", np.asarray(value, dtype=np.float32), allow_pickle=False)

    serialized_metadata = portableize_serialized_paths(metadata)
    (cache_path / "metadata.json").write_text(
        json.dumps(serialized_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Baseline native-interface cache saved.")


def _load_or_build_native_interface(
    config: dict[str, Any],
    args: argparse.Namespace,
    baseline: str,
    device: torch.device,
    baseline_args: argparse.Namespace | None = None,
) -> tuple[DSIBaselineDataLoaders, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    logger.info("Preparing the baseline native interface. baseline=%s", baseline)

    loaders = DSIBaselineDataLoaders(
        config["data"]["processed_dir"],
        device,
        int(args.baseline_batch_size),
        max_train_rows=args.baseline_train_rows,
        max_eval_rows=args.baseline_eval_rows,
        split_prefix=str(args.split_prefix),
        include_duration_bucket=baseline == "d2q",
        duration_bucket_count=int(args.duration_buckets),
        include_raw_features=str(args.feature_protocol) == "full",
    )
    cache_path = _interface_cache_path(config, args, baseline)
    cached = None if bool(args.refresh_interface_cache) else _load_native_interface_cache(cache_path)
    if cached is not None:
        representation, scores, metadata = cached
        logger.info("Baseline native interface prepared. baseline=%s source=cache", baseline)
        return loaders, representation, scores, metadata

    provider_args = baseline_args if baseline_args is not None else _baseline_namespace(args, baseline)
    apply_baseline_defaults(provider_args)
    model = build_baseline(baseline, loaders.description, provider_args).to(device)
    adapter = build_adapter(baseline, model).to(device)
    reference_info = bind_references(adapter, baseline, loaders, device, provider_args)
    base_history = train_base(adapter, loaders, provider_args, device)

    representation: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        split_representation, split_scores = _collect_native_interface(
            adapter,
            loaders[split].dataset,
            int(args.feature_batch_size),
            device,
            baseline,
        )
        representation[split] = split_representation
        if split == "test":
            test_scores = split_scores

    metadata = {
        "baseline": baseline,
        "seed": int(args.seed),
        "feature_protocol": str(args.feature_protocol),
        "representation_protocol": "native_distribution",
        "representation_dim": int(representation["train"].shape[1]),
        "reference_info": reference_info,
        "base_history": base_history,
        "cache_path": str(cache_path),
    }
    _save_native_interface_cache(cache_path, representation, test_scores, metadata)
    loaded = _load_native_interface_cache(cache_path)
    if loaded is None:
        raise RuntimeError(f"Failed to reload native interface cache: {cache_path}")
    cached_representation, cached_scores, cached_metadata = loaded
    logger.info("Baseline native interface prepared. baseline=%s source=trained", baseline)
    return loaders, cached_representation, cached_scores, cached_metadata


def _readout_payload(loaders: DSIBaselineDataLoaders, train_index: np.ndarray | None) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    train = _labels(loaders["train"].dataset, train_index)
    val = _labels(loaders["val"].dataset)
    test = _labels(loaders["test"].dataset)
    labels = {
        "long": {"train": train["long_label"], "val": val["long_label"], "test": test["long_label"]},
        "overplay": {"train": train["overplay_label"], "val": val["overplay_label"], "test": test["overplay_label"]},
    }
    play_time = {"train": train["play_time"], "val": val["play_time"], "test": test["play_time"]}
    return labels, play_time, test


def _run_one_baseline(config: dict[str, Any], args: argparse.Namespace, baseline: str, device: torch.device) -> dict[str, Any]:
    logger.info("Running the controlled baseline readout experiment. baseline=%s protocol=%s", baseline, args.representation_protocol)

    baseline_args = _baseline_namespace(args, baseline)
    apply_baseline_defaults(baseline_args)
    if str(args.representation_protocol) == "native_distribution":
        loaders, full_representation, _, interface_metadata = _load_or_build_native_interface(
            config,
            args,
            baseline,
            device,
            baseline_args,
        )
        train_dataset = loaders["train"].dataset
        train_index = _sample_train_indices(len(train_dataset), int(args.train_rows), int(args.seed), logger)
        representation = {
            "train": np.asarray(
                full_representation["train"] if train_index is None else full_representation["train"][train_index],
                dtype=np.float32,
            ),
            "val": np.asarray(full_representation["val"], dtype=np.float32),
            "test": np.asarray(full_representation["test"], dtype=np.float32),
        }
        reference_info = dict(interface_metadata.get("reference_info", {}))
        base_history = list(interface_metadata.get("base_history", []))
        base_test = None
        cache_path = str(interface_metadata.get("cache_path", ""))
        representation_columns = ["native_distribution", "log1p_proxy", "duration_ratio_proxy"]
    else:
        loaders = DSIBaselineDataLoaders(
            config["data"]["processed_dir"],
            device,
            int(args.baseline_batch_size),
            max_train_rows=args.baseline_train_rows,
            max_eval_rows=args.baseline_eval_rows,
            split_prefix=str(args.split_prefix),
            include_duration_bucket=baseline == "d2q",
            duration_bucket_count=int(args.duration_buckets),
            include_raw_features=str(args.feature_protocol) == "full",
        )
        model = build_baseline(baseline, loaders.description, baseline_args).to(device)
        adapter = build_adapter(baseline, model).to(device)
        reference_info = bind_references(adapter, baseline, loaders, device, baseline_args)
        base_history = train_base(adapter, loaders, baseline_args, device)
        base_test = evaluate(adapter, loaders, "test", device)
        train_dataset = loaders["train"].dataset
        train_index = _sample_train_indices(len(train_dataset), int(args.train_rows), int(args.seed), logger)
        representation = {
            "train": _collect_representation(adapter, train_dataset, train_index, int(args.feature_batch_size), device, baseline),
            "val": _collect_representation(adapter, loaders["val"].dataset, None, int(args.feature_batch_size), device, baseline),
            "test": _collect_representation(adapter, loaders["test"].dataset, None, int(args.feature_batch_size), device, baseline),
        }
        cache_path = None
        representation_columns = ["p_duration_proxy", "log1p_proxy", "hidden"]

    logger.info("Building baseline readout inputs. representation_only=%s", bool(args.representation_only))

    raw_context = None
    if not bool(args.representation_only):
        raw_context = {
            "train": _baseline_raw_context(train_dataset, train_index),
            "val": _baseline_raw_context(loaders["val"].dataset, None),
            "test": _baseline_raw_context(loaders["test"].dataset, None),
        }

    matrices = {
        split: _pad_readout_matrix(
            representation[split]
            if raw_context is None
            else np.concatenate([raw_context[split], representation[split]], axis=1).astype(np.float32, copy=False),
            args.readout_input_dim,
        )
        for split in ["train", "val", "test"]
    }
    logger.info("Baseline readout inputs built. dim=%d", int(matrices["train"].shape[1]))
    labels, play_time, eval_arrays = _readout_payload(loaders, train_index)

    seed_feature_key = "posterior" if bool(args.representation_only) else "fused"
    set_seed(int(args.seed) + _seed_offset(str(args.scorer), seed_feature_key))
    readout = _run_feature_set(
        str(args.scorer),
        baseline,
        matrices,
        labels,
        play_time,
        None,
        eval_arrays,
        device,
        args,
        logger,
    )
    logger.info("Controlled baseline readout experiment complete. baseline=%s", baseline)
    return {
        "baseline": baseline,
        "reference_info": reference_info,
        "base_history": base_history,
        "base_test": base_test,
        "representation": {
            "columns": (["raw_context"] if raw_context is not None else []) + representation_columns,
            "train_rows": int(matrices["train"].shape[0]),
            "raw_feature_dim": 0 if raw_context is None else int(raw_context["train"].shape[1]),
            "native_representation_dim": int(representation["train"].shape[1]),
            "feature_dim": int(matrices["train"].shape[1]),
            "readout_input_dim": int(matrices["train"].shape[1]),
            "representation_only": bool(args.representation_only),
            "representation_protocol": str(args.representation_protocol),
            "interface_cache": cache_path,
        },
        "readout": readout,
    }


def _build_generic_encoder(config: dict[str, Any], manifest: dict[str, Any]) -> GenericLatentControlModel:
    logger.info("Building the generic encoder.")

    model_cfg = config["model"]
    model = GenericLatentControlModel(
        shape=build_shape(manifest),
        embedding_dim=int(model_cfg["embedding_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        sparse_embeddings=bool(model_cfg.get("sparse_embeddings", False)),
        latent_dim=27,
        mlp_hidden_layers=int(model_cfg.get("mlp_hidden_layers", 2)),
        use_raw_features=bool(model_cfg.get("use_raw_features", True)),
    )

    logger.info("Generic encoder built.")
    return model


def _generic_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], restore_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    event_target = batch["watch_event"].long().clamp(max=output["event_logits"].shape[1] - 1)
    event_loss = F.cross_entropy(output["event_logits"], event_target)

    target_log_time = torch.log1p(batch["play_time_sec"].float().clamp_min(0.0))
    restore_loss = F.smooth_l1_loss(output["pred_log_time"], target_log_time)
    total = event_loss + float(restore_weight) * restore_loss
    return total, event_loss.detach(), restore_loss.detach()


@torch.no_grad()


def _generic_validation_loss(model: GenericLatentControlModel, loader, device: torch.device, restore_weight: float) -> float:
    logger.info("Computing generic-encoder validation loss.")
    model.eval()
    loss_sum = 0.0
    row_count = 0

    for batch in loader:
        batch = batch_to_device(batch, device)
        output = model(batch)
        loss, _, _ = _generic_loss(output, batch, restore_weight)
        batch_rows = int(batch["watch_event"].shape[0])
        loss_sum += float(loss.detach().cpu()) * batch_rows
        row_count += batch_rows

    value = loss_sum / max(row_count, 1)
    logger.info("Generic-encoder validation loss computed. loss=%.6f rows=%d", value, row_count)
    return float(value)


def _train_generic_encoder(config: dict[str, Any], args: argparse.Namespace, device: torch.device) -> tuple[GenericLatentControlModel, dict[str, Any], dict[str, Any]]:
    logger.info("Training the generic encoder.")

    train_config = copy.deepcopy(config)
    train_config.setdefault("debug", {})
    if args.baseline_train_rows is not None:
        train_config["debug"]["max_train_rows"] = int(args.baseline_train_rows)
    if args.baseline_eval_rows is not None:
        train_config["debug"]["max_eval_rows"] = int(args.baseline_eval_rows)
    if args.baseline_epochs is not None:
        train_config["training"]["epochs"] = int(args.baseline_epochs)
    if args.baseline_lr is not None:
        train_config["training"]["learning_rate"] = float(args.baseline_lr)
        train_config["training"]["sparse_learning_rate"] = float(args.baseline_lr)
    train_config["training"]["weight_decay"] = float(args.weight_decay)
    processed_dir = Path(train_config["data"]["processed_dir"]).expanduser().resolve()
    manifest = load_manifest(processed_dir)
    loaders = _make_loaders(train_config, processed_dir)

    model = _build_generic_encoder(train_config, manifest).to(device)
    total_params = int(sum(parameter.numel() for parameter in model.parameters()))
    control_only_params = int(
        sum(
            parameter.numel()
            for module in [model.generic_norm, model.latent_head, model.event_head, model.time_head]
            for parameter in module.parameters()
        )
    )
    matched_provider_params = int(total_params - control_only_params)
    optimizers = _build_optimizers(model, train_config)
    restore_weight = float(train_config.get("loss", {}).get("lambda_obs_set_restore", 1.0))

    checkpoint_dir = Path(__file__).resolve().parents[1] / "artifacts" / "runs" / "generic_latent_control" / processed_dir.name / f"seed_{int(args.seed)}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(train_config["training"]["epochs"]) + 1):
        logger.info("Starting generic-encoder training epoch. epoch=%d", epoch)
        model.train()
        total_sum = 0.0
        event_sum = 0.0
        restore_sum = 0.0
        row_count = 0
        for batch in loaders["standard_train"]:
            batch = batch_to_device(batch, device)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, event_loss, restore_loss = _generic_loss(output, batch, restore_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config["training"]["clip_grad_norm"]))
            for optimizer in optimizers:
                optimizer.step()
            batch_rows = int(batch["watch_event"].shape[0])
            total_sum += float(loss.detach().cpu()) * batch_rows
            event_sum += float(event_loss.cpu()) * batch_rows
            restore_sum += float(restore_loss.cpu()) * batch_rows
            row_count += batch_rows

        val_loss = _generic_validation_loss(model, loaders["standard_val"], device, restore_weight)
        row = {
            "epoch": int(epoch),
            "train_loss": float(total_sum / max(row_count, 1)),
            "train_event_ce": float(event_sum / max(row_count, 1)),
            "train_restore_huber": float(restore_sum / max(row_count, 1)),
            "val_loss": float(val_loss),
        }
        history.append(row)
        logger.info(
            "Generic-encoder training epoch complete. epoch=%d train=%.6f val=%.6f",
            epoch,
            row["train_loss"],
            row["val_loss"],
        )
        if val_loss < best_val:
            best_val = float(val_loss)
            best_epoch = int(epoch)
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": train_config,
                    "manifest": manifest,
                    "epoch": int(epoch),
                    "val_loss": float(val_loss),
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(train_config["training"].get("patience", 2)):
                logger.info("Generic-encoder early stopping triggered. epoch=%d", epoch)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    audit = {
        "generic_parameter_count": total_params,
        "matched_provider_parameter_count": matched_provider_params,
        "relative_parameter_overhead": float((total_params - matched_provider_params) / max(matched_provider_params, 1)),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    logger.info(
        "Generic-encoder training complete. params=%d matched_provider=%d overhead=%.6f",
        total_params,
        matched_provider_params,
        audit["relative_parameter_overhead"],
    )
    return model, loaders, audit


@torch.no_grad()


def _collect_generic_latent(model: GenericLatentControlModel, dataset, index: np.ndarray | None, batch_size: int, device: torch.device) -> np.ndarray:
    row_count = len(dataset) if index is None else int(index.shape[0])
    logger.info("Extracting generic latent features. rows=%d", row_count)
    model.eval()
    parts: list[np.ndarray] = []

    for batch_index in _iter_index(index, len(dataset), int(batch_size)):
        batch = dataset.__getitems__(batch_index)
        batch = batch_to_device(batch, device)
        latent = model(batch)["latent"]
        parts.append(latent.detach().cpu().numpy().astype(np.float32, copy=False))

    output = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    logger.info("Generic latent features extracted. shape=%s", tuple(output.shape))
    return output


def _generic_label_payload(dataset, index: np.ndarray | None = None) -> dict[str, np.ndarray]:

    lookup = slice(None) if index is None else index
    arrays = dataset.arrays
    event = np.asarray(arrays["watch_event"][lookup], dtype=np.int64)
    return {
        "play_time": np.asarray(arrays["play_time_sec"][lookup], dtype=np.float32).clip(min=0.0),
        "duration_sec": np.asarray(arrays["duration_sec"][lookup], dtype=np.float32).clip(min=0.0),
        "watch_event": event,
        "session_key": np.asarray(arrays["session_key"][lookup], dtype=np.int64),
        "long_label": np.isin(event, np.asarray([2, 3], dtype=np.int64)).astype(np.float32),
        "overplay_label": (event == 3).astype(np.float32),
    }


def _run_generic_control(config: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    logger.info("Running the controlled generic-latent comparison.")

    model, loaders, audit = _train_generic_encoder(config, args, device)
    train_dataset = loaders["standard_train"].dataset
    val_dataset = loaders["standard_val"].dataset
    test_dataset = loaders["standard_test"].dataset
    train_index = _sample_train_indices(len(train_dataset), int(args.train_rows), int(args.seed), logger)

    latent = {
        "train": _collect_generic_latent(model, train_dataset, train_index, int(args.feature_batch_size), device),
        "val": _collect_generic_latent(model, val_dataset, None, int(args.feature_batch_size), device),
        "test": _collect_generic_latent(model, test_dataset, None, int(args.feature_batch_size), device),
    }
    raw_train, _ = _raw_features(train_dataset.arrays, train_index)
    raw_val, _ = _raw_features(val_dataset.arrays, None)
    raw_test, _ = _raw_features(test_dataset.arrays, None)
    matrices = {
        "train": np.concatenate([raw_train, latent["train"]], axis=1).astype(np.float32, copy=False),
        "val": np.concatenate([raw_val, latent["val"]], axis=1).astype(np.float32, copy=False),
        "test": np.concatenate([raw_test, latent["test"]], axis=1).astype(np.float32, copy=False),
    }

    train_labels = _generic_label_payload(train_dataset, train_index)
    val_labels = _generic_label_payload(val_dataset)
    test_labels = _generic_label_payload(test_dataset)
    labels = {
        "long": {"train": train_labels["long_label"], "val": val_labels["long_label"], "test": test_labels["long_label"]},
        "overplay": {"train": train_labels["overplay_label"], "val": val_labels["overplay_label"], "test": test_labels["overplay_label"]},
    }
    play_time = {
        "train": train_labels["play_time"],
        "val": val_labels["play_time"],
        "test": test_labels["play_time"],
    }
    set_seed(int(args.seed) + _seed_offset(str(args.scorer), "fused"))
    readout = _run_feature_set(
        str(args.scorer),
        "generic_latent",
        matrices,
        labels,
        play_time,
        None,
        test_labels,
        device,
        args,
        logger,
    )

    logger.info("Controlled generic-latent comparison complete.")
    return {
        "baseline": "generic",
        "encoder": audit,
        "representation": {
            "columns": ["raw_context", "generic_latent"],
            "train_rows": int(matrices["train"].shape[0]),
            "raw_feature_dim": int(raw_train.shape[1]),
            "latent_dim": int(latent["train"].shape[1]),
            "feature_dim": int(matrices["train"].shape[1]),
        },
        "readout": readout,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DSI decision readout on frozen baseline representations.")
    parser.add_argument("--config", default="configs/kuairec.yaml")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--split-prefix", choices=["standard", "random"], default="standard")
    _add_baseline_args(parser)
    _add_readout_args(parser)
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    output_path = report_path(config_path, "baseline_readout_probe", int(args.seed), args.output)
    readout_log_path = log_path(config_path, "baseline_readout_probe", int(args.seed), args.log_path)
    setup_logger(readout_log_path)
    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    results = {}
    for baseline in [str(value) for value in args.baselines]:
        set_seed(int(args.seed))
        if baseline == "generic":
            results[baseline] = _run_generic_control(config, args, device)
        else:
            results[baseline] = _run_one_baseline(config, args, baseline, device)

    report = {
        "method": "baseline_representation_ordinal_decision_readout",
        "seed": int(args.seed),
        "config": str(config_path),
        "processed_dir": str(config["data"]["processed_dir"]),
        "split_prefix": str(args.split_prefix),
        "feature_protocol": str(args.feature_protocol),
        "baselines": [str(value) for value in args.baselines],
        "readout_config": {
            "scorer": str(args.scorer),
            "rank_target": str(args.rank_target),
            "validation_select": str(args.validation_select),
            "train_rows": int(args.train_rows),
            "epochs": int(args.scorer_epochs),
            "lr": float(args.scorer_lr),
            "rank_loss_weight": float(args.rank_loss_weight),
            "ordinal_bins": int(args.ordinal_bins),
            "ordinal_beta": float(args.ordinal_beta),
            "ordinal_restore_weight": float(args.ordinal_restore_weight),
            "ordinal_ce_weight": float(args.ordinal_ce_weight),
            "ordinal_prior_weight": float(args.ordinal_prior_weight),
            "fast_eval": bool(args.fast_eval),
            "fast_eval_rows": int(args.fast_eval_rows),
            "input_dim": None if args.readout_input_dim is None else int(args.readout_input_dim),
            "representation_only": bool(args.representation_only),
            "representation_protocol": str(args.representation_protocol),
            "interface_cache_dir": str(args.interface_cache_dir),
        },
        "results": results,
    }
    serialized_report = portableize_serialized_paths(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(serialized_report, handle, indent=2, ensure_ascii=False)
    logger.info("Controlled baseline readout report written. output=%s", output_path)
    print(json.dumps(serialized_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
