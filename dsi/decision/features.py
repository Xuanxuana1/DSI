from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dsi.commands import load_config as load_dsi_config
from dsi.dataset import NPZExposureDataset, load_manifest
from dsi.event_mass import posterior_event_mass_from_output
from dsi.event_mass import ratio_observation_set_posterior_event_mass_from_output
from dsi.metrics import expected_time_seconds_torch
from dsi.training import _build_model, _load_model_state_dict
from dsi.utils.bridge import portableize_serialized_paths, resolve_path
from dsi.utils.logging_utils import setup_logging

FEATURE_CACHE_SCHEMA_VERSION = 13
TENSOR_KEYS = {
    "user_idx",
    "item_idx",
    "tab",
    "prev_event",
    "numeric",
    "session_numeric",
    "content_category",
    "content_numeric",
    "raw_sparse",
    "raw_numeric",
    "raw_seq",
    "raw_seq_mask",
    "duration_sec",
    "event",
    "watch_event",
    "time_bin",
    "engagement_mark",
    "play_time_sec",
    "session_key",
}


@dataclass(frozen=True)


class FeatureSource:
    name: str
    model_type: str
    config_path: Path
    checkpoint_root: Path


def _parse_feature_source(value: str) -> FeatureSource:
    logger = setup_logging(None)
    logger.info("开始解析posterior source。value=%s", value)

    parts = str(value).split(":", 3)
    if len(parts) != 4:
        raise ValueError("--feature-source must use name:model_type:config_path:checkpoint_root")

    source = FeatureSource(
        name=parts[0],
        model_type=parts[1],
        config_path=resolve_path(parts[2]).resolve(),
        checkpoint_root=resolve_path(parts[3]).resolve(),
    )
    logger.info("posterior source解析完成。name=%s model=%s", source.name, source.model_type)
    return source


def _load_content_arrays(processed_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    logger = setup_logging(None)
    logger.info("开始加载共享item content特征。processed_dir=%s", processed_dir)

    mmap_dir = processed_dir / "item_content"
    if mmap_dir.is_dir():
        arrays = (
            np.load(mmap_dir / "content_category.npy", mmap_mode="r", allow_pickle=False),
            np.load(mmap_dir / "content_numeric.npy", mmap_mode="r", allow_pickle=False),
        )
        logger.info("共享item content特征加载完成。source=mmap")
        return arrays

    npz_path = processed_dir / "item_content.npz"
    if npz_path.exists():
        loaded = np.load(npz_path, allow_pickle=False)
        logger.info("共享item content特征加载完成。source=npz")
        return loaded["content_category"], loaded["content_numeric"]

    logger.info("共享item content特征不存在。")
    return None


def _prepare_model(
    source: FeatureSource,
    seed: int,
    device: torch.device,
    logger,
) -> tuple[dict[str, Any], torch.nn.Module, dict[str, Any]]:
    logger.info("开始加载冻结DSI provider。source=%s seed=%s", source.name, int(seed))

    config, _ = load_dsi_config(source.config_path)
    config["training"]["seed"] = int(seed)
    processed_dir = Path(config["data"]["processed_dir"]).expanduser().resolve()
    manifest = load_manifest(processed_dir)

    checkpoint_path = source.checkpoint_root / source.model_type / f"seed_{int(seed)}" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("config", config) if isinstance(checkpoint, dict) else config
    model = _build_model(source.model_type, manifest, model_config).to(device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    _load_model_state_dict(model, state)
    model.eval()

    logger.info("冻结DSI provider加载完成。checkpoint=%s", checkpoint_path)
    return config, model, manifest


def _posterior_event_mass(model_type: str, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], upper: torch.Tensor) -> torch.Tensor:

    if model_type not in {"obs_set_posterior", "time_only_posterior"}:
        raise ValueError(f"Unsupported model_type for DSI decision: {model_type}")

    return posterior_event_mass_from_output(output, batch["duration_sec"], upper)


def _posterior_summary_features(
    event_mass: torch.Tensor,
    upper: torch.Tensor,
    duration_sec: torch.Tensor,
    output: dict[str, torch.Tensor] | None = None,
    compact_only: bool = False,
) -> torch.Tensor:

    eps = event_mass.new_tensor(1e-6)
    typed_upper = upper.to(device=event_mass.device, dtype=event_mass.dtype).view(1, 1, -1)
    event_prob = event_mass.sum(dim=2).clamp(0.0, 1.0)
    time_prob = event_mass.sum(dim=1).clamp(0.0, 1.0)
    expected_time = expected_time_seconds_torch(event_mass, upper.to(device=event_mass.device, dtype=event_mass.dtype)).view(-1, 1)

    duration = duration_sec.to(device=event_mass.device, dtype=event_mass.dtype).view(-1, 1).clamp_min(1e-3)
    long_prob = event_prob[:, 2:4].sum(dim=1, keepdim=True).clamp(1e-6, 1.0 - 1e-6)
    event_logit = torch.logit(event_prob.clamp(1e-6, 1.0 - 1e-6))
    expected_ratio = (expected_time / duration).clamp(0.0, 20.0)

    event_time_sum = (event_mass * typed_upper).sum(dim=2)
    event_mean = event_time_sum / event_prob.clamp_min(eps)
    event_mean = torch.where(event_prob > eps, event_mean, torch.zeros_like(event_mean))
    event_mean_log = torch.log1p(event_mean.clamp_min(0.0))
    event_mean_ratio = (event_mean / duration).clamp(0.0, 20.0)

    event_entropy = -(event_prob.clamp_min(eps) * event_prob.clamp_min(eps).log()).sum(dim=1, keepdim=True)
    time_entropy = -(time_prob.clamp_min(eps) * time_prob.clamp_min(eps).log()).sum(dim=1, keepdim=True)
    event_top2 = torch.topk(event_prob, k=min(2, event_prob.shape[1]), dim=1).values
    time_top2 = torch.topk(time_prob, k=min(2, time_prob.shape[1]), dim=1).values
    event_margin = event_top2[:, :1] - event_top2[:, 1:2] if event_top2.shape[1] > 1 else event_top2[:, :1]
    time_margin = time_top2[:, :1] - time_top2[:, 1:2] if time_top2.shape[1] > 1 else time_top2[:, :1]
    completion_ratio = event_mean_ratio[:, 2:3] if event_mean_ratio.shape[1] > 2 else torch.zeros_like(expected_ratio)
    overplay_ratio = event_mean_ratio[:, 3:4] if event_mean_ratio.shape[1] > 3 else torch.zeros_like(expected_ratio)
    overplay_residual_ratio = (
        ((event_mean[:, 3:4] - duration).clamp_min(0.0) / duration).clamp(0.0, 20.0)
        if event_mean.shape[1] > 3
        else torch.zeros_like(expected_ratio)
    )

    upper_vector = upper.to(device=event_mass.device, dtype=event_mass.dtype).view(1, -1)
    cumulative_time = torch.cumsum(time_prob, dim=1).clamp(0.0, 1.0)
    quantile_levels = event_mass.new_tensor([0.25, 0.5, 0.75, 0.9]).view(1, -1, 1)
    reached = cumulative_time.unsqueeze(1) >= quantile_levels
    reached_any = reached.any(dim=2)
    quantile_idx = reached.to(torch.int64).argmax(dim=2)
    fallback_idx = torch.full_like(quantile_idx, int(upper_vector.shape[1]) - 1)
    quantile_idx = torch.where(reached_any, quantile_idx, fallback_idx)
    quantile_time = upper_vector.view(-1).index_select(0, quantile_idx.reshape(-1)).view(quantile_idx.shape)
    quantile_log = torch.log1p(quantile_time.clamp_min(0.0))
    quantile_ratio = (quantile_time / duration).clamp(0.0, 20.0)
    survival_levels = event_mass.new_tensor([0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.5, 2.0]).view(1, -1, 1)
    survival_threshold = duration.view(-1, 1, 1) * survival_levels
    survival_at_duration = (
        time_prob.unsqueeze(1)
        * (upper_vector.view(1, 1, -1) > survival_threshold).to(dtype=time_prob.dtype)
    ).sum(dim=2)

    features = [
        event_prob,
        event_logit,
        long_prob,
        torch.logit(long_prob),
        torch.log1p(expected_time.clamp_min(0.0)),
        expected_ratio,
        event_mean_log,
        event_mean_ratio,
        event_entropy,
        time_entropy,
        event_margin,
        time_margin,
        completion_ratio,
        overplay_ratio,
        overplay_residual_ratio,
    ]
    if bool(compact_only):
        return torch.cat(features, dim=1)

    features.extend([
        quantile_log,
        quantile_ratio,
        survival_at_duration,
    ])

    if output is not None and "ratio_obs_set_logits" in output:
        ratio_mass = ratio_observation_set_posterior_event_mass_from_output(output)
        ratio_upper = output["ratio_upper_bounds"].to(device=event_mass.device, dtype=event_mass.dtype)
        ratio_event_prob = ratio_mass.sum(dim=2).clamp(1e-6, 1.0 - 1e-6)
        ratio_prob = ratio_mass.sum(dim=1).clamp_min(eps)
        ratio_expected = (ratio_prob * ratio_upper.view(1, -1)).sum(dim=1, keepdim=True)
        ratio_survival = (1.0 - torch.cumsum(ratio_prob, dim=1)).clamp(1e-6, 1.0 - 1e-6)
        ratio_levels = event_mass.new_tensor([0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, 5.0]).view(1, -1, 1)
        ratio_survival_at = (
            ratio_prob.unsqueeze(1)
            * (ratio_upper.view(1, 1, -1) > ratio_levels).to(dtype=ratio_prob.dtype)
        ).sum(dim=2)
        ratio_event_mean = (ratio_mass * ratio_upper.view(1, 1, -1)).sum(dim=2) / ratio_event_prob.clamp_min(eps)
        ratio_event_mean = torch.where(ratio_event_prob > eps, ratio_event_mean, torch.zeros_like(ratio_event_mean))
        features.extend(
            [
                ratio_event_prob,
                torch.logit(ratio_event_prob),
                ratio_expected.clamp(0.0, 20.0),
                torch.log1p((ratio_expected * duration).clamp_min(0.0)),
                ratio_event_mean.clamp(0.0, 20.0),
                ratio_survival_at,
                ratio_survival.mean(dim=1, keepdim=True),
            ]
        )

    if output is not None and "duration_survival_logits" in output:
        duration_survival = torch.sigmoid(output["duration_survival_logits"]).clamp(1e-6, 1.0 - 1e-6)
        features.extend(
            [
                duration_survival,
                torch.logit(duration_survival),
                duration_survival.mean(dim=1, keepdim=True),
            ]
        )

    return torch.cat(features, dim=1)


def _plain_time_summary_features(
    event_mass: torch.Tensor,
    upper: torch.Tensor,
    duration_sec: torch.Tensor,
) -> torch.Tensor:

    eps = event_mass.new_tensor(1e-6)
    time_prob = event_mass.sum(dim=1).clamp_min(0.0)
    time_prob = time_prob / time_prob.sum(dim=1, keepdim=True).clamp_min(eps)
    upper_vector = upper.to(device=event_mass.device, dtype=event_mass.dtype).view(1, -1)
    duration = duration_sec.to(device=event_mass.device, dtype=event_mass.dtype).view(-1, 1).clamp_min(1e-3)

    cumulative = torch.cumsum(time_prob, dim=1).clamp(0.0, 1.0)
    quantile_levels = event_mass.new_tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).view(1, -1, 1)
    reached = cumulative.unsqueeze(1) >= quantile_levels
    reached_any = reached.any(dim=2)
    quantile_idx = reached.to(torch.int64).argmax(dim=2)
    fallback_idx = torch.full_like(quantile_idx, int(upper_vector.shape[1]) - 1)
    quantile_idx = torch.where(reached_any, quantile_idx, fallback_idx)
    quantile_time = upper_vector.view(-1).index_select(0, quantile_idx.reshape(-1)).view(quantile_idx.shape)
    quantile_log = torch.log1p(quantile_time.clamp_min(0.0))
    quantile_ratio = (quantile_time / duration).clamp(0.0, 20.0)

    mean_time = (time_prob * upper_vector).sum(dim=1, keepdim=True)
    centered = upper_vector - mean_time
    variance_time = (time_prob * centered.square()).sum(dim=1, keepdim=True).clamp_min(0.0)
    std_time = torch.sqrt(variance_time + eps)
    entropy = -(time_prob.clamp_min(eps) * time_prob.clamp_min(eps).log()).sum(dim=1, keepdim=True)
    top2 = torch.topk(time_prob, k=min(2, int(time_prob.shape[1])), dim=1).values
    top_mass = top2[:, :1]
    top_margin = top2[:, :1] - top2[:, 1:2] if top2.shape[1] > 1 else top2[:, :1]
    interquantile_ratio = ((quantile_time[:, 8:9] - quantile_time[:, :1]) / duration).clamp(0.0, 20.0)
    coefficient_variation = (std_time / mean_time.clamp_min(1e-3)).clamp(0.0, 20.0)
    global_statistics = torch.cat(
        [
            torch.log1p(mean_time.clamp_min(0.0)),
            (mean_time / duration).clamp(0.0, 20.0),
            torch.log1p(std_time),
            (std_time / duration).clamp(0.0, 20.0),
            entropy,
            top_mass,
            top_margin,
            interquantile_ratio,
            coefficient_variation,
        ],
        dim=1,
    )

    features = torch.cat([quantile_log, quantile_ratio, global_statistics], dim=1)
    if int(features.shape[1]) != 27:
        raise RuntimeError(f"Plain time summary must have 27 features, got {features.shape[1]}")
    return features


def _scalar_time_summary_features(
    event_mass: torch.Tensor,
    upper: torch.Tensor,
    duration_sec: torch.Tensor,
) -> torch.Tensor:
    eps = event_mass.new_tensor(1e-6)
    duration = duration_sec.to(device=event_mass.device, dtype=event_mass.dtype).view(-1, 1).clamp_min(1e-3)
    expected_time = expected_time_seconds_torch(event_mass, upper.to(device=event_mass.device, dtype=event_mass.dtype)).view(-1, 1)
    features = torch.cat(
        [
            torch.log1p(expected_time.clamp_min(0.0)),
            (expected_time / duration).clamp(0.0, 20.0),
        ],
        dim=1,
    )
    if not torch.isfinite(features).all():
        raise RuntimeError(f"Scalar time summary contains non-finite values: eps={eps.item()}")
    return features


def _full_time_distribution_features(event_mass: torch.Tensor) -> torch.Tensor:
    time_prob = event_mass.sum(dim=1).clamp_min(0.0)
    time_prob = time_prob / time_prob.sum(dim=1, keepdim=True).clamp_min(event_mass.new_tensor(1e-6))
    return time_prob


def _posterior_hidden_features(output: dict[str, torch.Tensor]) -> torch.Tensor | None:

    tensors = [
        value.float()
        for key in ("posterior_context", "posterior_hidden")
        if (value := output.get(key)) is not None
    ]
    if not tensors:
        return None

    return torch.cat(
        [torch.nn.functional.layer_norm(value, value.shape[-1:]) for value in tensors],
        dim=1,
    )


@torch.no_grad()


def _model_features(
    source: FeatureSource,
    config: dict[str, Any],
    model: torch.nn.Module,
    manifest: dict[str, Any],
    split: str,
    indices: np.ndarray | None,
    batch_size: int,
    device: torch.device,
    content_arrays: tuple[np.ndarray, np.ndarray] | None,
    summary_protocol: str,
    logger,
) -> np.ndarray:
    logger.info("开始生成posterior summary feature。split=%s source=%s", split, source.name)

    processed_dir = Path(config["data"]["processed_dir"]).expanduser().resolve()
    dataset = NPZExposureDataset(processed_dir / f"{split}.npz", content_arrays=content_arrays)
    row_indices = np.arange(len(dataset), dtype=np.int64) if indices is None else indices.astype(np.int64, copy=False)
    upper = torch.as_tensor(manifest["time_bin_upper_bounds_sec"], dtype=torch.float32, device=device)

    parts: list[np.ndarray] = []
    for batch_idx, start in enumerate(range(0, row_indices.shape[0], int(batch_size))):
        batch_indices = row_indices[start : start + int(batch_size)]
        batch = dataset.__getitems__(batch_indices)
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if key in TENSOR_KEYS
        }
        output = model(batch)
        event_mass = _posterior_event_mass(source.model_type, output, batch, upper.to(dtype=batch["duration_sec"].dtype))
        if summary_protocol == "plain_time_27":
            features = _plain_time_summary_features(event_mass, upper, batch["duration_sec"])
        elif summary_protocol == "scalar_2":
            features = _scalar_time_summary_features(event_mass, upper, batch["duration_sec"])
        elif summary_protocol == "full_time_69":
            features = _full_time_distribution_features(event_mass)
        else:
            features = _posterior_summary_features(
                event_mass,
                upper,
                batch["duration_sec"],
                output,
                summary_protocol == "compact_27",
            )
        hidden_features = None if summary_protocol in {"compact_27", "plain_time_27", "scalar_2", "full_time_69"} else _posterior_hidden_features(output)
        if summary_protocol == "hidden_only":
            if hidden_features is None:
                raise ValueError("hidden_only posterior summary requires provider hidden/context tensors")
            features = hidden_features
        elif hidden_features is not None:
            features = torch.cat([features, hidden_features], dim=1)
        parts.append(features.detach().cpu().numpy().astype(np.float32, copy=False))
        if (batch_idx + 1) % 100 == 0:
            logger.info("posterior summary生成进度。split=%s batches=%s", split, batch_idx + 1)

    output_features = np.concatenate(parts, axis=0)
    logger.info("posterior summary feature生成完成。split=%s rows=%s dim=%s", split, output_features.shape[0], output_features.shape[1])
    return output_features


def _raw_features(arrays: dict[str, np.ndarray], indices: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:

    lookup = slice(None) if indices is None else indices
    numeric = np.asarray(arrays["numeric"][lookup], dtype=np.float32)
    session_numeric = np.asarray(arrays["session_numeric"][lookup], dtype=np.float32)
    duration = np.asarray(arrays["duration_sec"][lookup], dtype=np.float32).reshape(-1, 1)
    tab = np.asarray(arrays["tab"][lookup], dtype=np.float32).reshape(-1, 1)
    prev_event = np.asarray(arrays["prev_event"][lookup], dtype=np.float32).reshape(-1, 1)

    features = np.concatenate(
        [
            numeric,
            session_numeric,
            np.log1p(np.clip(duration, 0.0, None)),
            tab,
            prev_event,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    session = np.asarray(arrays["session_key"][lookup], dtype=np.int64)
    return features, session


def _id_features(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray | None,
    content_arrays: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    logger = setup_logging(None)
    logger.info("开始抽取decision稀疏ID特征。")

    lookup = slice(None) if indices is None else indices
    item_idx = np.asarray(arrays["item_idx"][lookup], dtype=np.int64)

    parts = [
        np.asarray(arrays["user_idx"][lookup], dtype=np.int64),
        item_idx,
        np.asarray(arrays["tab"][lookup], dtype=np.int64),
        np.asarray(arrays["prev_event"][lookup], dtype=np.int64),
    ]

    if content_arrays is not None and content_arrays[0].shape[0] > 0 and content_arrays[0].shape[1] > 0:
        content_category = np.asarray(content_arrays[0][item_idx], dtype=np.int64)
        for idx in range(content_category.shape[1]):
            parts.append(content_category[:, idx])

    output = np.column_stack(parts).astype(np.int64, copy=False)
    logger.info("decision稀疏ID特征抽取完成。dim=%s", int(output.shape[1]))
    return output


def _decision_sparse_metadata(manifest: dict[str, Any]) -> tuple[list[str], list[int]]:
    logger = setup_logging(None)
    logger.info("开始构造decision稀疏特征元信息。")

    names = ["user_idx", "item_idx", "tab", "prev_event"]
    cardinalities = [
        int(manifest["num_users"]),
        int(manifest["num_items"]),
        int(manifest["num_tabs"]),
        int(manifest.get("unknown_prev_event", 5)) + 1,
    ]

    content_meta = manifest.get("content_features", {})
    category_names = [str(value) for value in content_meta.get("content_category_names", [])]
    category_cardinalities = [int(value) for value in content_meta.get("content_category_cardinalities", [])]
    names.extend([f"content_{value}" for value in category_names[: len(category_cardinalities)]])
    cardinalities.extend(category_cardinalities)

    logger.info("decision稀疏特征元信息构造完成。fields=%s", int(len(names)))
    return names, cardinalities


def _rank_label(npz: np.lib.npyio.NpzFile, target: str, indices: np.ndarray | None) -> np.ndarray:

    lookup = slice(None) if indices is None else indices
    event = np.asarray(npz["watch_event"][lookup], dtype=np.int64)

    if str(target) == "long":
        return np.isin(event, np.asarray([2, 3], dtype=np.int64)).astype(np.float32)
    if str(target) == "overplay":
        return (event == 3).astype(np.float32)
    raise ValueError(f"Unsupported rank_target={target}")


def _play_time(npz: np.lib.npyio.NpzFile, indices: np.ndarray | None) -> np.ndarray:

    lookup = slice(None) if indices is None else indices
    return np.asarray(npz["play_time_sec"][lookup], dtype=np.float32).clip(min=0.0)


def _sample_train_indices(train_len: int, train_rows: int, seed: int, logger) -> np.ndarray | None:
    logger.info("开始采样decision训练行。train_len=%s train_rows=%s", int(train_len), int(train_rows))

    rows = min(int(train_rows), int(train_len))
    if rows >= int(train_len):
        logger.info("decision训练使用全量行。rows=%s", int(train_len))
        return None

    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(int(train_len), size=rows, replace=False).astype(np.int64))

    logger.info("decision训练行采样完成。rows=%s", int(indices.shape[0]))
    return indices


def _train_index_count(train_idx: np.ndarray | None, train_len: int) -> int:

    return int(train_len) if train_idx is None else int(train_idx.shape[0])


def _feature_cache_dir(args: argparse.Namespace) -> Path:

    if getattr(args, "cache_dir", None):
        return resolve_path(str(args.cache_dir)).resolve()
    return Path(__file__).resolve().parents[2] / "artifacts" / "cache" / "decision_features"


def _path_fingerprint(path: Path) -> dict[str, Any]:

    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}

    stat = resolved.stat()
    return {"path": str(resolved), "exists": True, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _feature_source_fingerprint(raw_source: str, seed: int) -> dict[str, Any]:

    output: dict[str, Any] = {"source": str(raw_source)}
    parts = str(raw_source).split(":", 3)
    if len(parts) == 4:
        model_name = parts[1]
        config_path = resolve_path(parts[2])
        checkpoint_root = resolve_path(parts[3])
        checkpoint_path = checkpoint_root / str(model_name) / f"seed_{int(seed)}" / "best.pt"
        output["config"] = _path_fingerprint(config_path)
        output["checkpoint"] = _path_fingerprint(checkpoint_path)
    return output


def _feature_cache_path(args: argparse.Namespace) -> Path:

    config, config_path = load_dsi_config(args.base_config)
    processed_dir = Path(config["data"]["processed_dir"]).expanduser().resolve()
    split_fingerprints = {
        name: _path_fingerprint(processed_dir / f"{name}.npz")
        for name in ["standard_train", "standard_val", "standard_test"]
    }
    key_payload = {
        "schema": int(FEATURE_CACHE_SCHEMA_VERSION),
        "base_config": _path_fingerprint(config_path),
        "processed_dir": str(processed_dir),
        "splits": split_fingerprints,
        "item_content": _path_fingerprint(processed_dir / "item_content.npz"),
        "seed": int(args.seed),
        "train_rows": int(args.train_rows),
        "posterior_summary_protocol": str(getattr(args, "posterior_summary_protocol", "auto")),
        "feature_sources": [_feature_source_fingerprint(str(source), int(args.seed)) for source in getattr(args, "feature_source", [])],
    }
    digest = hashlib.sha1(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    return _feature_cache_dir(args) / f"{Path(config_path).stem}_seed{int(args.seed)}_rows{int(args.train_rows)}_{digest}.npz"


def _rebuild_feature_sets(
    raw: dict[str, np.ndarray],
    posterior: dict[str, np.ndarray] | None,
    logger=None,
) -> dict[str, dict[str, np.ndarray]]:

    feature_sets: dict[str, dict[str, np.ndarray]] = {"raw": raw}
    if posterior is None:
        return feature_sets

    if logger is not None:
        logger.info("开始构造等维特征消融。")

    feature_sets["posterior"] = posterior

    raw_equivalent = {
        split: np.concatenate([raw[split], np.zeros_like(posterior[split], dtype=np.float32)], axis=1).astype(np.float32, copy=False)
        for split in ["train", "val", "test"]
    }

    posterior_equivalent = {
        split: np.concatenate([np.zeros_like(raw[split], dtype=np.float32), posterior[split]], axis=1).astype(np.float32, copy=False)
        for split in ["train", "val", "test"]
    }

    feature_sets["fused"] = {
        split: np.concatenate([raw[split], posterior[split]], axis=1).astype(np.float32, copy=False)
        for split in ["train", "val", "test"]
    }
    feature_sets["raw_padded"] = raw_equivalent
    feature_sets["raw_equivalent"] = raw_equivalent
    feature_sets["posterior_equivalent"] = posterior_equivalent

    if logger is not None:
        logger.info("等维特征消融构造完成。dim=%s", int(feature_sets["fused"]["train"].shape[1]))
    return feature_sets


def _load_feature_cache(cache_path: Path, logger) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, Any]] | None:

    if not cache_path.exists():
        logger.info("decision feature cache未命中。path=%s", cache_path)
        return None
    logger.info("开始读取decision feature cache。path=%s", cache_path)

    with np.load(cache_path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"].item()))
        raw = {split: np.asarray(cache[f"feature_raw_{split}"], dtype=np.float32) for split in ["train", "val", "test"]}
        posterior = None
        if "feature_posterior_train" in cache.files:
            posterior = {split: np.asarray(cache[f"feature_posterior_{split}"], dtype=np.float32) for split in ["train", "val", "test"]}
        labels = {
            target: {split: np.asarray(cache[f"label_{target}_{split}"], dtype=np.float32) for split in ["train", "val", "test"]}
            for target in ["long", "overplay"]
        }
        eval_arrays = {
            key: np.asarray(cache[f"eval_{key}"])
            for key in ["play_time", "watch_event", "duration_sec", "session_key", "long_label", "overplay_label"]
        }
        id_features = None
        if "id_train" in cache.files:
            id_features = {split: np.asarray(cache[f"id_{split}"], dtype=np.int64) for split in ["train", "val", "test"]}

    feature_sets = _rebuild_feature_sets(raw, posterior, logger)
    metadata["feature_cache"] = {"enabled": True, "hit": True, "path": str(cache_path)}
    payload = {"labels": labels, "eval": eval_arrays, "id_features": id_features}
    logger.info("decision feature cache读取完成。path=%s", cache_path)
    return metadata, feature_sets, payload


def _save_feature_cache(
    cache_path: Path,
    metadata: dict[str, Any],
    feature_sets: dict[str, dict[str, np.ndarray]],
    payload: dict[str, Any],
    logger,
) -> None:
    logger.info("开始写入decision feature cache。path=%s", cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_metadata = portableize_serialized_paths(metadata)
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(
            json.dumps(serialized_metadata, sort_keys=True),
            dtype=np.str_,
        )
    }
    for split in ["train", "val", "test"]:
        arrays[f"feature_raw_{split}"] = feature_sets["raw"][split].astype(np.float32, copy=False)
        if "posterior" in feature_sets:
            arrays[f"feature_posterior_{split}"] = feature_sets["posterior"][split].astype(np.float32, copy=False)
    for target in ["long", "overplay"]:
        for split in ["train", "val", "test"]:
            arrays[f"label_{target}_{split}"] = payload["labels"][target][split].astype(np.float32, copy=False)
    for key, value in payload["eval"].items():
        arrays[f"eval_{key}"] = np.asarray(value)
    if payload.get("id_features") is not None:
        for split, value in payload["id_features"].items():
            arrays[f"id_{split}"] = np.asarray(value, dtype=np.int64)

    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    with tmp_path.open("wb") as handle:
        np.savez(handle, **arrays)
    tmp_path.replace(cache_path)

    logger.info("decision feature cache写入完成。path=%s", cache_path)


def _build_feature_matrices(args: argparse.Namespace, device: torch.device, logger) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    logger.info("开始构造decision feature矩阵。seed=%s", int(args.seed))

    base_config, _ = load_dsi_config(args.base_config)
    base_config["training"]["seed"] = int(args.seed)
    processed_dir = Path(base_config["data"]["processed_dir"]).expanduser().resolve()
    manifest = load_manifest(processed_dir)
    content_arrays = _load_content_arrays(processed_dir)
    train_npz = np.load(processed_dir / "standard_train.npz", allow_pickle=False)
    val_npz = np.load(processed_dir / "standard_val.npz", allow_pickle=False)
    test_npz = np.load(processed_dir / "standard_test.npz", allow_pickle=False)

    train_idx = _sample_train_indices(int(train_npz["watch_event"].shape[0]), int(args.train_rows), int(args.seed), logger)
    train_raw, _ = _raw_features(train_npz, train_idx)
    val_raw, _ = _raw_features(val_npz, None)
    test_raw, test_session = _raw_features(test_npz, None)
    id_features = {
        "train": _id_features(train_npz, train_idx, content_arrays),
        "val": _id_features(val_npz, None, content_arrays),
        "test": _id_features(test_npz, None, content_arrays),
    }
    labels = {
        "long": {
            "train": _rank_label(train_npz, "long", train_idx),
            "val": _rank_label(val_npz, "long", None),
            "test": _rank_label(test_npz, "long", None),
        },
        "overplay": {
            "train": _rank_label(train_npz, "overplay", train_idx),
            "val": _rank_label(val_npz, "overplay", None),
            "test": _rank_label(test_npz, "overplay", None),
        },
    }

    protocol = str(getattr(args, "posterior_summary_protocol", "auto"))
    if protocol == "auto":
        protocol = "wechat_engineering" if str(base_config.get("data", {}).get("name", "")).lower().startswith("wechat21") else "compact_27"
    elif protocol == "engineering":
        protocol = "wechat_engineering"

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    sources: list[str] = []
    for raw_source in args.feature_source:
        source = _parse_feature_source(raw_source)
        source_config, source_model, source_manifest = _prepare_model(source, int(args.seed), device, logger)
        train_parts.append(_model_features(source, source_config, source_model, source_manifest, "standard_train", train_idx, int(args.feature_batch_size), device, content_arrays, protocol, logger))
        val_parts.append(_model_features(source, source_config, source_model, source_manifest, "standard_val", None, int(args.feature_batch_size), device, content_arrays, protocol, logger))
        test_parts.append(_model_features(source, source_config, source_model, source_manifest, "standard_test", None, int(args.feature_batch_size), device, content_arrays, protocol, logger))
        sources.append(source.name)
        del source_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    feature_sets: dict[str, dict[str, np.ndarray]] = {
        "raw": {"train": train_raw, "val": val_raw, "test": test_raw},
    }
    if train_parts:
        posterior = {
            "train": np.concatenate(train_parts, axis=1).astype(np.float32, copy=False),
            "val": np.concatenate(val_parts, axis=1).astype(np.float32, copy=False),
            "test": np.concatenate(test_parts, axis=1).astype(np.float32, copy=False),
        }
        feature_sets.update(_rebuild_feature_sets(feature_sets["raw"], posterior, logger))

    eval_arrays = {
        "play_time": np.asarray(test_npz["play_time_sec"], dtype=np.float64),
        "watch_event": np.asarray(test_npz["watch_event"], dtype=np.int64),
        "duration_sec": np.asarray(test_npz["duration_sec"], dtype=np.float64),
        "session_key": np.asarray(test_session, dtype=np.int64),
        "long_label": labels["long"]["test"].astype(np.float64, copy=False),
        "overplay_label": labels["overplay"]["test"].astype(np.float64, copy=False),
    }
    sparse_names, sparse_cardinalities = _decision_sparse_metadata(manifest)
    metadata = {
        "processed_dir": str(processed_dir),
        "event_names": list(manifest.get("watch_event_names") or manifest.get("event_names") or []),
        "train_rows": _train_index_count(train_idx, int(train_npz["watch_event"].shape[0])),
        "test_rows": int(eval_arrays["play_time"].shape[0]),
        "sources": sources,
        "posterior_summary_protocol": protocol,
        "raw_feature_dim": int(train_raw.shape[1]),
        "content_numeric_feature_dim": 0,
        "posterior_feature_dim": int(feature_sets.get("posterior", feature_sets["raw"])["train"].shape[1]),
        "fused_feature_dim": int(feature_sets.get("fused", feature_sets["raw"])["train"].shape[1]),
        "decision_sparse_feature_names": sparse_names,
        "decision_sparse_cardinalities": sparse_cardinalities,
    }

    logger.info("decision feature矩阵构造完成。feature_sets=%s sources=%s", sorted(feature_sets), sources)
    return metadata, feature_sets, {"labels": labels, "eval": eval_arrays, "id_features": id_features}


def _cached_feature_matrices(args: argparse.Namespace, device: torch.device, logger) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, Any]]:

    if bool(getattr(args, "no_feature_cache", False)):
        logger.info("decision feature cache已关闭。")
        metadata, feature_sets, payload = _build_feature_matrices(args, device, logger)
        metadata["feature_cache"] = {"enabled": False, "hit": False, "path": None}
        return metadata, feature_sets, payload

    cache_path = _feature_cache_path(args)
    if not bool(getattr(args, "refresh_feature_cache", False)):
        cached = _load_feature_cache(cache_path, logger)
        if cached is not None:
            return cached

    if bool(getattr(args, "refresh_feature_cache", False)):
        logger.info("强制刷新decision feature cache。path=%s", cache_path)
    metadata, feature_sets, payload = _build_feature_matrices(args, device, logger)
    metadata["feature_cache"] = {"enabled": True, "hit": False, "path": str(cache_path)}
    _save_feature_cache(cache_path, metadata, feature_sets, payload, logger)
    return metadata, feature_sets, payload
