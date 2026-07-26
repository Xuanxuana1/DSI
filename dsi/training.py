from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from dsi.calibration import fit_event_marginal_calibration_from_arrays
from dsi.utils.config import dump_config
from dsi.dataset import BlockShuffleBatchSampler, NPZExposureDataset, batch_to_device, identity_collate, load_manifest
from dsi.event_mass import (
    observation_set_posterior_event_mass_from_output,
    posterior_event_mass_from_output,
)
from dsi.losses import (
    observation_set_posterior_losses,
    time_only_posterior_losses,
)
from dsi.metrics import evaluate_model, evaluate_model_light
from dsi.models import (
    ObservationSetPosteriorModel,
    TimeOnlyPosteriorModel,
    build_shape,
)
from dsi.utils.bridge import portableize_serialized_paths

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ddp_context() -> dict[str, Any]:

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    return {
        "enabled": world_size > 1,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_main": rank == 0,
    }


def _init_ddp_if_needed(ctx: dict[str, Any]) -> None:

    if ctx["enabled"] and torch.cuda.is_available():
        torch.cuda.set_device(int(ctx["local_rank"]))

    if ctx["enabled"] and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")


def _cleanup_ddp_if_needed(ctx: dict[str, Any]) -> None:

    if ctx["enabled"] and dist.is_initialized():
        dist.destroy_process_group()


def _ddp_mean(value: float, device: torch.device, ctx: dict[str, Any]) -> float:

    if not ctx["enabled"]:
        return float(value)

    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= int(ctx["world_size"])
    return float(tensor.item())


def _ddp_stop_signal(stop: bool, device: torch.device, ctx: dict[str, Any]) -> bool:

    if not ctx["enabled"]:
        return bool(stop)

    tensor = torch.tensor(1 if stop else 0, dtype=torch.int64, device=device)
    dist.broadcast(tensor, src=0)
    return bool(tensor.item())


def _model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:

    if isinstance(model, DistributedDataParallel):
        return model.module.state_dict()
    return model.state_dict()


def _load_model_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    logger.info("开始加载模型权重。")

    target = model.module if isinstance(model, DistributedDataParallel) else model

    result = target.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        logger.info(
            "模型权重非严格加载完成。missing=%s unexpected=%s",
            list(result.missing_keys),
            list(result.unexpected_keys),
        )
    else:
        logger.info("模型权重加载完成。")


def _make_loaders(config: dict[str, Any], processed_dir: Path, ddp_ctx: dict[str, Any] | None = None) -> dict[str, DataLoader]:
    train_cfg = config["training"]
    debug_cfg = config["debug"]
    ddp_ctx = ddp_ctx or {"enabled": False, "rank": 0, "world_size": 1}
    loaders: dict[str, DataLoader] = {}

    content_arrays = None
    content_path = processed_dir / "item_content.npz"
    content_mmap_dir = processed_dir / "item_content"
    if content_mmap_dir.is_dir():
        content_arrays = (
            np.load(content_mmap_dir / "content_category.npy", mmap_mode="r", allow_pickle=False),
            np.load(content_mmap_dir / "content_numeric.npy", mmap_mode="r", allow_pickle=False),
        )
    elif content_path.exists():
        content = np.load(content_path, allow_pickle=False)
        content_arrays = (content["content_category"], content["content_numeric"])

    for name in ["standard_train", "standard_val", "standard_test"]:
        split_path = processed_dir / f"{name}.npz"
        max_rows = debug_cfg.get("max_train_rows") if name.endswith("train") else debug_cfg.get("max_eval_rows")
        dataset = NPZExposureDataset(split_path, max_rows=max_rows, content_arrays=content_arrays)
        if len(dataset) == 0:

            loaders[name] = None
            continue
        if name == "standard_train" and train_cfg.get("block_shuffle_rows"):

            batch_sampler = BlockShuffleBatchSampler(
                len(dataset),
                int(train_cfg["batch_size"]),
                int(train_cfg["block_shuffle_rows"]),
                int(train_cfg["seed"]),
                int(ddp_ctx["rank"]),
                int(ddp_ctx["world_size"]),
            )

            loaders[name] = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=int(train_cfg["num_workers"]),
                pin_memory=torch.cuda.is_available(),
                collate_fn=identity_collate,
            )
            continue

        batch_size = int(train_cfg.get("eval_batch_size", train_cfg["batch_size"])) if not name.endswith("train") else int(train_cfg["batch_size"])
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name.endswith("train"),
            num_workers=int(train_cfg["num_workers"]),
            pin_memory=torch.cuda.is_available(),
            collate_fn=identity_collate,
        )
    return loaders


def _build_model(model_type: str, manifest: dict[str, Any], config: dict[str, Any]) -> torch.nn.Module:
    shape = build_shape(manifest)
    model_cfg = config["model"]
    sparse_embeddings = bool(model_cfg.get("sparse_embeddings", False))
    if model_type == "obs_set_posterior":
        event_cfg = config.get("event_taxonomy", {})
        duration_ratio_edges = model_cfg.get("duration_ratio_bucket_edges")
        return ObservationSetPosteriorModel(
            shape,
            model_cfg["embedding_dim"],
            model_cfg["hidden_dim"],
            model_cfg["dropout"],
            sparse_embeddings,
            alpha=float(event_cfg.get("alpha", 0.3)),
            mid_cut=float(event_cfg.get("mid_completion_cut", event_cfg.get("mid_cut", 0.9))),
            beta=float(event_cfg.get("beta", 1.1)),
            support_mode=str(event_cfg.get("support_mode", "support_aware")),
            mlp_hidden_layers=int(model_cfg.get("mlp_hidden_layers", 2)),
            use_raw_features=bool(model_cfg.get("use_raw_features", True)),
            ratio_axis_bins=int(model_cfg.get("ratio_axis_bins", 0)),
            duration_survival_bins=int(model_cfg.get("duration_survival_bins", 0)),
            duration_ratio_bucket_edges=duration_ratio_edges,
            duration_prior_bucket_edges=model_cfg.get("duration_prior_bucket_edges"),
            duration_prior_values=model_cfg.get("duration_prior_values"),
            posterior_primary_axis=str(model_cfg.get("posterior_primary_axis", "seconds")),
        )
    if model_type == "time_only_posterior":
        event_cfg = config.get("event_taxonomy", {})
        return TimeOnlyPosteriorModel(
            shape,
            model_cfg["embedding_dim"],
            model_cfg["hidden_dim"],
            model_cfg["dropout"],
            sparse_embeddings,
            alpha=float(event_cfg.get("alpha", 0.3)),
            mid_cut=float(event_cfg.get("mid_completion_cut", event_cfg.get("mid_cut", 0.9))),
            beta=float(event_cfg.get("beta", 1.1)),
            mlp_hidden_layers=int(model_cfg.get("mlp_hidden_layers", 2)),
            use_raw_features=bool(model_cfg.get("use_raw_features", True)),
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def _duration_bucket_edges_from_train(processed_dir: Path, bucket_count: int, logger) -> list[float]:
    logger.info("开始计算provider duration-local ratio边界。buckets=%s", int(bucket_count))

    if int(bucket_count) <= 1:
        logger.info("provider duration-local ratio边界为空。")
        return []

    train_path = processed_dir / "standard_train.npz"
    with np.load(train_path, mmap_mode="r", allow_pickle=False) as arrays:
        values = np.asarray(arrays["duration_sec"], dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        logger.info("provider duration-local ratio边界为空。reason=no_duration")
        return []
    quantiles = np.linspace(0.0, 1.0, int(bucket_count) + 1, dtype=np.float64)[1:-1]
    edges = np.unique(np.quantile(values, quantiles)).astype(np.float32)
    logger.info("provider duration-local ratio边界计算完成。actual_buckets=%s", int(edges.size + 1))
    return [float(value) for value in edges.tolist()]


def _duration_prior_from_train(processed_dir: Path, bucket_count: int, logger) -> tuple[list[float], list[float]]:
    logger.info("开始计算provider duration prior。buckets=%s", int(bucket_count))

    if int(bucket_count) <= 1:
        logger.info("provider duration prior关闭。")
        return [], []

    train_path = processed_dir / "standard_train.npz"
    with np.load(train_path, mmap_mode="r", allow_pickle=False) as arrays:
        duration = np.asarray(arrays["duration_sec"], dtype=np.float64)
        play_time = np.asarray(arrays["play_time_sec"], dtype=np.float64)
    valid = np.isfinite(duration) & np.isfinite(play_time)
    duration = duration[valid]
    play_time = play_time[valid]
    if duration.size == 0:
        logger.info("provider duration prior为空。reason=no_valid_rows")
        return [], []

    quantiles = np.linspace(0.0, 1.0, int(bucket_count) + 1, dtype=np.float64)[1:-1]
    edges = np.unique(np.quantile(duration, quantiles)).astype(np.float32)
    bucket = np.searchsorted(edges, duration.astype(np.float32), side="left")
    bucket_count_actual = int(edges.size + 1)
    global_median = float(np.median(play_time))
    values: list[float] = []
    for bucket_idx in range(bucket_count_actual):
        mask = bucket == bucket_idx
        value = float(np.median(play_time[mask])) if bool(mask.any()) else global_median
        values.append(value)

    logger.info("provider duration prior计算完成。actual_buckets=%s", bucket_count_actual)
    return [float(value) for value in edges.tolist()], values


def _build_optimizers(model: torch.nn.Module, config: dict[str, Any]) -> list[torch.optim.Optimizer]:

    sparse_params: list[torch.nn.Parameter] = []
    dense_params: list[torch.nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding) and bool(module.sparse):
            sparse_params.extend(list(module.parameters(recurse=False)))
        else:
            for param in module.parameters(recurse=False):
                dense_params.append(param)

    optimizers: list[torch.optim.Optimizer] = []
    if dense_params:
        optimizers.append(
            torch.optim.AdamW(
                dense_params,
                lr=float(config["training"]["learning_rate"]),
                weight_decay=float(config["training"]["weight_decay"]),
            )
        )
    if sparse_params:
        optimizers.append(
            torch.optim.SparseAdam(
                sparse_params,
                lr=float(config["training"].get("sparse_learning_rate", config["training"]["learning_rate"])),
            )
        )
    return optimizers


def _grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        value = param.grad.detach().data.norm(2).item()
        total += value * value
    return float(total ** 0.5)


def _git_metadata(repo_dir: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=repo_dir, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run_git(["status", "--short"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(repo_dir: Path) -> dict[str, str]:
    files = [
        "dsi/calibration.py",
        "dsi/dataset.py",
        "dsi/losses.py",
        "dsi/metrics.py",
        "dsi/models.py",
        "dsi/training.py",
        "dsi/utils/time_bins.py",
        "run.py",
    ]
    return {name: _file_sha256(repo_dir / name) for name in files if (repo_dir / name).exists()}


def _session_nll_summary(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    order = np.argsort(arrays["session_key"], kind="mergesort")
    session_key = arrays["session_key"][order].astype(np.int64, copy=False)
    per_sample_nll = arrays["per_sample_nll"][order].astype(np.float64, copy=False)
    unique_session, starts = np.unique(session_key, return_index=True)
    counts = np.diff(np.append(starts, len(session_key))).astype(np.int64)
    nll_sum = np.add.reduceat(per_sample_nll, starts)
    return {
        "session_key": unique_session.astype(np.int64, copy=False),
        "nll_sum": nll_sum.astype(np.float64, copy=False),
        "count": counts,
    }


def _duration_quantile_edges(arrays: dict[str, np.ndarray], buckets: int) -> np.ndarray:
    logger.info("开始计算duration分位边界。buckets=%s", int(buckets))

    duration = np.asarray(arrays.get("duration_sec", []), dtype=np.float32).reshape(-1)
    duration = duration[np.isfinite(duration)]
    if duration.size == 0 or int(buckets) <= 1:
        logger.info("duration分位边界为空。rows=%s buckets=%s", int(duration.size), int(buckets))
        return np.asarray([], dtype=np.float32)

    quantiles = np.linspace(0.0, 1.0, int(buckets) + 1, dtype=np.float64)[1:-1]
    edges = np.quantile(duration, quantiles).astype(np.float32)
    logger.info("duration分位边界计算完成。edges=%s", edges.tolist())
    return edges


def _save_eval_artifacts(run_dir: Path, split_name: str, result, logger) -> None:
    logger.info("开始写出%s评估审计数组...", split_name)

    if {"session_key", "nll_sum", "count"}.issubset(result.arrays):
        session_summary = {
            "session_key": result.arrays["session_key"],
            "nll_sum": result.arrays["nll_sum"],
            "count": result.arrays["count"],
        }
    else:
        session_summary = _session_nll_summary(result.arrays)
    np.savez(run_dir / f"{split_name}_session_nll.npz", **session_summary)

    logger.info("写出%s评估审计数组完成。", split_name)


def _evaluate_split(
    eval_model: torch.nn.Module,
    loader: DataLoader,
    model_type: str,
    device: torch.device,
    manifest: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    primary_metric: str,
    split_name: str,
    calibration: dict[str, Any] | None,
    logger,
):

    logger.info("开始%s评估。rows=%s", split_name, len(loader.dataset))

    result = evaluate_model(
        eval_model,
        loader,
        model_type,
        device,
        manifest["time_bin_upper_bounds_sec"],
        manifest["event_names"],
        config["evaluation"]["max_batches"],
        config["evaluation"]["ece_bins"],
        config["evaluation"]["cindex_buckets"],
        config["evaluation"]["cindex_max_pairs_per_bucket"],
        seed,
        primary_metric,
        calibration,
    )
    logger.info("%s评估完成。primary_nll=%.6f", split_name, result.metrics["primary_nll"])
    return result


def _checkpoint_payload(
    model: torch.nn.Module,
    manifest: dict[str, Any],
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    logger.info("开始构造checkpoint载荷。epoch=%s", epoch)

    payload = {
        "model": _model_state_dict(model),
        "manifest": manifest,
        "config": config,
        "epoch": int(epoch),
        "validation_metrics": metrics,
    }

    logger.info("checkpoint载荷构造完成。epoch=%s", epoch)
    return payload


def train_model(config: dict[str, Any], model_type: str, logger) -> dict[str, Any]:
    ddp_ctx = _ddp_context()
    _init_ddp_if_needed(ddp_ctx)
    processed_dir = Path(config["data"]["processed_dir"]).expanduser().resolve()
    output_root = Path(config["outputs"]["root_dir"]).expanduser().resolve()
    run_dir = output_root / model_type / f"seed_{config['training']['seed']}"
    if int(config.get("model", {}).get("duration_ratio_buckets", 0)) > 1:
        config.setdefault("model", {})
        config["model"]["duration_ratio_bucket_edges"] = _duration_bucket_edges_from_train(
            processed_dir,
            int(config["model"]["duration_ratio_buckets"]),
            logger,
        )
    if int(config.get("model", {}).get("duration_prior_buckets", 0)) > 1:
        config.setdefault("model", {})
        prior_edges, prior_values = _duration_prior_from_train(
            processed_dir,
            int(config["model"]["duration_prior_buckets"]),
            logger,
        )
        config["model"]["duration_prior_bucket_edges"] = prior_edges
        config["model"]["duration_prior_values"] = prior_values
    if ddp_ctx["is_main"]:
        run_dir.mkdir(parents=True, exist_ok=True)
        dump_config(config, run_dir / "resolved_config.yaml")
    if ddp_ctx["enabled"]:
        dist.barrier()

    set_seed(int(config["training"]["seed"]) + int(ddp_ctx["rank"]))
    manifest = load_manifest(processed_dir)
    loaders = _make_loaders(config, processed_dir, ddp_ctx)
    time_upper_bounds = torch.tensor(manifest["time_bin_upper_bounds_sec"], dtype=torch.float32)
    primary_metric = config.get("evaluation", {}).get(
        "primary_metric",
        "exact_joint_nll",
    )
    if ddp_ctx["enabled"] and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(ddp_ctx['local_rank'])}")
    else:
        device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    model = _build_model(model_type, manifest, config).to(device)
    if ddp_ctx["enabled"]:
        model = DistributedDataParallel(
            model,
            device_ids=[int(ddp_ctx["local_rank"])] if device.type == "cuda" else None,
            output_device=int(ddp_ctx["local_rank"]) if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizers = _build_optimizers(model, config)

    if ddp_ctx["is_main"]:
        logger.info(
            "Training %s on %s | ddp=%s world_size=%s",
            model_type,
            device,
            ddp_ctx["enabled"],
            ddp_ctx["world_size"],
        )
        logger.info("DSI评估主指标: %s", primary_metric)
    parameter_count = int(sum(param.numel() for param in _model_state_dict(model).values()))
    best_val = float("inf")
    best_path = run_dir / "best.pt"
    bad_epochs = 0
    grad_overflow_steps = 0
    collapse_steps = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        start = time.time()
        model.train()
        train_losses: list[float] = []
        loss_sums: dict[str, float] = {}
        grad_norms: list[float] = []
        for step_idx, batch in enumerate(loaders["standard_train"]):
            batch = batch_to_device(batch, device)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            output = model(batch)

            if step_idx == 0:
                logger.info("开始DSI observation-set posterior训练epoch。epoch=%s", epoch)

            if model_type == "time_only_posterior":
                loss_parts = time_only_posterior_losses(
                    output,
                    batch,
                    float(config["loss"].get("lambda_obs_set_restore", 1.0)),
                    time_upper_bounds.to(device),
                    str(config["loss"].get("obs_set_restore_mode", "log")),
                )
            else:
                loss_parts = observation_set_posterior_losses(
                    output,
                    batch,
                    float(config["loss"].get("lambda_obs_set_restore", 1.0)),
                    time_upper_bounds.to(device),
                    str(config["loss"].get("obs_set_overplay_loss", "observed")),
                    int(config.get("event_taxonomy", {}).get("overplay_event_id", 3)),
                    float(config.get("event_taxonomy", {}).get("overplay_band_beta", config.get("event_taxonomy", {}).get("beta", 1.1))),
                    str(config["loss"].get("obs_set_restore_mode", "log")),
                    float(config["loss"].get("lambda_obs_set_survival", 0.0)),
                    float(config["loss"].get("lambda_obs_set_ratio", 0.0)),
                    float(config["loss"].get("lambda_duration_survival", 0.0)),
                    float(config["loss"].get("lambda_seconds_aux", 0.1)),
                )

            if step_idx == 0:
                logger.info("开始检查事件边际坍缩。")

            final_event_mass = posterior_event_mass_from_output(
                output,
                batch["duration_sec"],
                time_upper_bounds.to(device),
            )

            final_event_prob = final_event_mass.detach().sum(dim=2).mean(dim=0)
            if step_idx == 0:
                logger.info("事件边际坍缩检查完成。")
            if torch.any(final_event_prob < 1e-5):
                collapse_steps += 1
            else:
                collapse_steps = 0
            if step_idx == 0:
                logger.info("DSI observation-set posterior训练epoch首步完成。epoch=%s", epoch)

            loss = loss_parts["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became NaN or Inf")
            loss.backward()
            norm = _grad_norm(model.parameters())
            grad_norms.append(norm)
            grad_overflow_steps = grad_overflow_steps + 1 if norm > 100.0 else 0
            if grad_overflow_steps >= 50:
                raise FloatingPointError("Gradient global norm exceeded 100 for 50 consecutive steps")
            if collapse_steps >= 500:
                raise FloatingPointError("At least one event hazard collapsed below 1e-5 for 500 steps")
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["clip_grad_norm"]))
            for optimizer in optimizers:
                optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            for key, value in loss_parts.items():
                if key == "total":
                    continue
                if isinstance(value, torch.Tensor):
                    scalar_value = float(value.detach().cpu())
                else:
                    scalar_value = float(value)
                loss_sums[key] = loss_sums.get(key, 0.0) + scalar_value

        local_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        local_grad_norm_mean = float(np.mean(grad_norms)) if grad_norms else 0.0
        local_grad_norm_max = float(np.max(grad_norms)) if grad_norms else 0.0
        train_loss_mean = _ddp_mean(local_train_loss, device, ddp_ctx)
        grad_norm_mean = _ddp_mean(local_grad_norm_mean, device, ddp_ctx)
        grad_norm_max = _ddp_mean(local_grad_norm_max, device, ddp_ctx)
        stop_now = False
        if ddp_ctx["is_main"]:
            logger.info("开始训练期轻量validation。epoch=%s", epoch)

            val_result = evaluate_model_light(
                model.module if isinstance(model, DistributedDataParallel) else model,
                loaders["standard_val"],
                model_type,
                device,
                manifest["time_bin_upper_bounds_sec"],
                manifest["event_names"],
                config["evaluation"]["max_batches"],
                config["evaluation"]["ece_bins"],
                primary_metric,
            )
            logger.info("训练期轻量validation完成。epoch=%s", epoch)
            val_nll = val_result.metrics["primary_nll"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss_mean,
                "val_primary_nll": val_nll,
                "val_competing_risk_nll": val_result.metrics["competing_risk_nll"],
                "val_wcr_nll": val_result.metrics["wcr_nll"],
                "val_macro_ece": val_result.metrics["macro_ece"],
                "val_neutral": val_result.metrics.get("neutral_metrics"),
                "train_loss_parts": {key: value / max(len(train_losses), 1) for key, value in loss_sums.items()},
                "grad_norm_mean": grad_norm_mean,
                "grad_norm_max": grad_norm_max,
                "seconds": time.time() - start,
            }
            history.append(row)
            logger.info(
                "Epoch %d | train %.6f | val_primary_nll %.6f | val_ece %.6f",
                epoch,
                row["train_loss"],
                val_nll,
                row["val_macro_ece"],
            )

            if val_nll < best_val:
                best_val = val_nll
                bad_epochs = 0
                torch.save(_checkpoint_payload(model, manifest, config, epoch, val_result.metrics), best_path)
            else:
                bad_epochs += 1
                if bad_epochs >= int(config["training"]["patience"]):
                    logger.info("Early stopping at epoch %d", epoch)
                    stop_now = True
        if _ddp_stop_signal(stop_now, device, ddp_ctx):
            break

    if not ddp_ctx["is_main"]:
        _cleanup_ddp_if_needed(ddp_ctx)
        return {}
    checkpoint = torch.load(best_path, map_location=device)
    _load_model_state_dict(model, checkpoint["model"])
    eval_model = model.module if isinstance(model, DistributedDataParallel) else model
    calibration = None
    calibration_cfg = config.get("calibration", {})
    validation_raw_result = None
    duration_edges = np.asarray([], dtype=np.float32)
    if bool(calibration_cfg.get("enabled", False)):
        calibration_type = str(calibration_cfg.get("type", "event_marginal"))
        if calibration_type != "event_marginal":
            raise ValueError(f"DSI now supports event_marginal calibration only: {calibration_type}")
        logger.info("开始validation-only event-marginal calibration流程。")
        validation_raw_result = evaluate_model(
            eval_model,
            loaders["standard_val"],
            model_type,
            device,
            manifest["time_bin_upper_bounds_sec"],
            manifest["event_names"],
            config["evaluation"]["max_batches"],
            config["evaluation"]["ece_bins"],
            config["evaluation"]["cindex_buckets"],
            config["evaluation"]["cindex_max_pairs_per_bucket"],
            int(config["training"]["seed"]),
            primary_metric,
        )
        duration_edges = _duration_quantile_edges(
            validation_raw_result.arrays,
            int(config["evaluation"]["cindex_buckets"]),
        )
        calibration = fit_event_marginal_calibration_from_arrays(
            validation_raw_result.arrays,
            int(calibration_cfg.get("max_steps", 200)),
            logger,
        )
        logger.info("validation-only event-marginal calibration流程完成。")
    test_raw_result = _evaluate_split(
        eval_model,
        loaders["standard_test"],
        model_type,
        device,
        manifest,
        config,
        int(config["training"]["seed"]),
        primary_metric,
        "standard_test_raw",
        None,
        logger,
    ) if calibration is not None else None
    test_result = _evaluate_split(
        eval_model,
        loaders["standard_test"],
        model_type,
        device,
        manifest,
        config,
        int(config["training"]["seed"]),
        primary_metric,
        "standard_test",
        calibration,
        logger,
    )
    if test_raw_result is not None:
        _save_eval_artifacts(run_dir, "standard_test_raw", test_raw_result, logger)
    _save_eval_artifacts(run_dir, "standard_test", test_result, logger)
    if duration_edges.size == 0:
        duration_edges = _duration_quantile_edges(
            test_result.arrays,
            int(config["evaluation"]["cindex_buckets"]),
        )

    metrics = {
        "model_type": model_type,
        "run_dir": str(run_dir),
        "parameter_count": parameter_count,
        "git": _git_metadata(Path(__file__).resolve().parents[1]),
        "source_hashes": _source_hashes(Path(__file__).resolve().parents[1]),
        "peak_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
        "duration_quintile_edges": duration_edges.tolist(),
        "best_val_primary_nll": best_val,
        "best_val_competing_risk_nll": best_val,
        "calibration": calibration,
        "validation_raw": validation_raw_result.metrics if validation_raw_result is not None else None,
        "test_raw": test_raw_result.metrics if test_raw_result is not None else None,
        "test": test_result.metrics,
        "history": history,
    }
    serialized_metrics = portableize_serialized_paths(metrics)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(serialized_metrics, handle, indent=2, ensure_ascii=False)
    logger.info("Finished %s. Test primary NLL %.6f", model_type, metrics["test"]["primary_nll"])
    _cleanup_ddp_if_needed(ddp_ctx)
    return metrics
