from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import torch

from dsi.calibration import apply_event_marginal_calibration
from dsi.decision.neutral import neutral_metric_summary
from dsi.event_mass import posterior_event_mass_from_output
from dsi.losses import (
    competing_risk_nll_from_event_mass,
    exact_event_time_nll_from_event_mass,
    windowed_competing_risk_nll_from_event_mass,
)


@dataclass


class EvalResult:
    metrics: dict[str, Any]
    arrays: dict[str, np.ndarray]


def move_eval_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:

    eval_keys = {
        "user_idx",
        "item_idx",
        "tab",
        "prev_event",
        "numeric",
        "content_category",
        "content_numeric",
        "raw_sparse",
        "raw_numeric",
        "raw_seq",
        "raw_seq_mask",
        "duration_sec",
        "watch_event",
        "event",
        "time_bin",
        "play_time_sec",
        "session_key",
    }

    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key in eval_keys
    }


def _batch_event_mass(model_type: str, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], upper_torch: torch.Tensor) -> torch.Tensor:

    if model_type not in {"obs_set_posterior", "time_only_posterior"}:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return posterior_event_mass_from_output(output, batch["duration_sec"], upper_torch)


def expected_time_seconds_torch(event_mass: torch.Tensor, upper_bounds: torch.Tensor) -> torch.Tensor:

    marginal_t = event_mass.sum(dim=1)
    mass = marginal_t.sum(dim=1).clamp_min(1e-12)
    return (marginal_t * upper_bounds.view(1, -1)).sum(dim=1) / mass


def _mixed_expected_time(output: dict[str, torch.Tensor], event_mass: torch.Tensor, upper_bounds: torch.Tensor) -> torch.Tensor:

    positive_expected = expected_time_seconds_torch(event_mass, upper_bounds)
    if "zero_logit" not in output:
        return positive_expected

    zero_prob = torch.sigmoid(output["zero_logit"]).to(dtype=positive_expected.dtype)
    return (1.0 - zero_prob) * positive_expected


def _mixed_event_prob(output: dict[str, torch.Tensor], event_mass: torch.Tensor, zero_event_id: int) -> torch.Tensor:

    event_prob = event_mass.sum(dim=2)
    if "zero_logit" not in output:
        return event_prob

    zero_prob = torch.sigmoid(output["zero_logit"]).to(dtype=event_prob.dtype).view(-1, 1)
    mixed = (1.0 - zero_prob) * event_prob
    if 0 <= int(zero_event_id) < mixed.shape[1]:
        mixed = mixed.clone()
        mixed[:, int(zero_event_id)] = mixed[:, int(zero_event_id)] + zero_prob.squeeze(1)
    return mixed


def _mixed_nll(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], positive_nll: torch.Tensor, zero_threshold_sec: float = 1e-6) -> torch.Tensor:

    if "zero_logit" not in output:
        return positive_nll

    zero_prob = torch.sigmoid(output["zero_logit"]).clamp(1e-6, 1.0 - 1e-6)
    zero_mask = batch["play_time_sec"] <= float(zero_threshold_sec)
    zero_nll = -torch.log(zero_prob)
    nonzero_nll = -torch.log((1.0 - zero_prob).clamp_min(1e-6)) + positive_nll
    return torch.where(zero_mask, zero_nll, nonzero_nll)


def ece_score(pred: np.ndarray, label: np.ndarray, bins: int) -> float:

    values = np.asarray(pred, dtype=np.float64).reshape(-1)
    targets = np.asarray(label, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = max(int(values.shape[0]), 1)
    score = 0.0

    for idx, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if idx + 1 == len(edges) - 1:
            mask = (values >= low) & (values <= high)
        else:
            mask = (values >= low) & (values < high)
        if not np.any(mask):
            continue
        score += (float(mask.sum()) / total) * abs(float(values[mask].mean()) - float(targets[mask].mean()))
    return float(score)


@torch.no_grad()


def evaluate_model_light(
    model: torch.nn.Module,
    loader,
    model_type: str,
    device: torch.device,
    time_upper_bounds: list[float],
    event_names: list[str],
    max_batches: int | None,
    ece_bins: int,
    primary_metric: str = "competing_risk_nll",
) -> EvalResult:
    logger = __import__("logging").getLogger(__name__)
    logger.info("开始执行DSI轻量评估。rows=%s", len(loader.dataset))

    model.eval()
    started_at = time.time()
    nll_sum = 0.0
    exact_nll_sum = 0.0
    wcr_nll_sum = 0.0
    primary_nll_sum = 0.0
    nll_count = 0
    num_events = int(len(event_names))
    ece_edges = np.linspace(0.0, 1.0, int(ece_bins) + 1, dtype=np.float64)
    ece_count = np.zeros((num_events, int(ece_bins)), dtype=np.float64)
    ece_pred_sum = np.zeros((num_events, int(ece_bins)), dtype=np.float64)
    ece_label_sum = np.zeros((num_events, int(ece_bins)), dtype=np.float64)
    upper_torch = torch.as_tensor(time_upper_bounds, dtype=torch.float32, device=device)

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_eval_batch_to_device(batch, device)
        output = model(batch)
        target_event = batch["watch_event"]
        event_mass = _batch_event_mass(model_type, output, batch, upper_torch.to(dtype=batch["duration_sec"].dtype))
        per_sample_nll = _mixed_nll(
            output,
            batch,
            competing_risk_nll_from_event_mass(event_mass, target_event, batch["time_bin"], reduction="none"),
        )
        per_sample_exact_nll = _mixed_nll(
            output,
            batch,
            exact_event_time_nll_from_event_mass(event_mass, target_event, batch["time_bin"], reduction="none"),
        )
        per_sample_wcr_nll = _mixed_nll(
            output,
            batch,
            windowed_competing_risk_nll_from_event_mass(event_mass, target_event, batch["time_bin"], reduction="none"),
        )
        if primary_metric == "wcr_nll":
            per_sample_primary_nll = per_sample_wcr_nll
        elif primary_metric == "competing_risk_nll":
            per_sample_primary_nll = per_sample_nll
        elif primary_metric == "exact_joint_nll":
            per_sample_primary_nll = per_sample_exact_nll
        else:
            raise ValueError(f"Unsupported primary_metric: {primary_metric}")

        batch_size = int(per_sample_nll.numel())
        nll_sum += float(per_sample_nll.detach().sum().cpu())
        exact_nll_sum += float(per_sample_exact_nll.detach().sum().cpu())
        wcr_nll_sum += float(per_sample_wcr_nll.detach().sum().cpu())
        primary_nll_sum += float(per_sample_primary_nll.detach().sum().cpu())
        nll_count += batch_size

        event_prob = _mixed_event_prob(output, event_mass, 0).detach().cpu().numpy()
        event_label = target_event.detach().cpu().numpy().reshape(-1)
        for event_idx in range(min(num_events, event_prob.shape[1])):
            pred = event_prob[:, event_idx].astype(np.float64, copy=False)
            label = (event_label == event_idx).astype(np.float64)
            bins = np.searchsorted(ece_edges, pred, side="right") - 1
            bins = np.clip(bins, 0, int(ece_bins) - 1)
            ece_count[event_idx] += np.bincount(bins, minlength=int(ece_bins)).astype(np.float64)
            ece_pred_sum[event_idx] += np.bincount(bins, weights=pred, minlength=int(ece_bins)).astype(np.float64)
            ece_label_sum[event_idx] += np.bincount(bins, weights=label, minlength=int(ece_bins)).astype(np.float64)

    event_ece: dict[str, float] = {}
    total_rows = max(int(nll_count), 1)
    for event_idx, name in enumerate(event_names[:num_events]):
        count = ece_count[event_idx]
        nonzero = count > 0
        if not np.any(nonzero):
            event_ece[name] = float("nan")
            continue
        pred_mean = np.divide(ece_pred_sum[event_idx], count, out=np.zeros_like(count), where=nonzero)
        label_mean = np.divide(ece_label_sum[event_idx], count, out=np.zeros_like(count), where=nonzero)
        event_ece[name] = float(np.sum((count[nonzero] / total_rows) * np.abs(pred_mean[nonzero] - label_mean[nonzero])))

    elapsed = max(time.time() - started_at, 1e-12)
    metrics = {
        "primary_metric": primary_metric,
        "primary_nll": float(primary_nll_sum / total_rows),
        "competing_risk_nll": float(nll_sum / total_rows),
        "exact_joint_nll": float(exact_nll_sum / total_rows),
        "wcr_nll": float(wcr_nll_sum / total_rows),
        "macro_ece": float(np.nanmean(list(event_ece.values()))) if event_ece else float("nan"),
        "event_ece": event_ece,
        "neutral_metrics": None,
        "num_eval_rows": int(nll_count),
        "eval_seconds": float(elapsed),
        "eval_rows_per_second": float(nll_count / elapsed),
        "light_eval": True,
    }
    logger.info("DSI轻量评估完成。rows=%s primary_nll=%.6f", int(nll_count), metrics["primary_nll"])
    return EvalResult(metrics=metrics, arrays={})


@torch.no_grad()


def evaluate_model(
    model: torch.nn.Module,
    loader,
    model_type: str,
    device: torch.device,
    time_upper_bounds: list[float],
    event_names: list[str],
    max_batches: int | None,
    ece_bins: int,
    cindex_buckets: int,
    cindex_max_pairs: int,
    seed: int,
    primary_metric: str = "competing_risk_nll",
    calibration: dict[str, Any] | None = None,
    store_event_mass: bool = True,
) -> EvalResult:
    model.eval()
    started_at = time.time()

    nll_sum = 0.0
    exact_nll_sum = 0.0
    wcr_nll_sum = 0.0
    primary_nll_sum = 0.0
    nll_count = 0
    event_mass_parts: list[np.ndarray] = []
    nll_parts: list[np.ndarray] = []
    primary_nll_parts: list[np.ndarray] = []
    event_parts: list[np.ndarray] = []
    watch_event_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    duration_parts: list[np.ndarray] = []
    session_parts: list[np.ndarray] = []
    play_time_parts: list[np.ndarray] = []
    event_prob_parts: list[np.ndarray] = []
    expected_time_parts: list[np.ndarray] = []
    upper = np.asarray(time_upper_bounds, dtype=np.float32)
    upper_torch = torch.as_tensor(time_upper_bounds, dtype=torch.float32, device=device)
    early_skip_idx = event_names.index("early_skip") if "early_skip" in event_names else 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_eval_batch_to_device(batch, device)
        output = model(batch)
        target_event = batch["watch_event"]
        event_mass = _batch_event_mass(model_type, output, batch, upper_torch.to(dtype=batch["duration_sec"].dtype))
        event_mass = apply_event_marginal_calibration(event_mass, calibration)

        per_sample_nll = _mixed_nll(
            output,
            batch,
            competing_risk_nll_from_event_mass(event_mass, target_event, batch["time_bin"], reduction="none"),
        )
        per_sample_exact_nll = _mixed_nll(
            output,
            batch,
            exact_event_time_nll_from_event_mass(event_mass, target_event, batch["time_bin"], reduction="none"),
        )
        per_sample_wcr_nll = _mixed_nll(
            output,
            batch,
            windowed_competing_risk_nll_from_event_mass(event_mass, target_event, batch["time_bin"], reduction="none"),
        )
        if primary_metric == "wcr_nll":
            per_sample_primary_nll = per_sample_wcr_nll
        elif primary_metric == "competing_risk_nll":
            per_sample_primary_nll = per_sample_nll
        elif primary_metric == "exact_joint_nll":
            per_sample_primary_nll = per_sample_exact_nll
        else:
            raise ValueError(f"Unsupported primary_metric: {primary_metric}")

        batch_size = int(per_sample_nll.numel())
        nll_sum += float(per_sample_nll.detach().sum().cpu())
        exact_nll_sum += float(per_sample_exact_nll.detach().sum().cpu())
        wcr_nll_sum += float(per_sample_wcr_nll.detach().sum().cpu())
        primary_nll_sum += float(per_sample_primary_nll.detach().sum().cpu())
        nll_count += batch_size
        nll_parts.append(per_sample_nll.detach().cpu().numpy())
        primary_nll_parts.append(per_sample_primary_nll.detach().cpu().numpy())
        if store_event_mass:
            event_mass_parts.append(event_mass.detach().cpu().numpy())
        event_prob_parts.append(_mixed_event_prob(output, event_mass, int(early_skip_idx)).detach().cpu().numpy())
        expected_time_parts.append(_mixed_expected_time(output, event_mass, upper_torch.to(dtype=event_mass.dtype)).detach().cpu().numpy())
        event_parts.append(target_event.detach().cpu().numpy())
        watch_event_parts.append(batch["watch_event"].detach().cpu().numpy())
        time_parts.append(batch["time_bin"].detach().cpu().numpy())
        duration_parts.append(batch["duration_sec"].detach().cpu().numpy())
        session_parts.append(batch["session_key"].detach().cpu().numpy())
        if "play_time_sec" in batch:
            play_time_parts.append(batch["play_time_sec"].detach().cpu().numpy())

    event_mass_all = np.concatenate(event_mass_parts, axis=0) if event_mass_parts else None
    nll_all = np.concatenate(nll_parts, axis=0)
    primary_nll_all = np.concatenate(primary_nll_parts, axis=0)
    event_all = np.concatenate(event_parts, axis=0)
    watch_event_all = np.concatenate(watch_event_parts, axis=0)
    time_all = np.concatenate(time_parts, axis=0)
    duration_all = np.concatenate(duration_parts, axis=0)
    session_all = np.concatenate(session_parts, axis=0)
    event_prob = np.concatenate(event_prob_parts, axis=0)
    observed_time = upper[time_all]
    play_time_all = np.concatenate(play_time_parts, axis=0) if play_time_parts else observed_time
    expected_time = np.concatenate(expected_time_parts, axis=0)

    eval_event_names = event_names[: event_prob.shape[1]]
    event_ece = {
        name: ece_score(event_prob[:, idx], (event_all == idx).astype(np.float32), int(ece_bins))
        for idx, name in enumerate(eval_event_names)
    }
    elapsed = max(time.time() - started_at, 1e-12)
    metrics = {
        "primary_metric": primary_metric,
        "primary_nll": float(primary_nll_sum / max(nll_count, 1)),
        "competing_risk_nll": float(nll_sum / max(nll_count, 1)),
        "exact_joint_nll": float(exact_nll_sum / max(nll_count, 1)),
        "wcr_nll": float(wcr_nll_sum / max(nll_count, 1)),
        "macro_ece": float(np.mean(list(event_ece.values()))) if event_ece else float("nan"),
        "event_ece": event_ece,
        "neutral_metrics": neutral_metric_summary(
            expected_time,
            play_time_all,
            event_prob,
            watch_event_all,
            session_all,
            eval_event_names,
        ),
        "num_eval_rows": int(event_all.shape[0]),
        "eval_seconds": float(elapsed),
        "eval_rows_per_second": float(event_all.shape[0] / elapsed),
    }

    arrays = {
        "event": event_all,
        "watch_event": watch_event_all,
        "time_bin": time_all,
        "duration_sec": duration_all,
        "play_time_sec": play_time_all,
        "session_key": session_all,
        "event_prob": event_prob,
        "per_sample_nll": primary_nll_all,
        "per_sample_primary_nll": primary_nll_all,
        "per_sample_cumulative_nll": nll_all,
        "expected_time_sec": expected_time,
    }
    if event_mass_all is not None:
        arrays["event_mass"] = event_mass_all
    return EvalResult(metrics=metrics, arrays=arrays)
