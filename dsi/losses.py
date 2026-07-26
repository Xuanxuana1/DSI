from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from dsi.event_mass import observation_set_posterior_event_mass_from_output
from dsi.event_mass import ratio_observation_set_posterior_event_mass_from_output
from dsi.event_mass import time_only_posterior_event_mass_from_output

logger = logging.getLogger(__name__)
_LOG_ONCE_KEYS: set[str] = set()


def _log_once(key: str, message: str, *args: object) -> None:

    if key in _LOG_ONCE_KEYS:
        return

    logger.info(message, *args)
    _LOG_ONCE_KEYS.add(key)


def cdf_and_survival_from_event_mass(event_mass: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cdf = torch.cumsum(event_mass, dim=2).clamp(max=1.0)
    survival = (1.0 - cdf).clamp_min(1e-12)
    return cdf.clamp_min(1e-12), survival


def target_event_value(values: torch.Tensor, event: torch.Tensor, time_bin: torch.Tensor) -> torch.Tensor:
    batch_idx = torch.arange(event.shape[0], device=event.device)
    return values[batch_idx, event, time_bin].clamp_min(1e-12)


def competing_risk_nll_from_event_mass(
    event_mass: torch.Tensor,
    event: torch.Tensor,
    time_bin: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    cdf, survival = cdf_and_survival_from_event_mass(event_mass)
    batch_idx = torch.arange(event.shape[0], device=event.device)
    target_cdf = cdf[batch_idx, event, time_bin].clamp_min(1e-12)
    survival_at_t = survival[batch_idx, :, time_bin]
    event_mask = torch.nn.functional.one_hot(event, num_classes=event_mass.shape[1]).bool()
    other_survival = survival_at_t.masked_fill(event_mask, 1.0)
    per_sample = -torch.log(target_cdf) - torch.log(other_survival.clamp_min(1e-12)).sum(dim=1)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def observation_set_posterior_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    lambda_restore: float,
    time_upper_bounds: torch.Tensor,
    overplay_loss_mode: str = "observed",
    overplay_event_id: int = 3,
    beta: float = 1.1,
    restore_mode: str = "log",
    lambda_survival: float = 0.0,
    lambda_ratio: float = 0.0,
    lambda_duration_survival: float = 0.0,
    lambda_seconds_aux: float = 0.1,
) -> dict[str, torch.Tensor]:
    _log_once("obs_set_posterior_start", "Computing the DSI observation-set posterior training objective...")

    watch_mass = observation_set_posterior_event_mass_from_output(
        output,
        batch["duration_sec"],
        time_upper_bounds,
    )
    sample_idx = torch.arange(batch["watch_event"].shape[0], device=batch["watch_event"].device)
    event = batch["watch_event"].to(device=watch_mass.device).clamp(max=watch_mass.shape[1] - 1)
    observed_bin = batch["time_bin"].to(device=watch_mass.device).clamp(max=watch_mass.shape[2] - 1)
    duration = batch["duration_sec"].to(device=watch_mass.device, dtype=watch_mass.dtype).clamp_min(1e-3)
    upper = time_upper_bounds.to(device=watch_mass.device, dtype=watch_mass.dtype)
    completion_id = int(output.get("true_completion_event_id", 2))
    overplay_id = int(overplay_event_id)

    point_mass = watch_mass[sample_idx, event, observed_bin].clamp_min(1e-12)
    obs_mass = point_mass

    if 0 <= completion_id < watch_mass.shape[1]:
        completion_mask = event == completion_id
        if torch.any(completion_mask):
            completion_mass = watch_mass[:, completion_id, :].sum(dim=1).clamp_min(1e-12)
            obs_mass = torch.where(completion_mask, completion_mass, obs_mass)

    if 0 <= overplay_id < watch_mass.shape[1]:
        overplay_mask = event == overplay_id
        if torch.any(overplay_mask):
            mode = str(overplay_loss_mode)
            if mode == "observed":
                overplay_mass = watch_mass[sample_idx, overplay_id, observed_bin].clamp_min(1e-12)
            elif mode == "interval":
                threshold = (duration * float(beta)).clamp_min(1e-3)
                threshold_bin = torch.bucketize(threshold, upper, right=False).clamp(max=watch_mass.shape[2] - 1)
                time_grid = torch.arange(watch_mass.shape[2], device=watch_mass.device).view(1, -1)
                interval_mask = (time_grid >= threshold_bin.view(-1, 1)) & (time_grid <= observed_bin.view(-1, 1))
                overplay_mass = (watch_mass[:, overplay_id, :] * interval_mask.to(watch_mass.dtype)).sum(dim=1).clamp_min(1e-12)
            else:
                raise ValueError(f"Unsupported observation-set overplay_loss_mode: {overplay_loss_mode}")
            obs_mass = torch.where(overplay_mask, overplay_mass, obs_mass)

    watch_loss = -torch.log(obs_mass.clamp_min(1e-12)).mean()
    restored_time = (watch_mass.sum(dim=1) * upper.view(1, -1)).sum(dim=1)
    target_time = batch["play_time_sec"].to(device=watch_mass.device, dtype=watch_mass.dtype).clamp_min(0.0)
    restore_log_huber = F.smooth_l1_loss(
        torch.log1p(restored_time.clamp_min(0.0)),
        torch.log1p(target_time),
    )
    restore_seconds_huber = F.smooth_l1_loss(restored_time.clamp_min(0.0), target_time)
    mode = str(restore_mode)
    if mode == "log":
        restore_huber = restore_log_huber
    elif mode == "seconds":
        restore_huber = restore_seconds_huber
    elif mode == "both":
        restore_huber = restore_log_huber + restore_seconds_huber
    else:
        raise ValueError(f"Unsupported observation-set restore_mode: {restore_mode}")
    event_prob = watch_mass.sum(dim=2).clamp_min(1e-12)

    _log_once("obs_set_survival_start", "Computing optional survival-curve supervision.")

    survival_bce = watch_loss.new_zeros(())
    if float(lambda_survival) > 0.0:

        time_prob = watch_mass.sum(dim=1).clamp_min(1e-12)
        survival_prob = (1.0 - torch.cumsum(time_prob, dim=1)).clamp(1e-6, 1.0 - 1e-6)

        survival_target = (target_time.view(-1, 1) > upper.view(1, -1)).to(dtype=watch_mass.dtype)

        survival_bce = F.binary_cross_entropy(survival_prob, survival_target)

    _log_once("obs_set_survival_done", "Optional survival-curve supervision computed. enabled=%s", bool(float(lambda_survival) > 0.0))

    _log_once("ratio_obs_set_start", "Computing optional ratio-aware support supervision.")

    ratio_nll = watch_loss.new_zeros(())
    ratio_restore_huber = watch_loss.new_zeros(())
    ratio_survival_bce = watch_loss.new_zeros(())
    primary_axis = str(output.get("posterior_primary_axis", "seconds"))
    ratio_enabled = float(lambda_ratio) > 0.0 or primary_axis == "ratio"
    if ratio_enabled:

        ratio_mass = ratio_observation_set_posterior_event_mass_from_output(output)
        ratio_upper = output["ratio_upper_bounds"].to(device=watch_mass.device, dtype=watch_mass.dtype)
        observed_ratio = (target_time / duration).clamp(0.0, float(ratio_upper[-1]))
        observed_ratio_bin = torch.bucketize(observed_ratio, ratio_upper, right=False).clamp(max=ratio_mass.shape[2] - 1)

        ratio_point_mass = ratio_mass[sample_idx, event, observed_ratio_bin].clamp_min(1e-12)
        ratio_obs_mass = ratio_point_mass
        if 0 <= completion_id < ratio_mass.shape[1]:
            completion_mask = event == completion_id
            if torch.any(completion_mask):
                ratio_completion_mass = ratio_mass[:, completion_id, :].sum(dim=1).clamp_min(1e-12)
                ratio_obs_mass = torch.where(completion_mask, ratio_completion_mass, ratio_obs_mass)
        if 0 <= overplay_id < ratio_mass.shape[1]:
            overplay_mask = event == overplay_id
            if torch.any(overplay_mask):
                mode = str(overplay_loss_mode)
                if mode == "observed":
                    ratio_overplay_mass = ratio_mass[sample_idx, overplay_id, observed_ratio_bin].clamp_min(1e-12)
                elif mode == "interval":
                    ratio_threshold_bin = torch.bucketize(
                        target_time.new_full(target_time.shape, float(beta)),
                        ratio_upper,
                        right=False,
                    ).clamp(max=ratio_mass.shape[2] - 1)
                    ratio_grid = torch.arange(ratio_mass.shape[2], device=ratio_mass.device).view(1, -1)
                    ratio_interval_mask = (ratio_grid >= ratio_threshold_bin.view(-1, 1)) & (ratio_grid <= observed_ratio_bin.view(-1, 1))
                    ratio_overplay_mass = (ratio_mass[:, overplay_id, :] * ratio_interval_mask.to(ratio_mass.dtype)).sum(dim=1).clamp_min(1e-12)
                else:
                    raise ValueError(f"Unsupported observation-set overplay_loss_mode: {overplay_loss_mode}")
                ratio_obs_mass = torch.where(overplay_mask, ratio_overplay_mass, ratio_obs_mass)
        ratio_nll = -torch.log(ratio_obs_mass.clamp_min(1e-12)).mean()

        ratio_time_grid = ratio_upper.view(1, -1) * duration.view(-1, 1)
        ratio_restored_time = (ratio_mass.sum(dim=1) * ratio_time_grid).sum(dim=1)
        if str(restore_mode) == "log":
            ratio_restore_huber = F.smooth_l1_loss(torch.log1p(ratio_restored_time.clamp_min(0.0)), torch.log1p(target_time))
        elif str(restore_mode) == "seconds":
            ratio_restore_huber = F.smooth_l1_loss(ratio_restored_time.clamp_min(0.0), target_time)
        elif str(restore_mode) == "both":
            ratio_restore_huber = (
                F.smooth_l1_loss(torch.log1p(ratio_restored_time.clamp_min(0.0)), torch.log1p(target_time))
                + F.smooth_l1_loss(ratio_restored_time.clamp_min(0.0), target_time)
            )
        else:
            raise ValueError(f"Unsupported observation-set restore_mode: {restore_mode}")
        ratio_prob = ratio_mass.sum(dim=1).clamp_min(1e-12)
        ratio_survival_prob = (1.0 - torch.cumsum(ratio_prob, dim=1)).clamp(1e-6, 1.0 - 1e-6)
        ratio_survival_target = (observed_ratio.view(-1, 1) > ratio_upper.view(1, -1)).to(dtype=watch_mass.dtype)
        ratio_survival_bce = F.binary_cross_entropy(ratio_survival_prob, ratio_survival_target)

    _log_once("ratio_obs_set_done", "Optional ratio-aware support supervision computed. enabled=%s", bool(ratio_enabled))

    _log_once("duration_survival_start", "Computing optional duration-conditioned survival supervision.")

    duration_survival_bce = watch_loss.new_zeros(())
    if float(lambda_duration_survival) > 0.0:
        if "duration_survival_logits" not in output or "duration_survival_ratio_bounds" not in output:
            raise KeyError("duration_survival_logits and duration_survival_ratio_bounds are required when lambda_duration_survival > 0")
        survival_logits = output["duration_survival_logits"]
        survival_ratio = output["duration_survival_ratio_bounds"].to(device=survival_logits.device, dtype=survival_logits.dtype)
        observed_ratio = (target_time / duration).clamp(0.0, float(survival_ratio[-1]))
        survival_target = (observed_ratio.view(-1, 1) > survival_ratio.view(1, -1)).to(dtype=survival_logits.dtype)
        duration_survival_bce = F.binary_cross_entropy_with_logits(survival_logits, survival_target)

    _log_once(
        "duration_survival_done",
        "Optional duration-conditioned survival supervision computed. enabled=%s",
        bool(float(lambda_duration_survival) > 0.0),
    )

    watch_cif_nll = competing_risk_nll_from_event_mass(watch_mass, batch["watch_event"], batch["time_bin"])
    watch_exact_nll = exact_event_time_nll_from_event_mass(watch_mass, batch["watch_event"], batch["time_bin"])

    if primary_axis == "ratio":
        total = (
            ratio_nll
            + float(lambda_restore) * ratio_restore_huber
            + float(lambda_survival) * ratio_survival_bce
            + float(lambda_seconds_aux) * (watch_loss + float(lambda_restore) * restore_huber + float(lambda_survival) * survival_bce)
            + float(lambda_duration_survival) * duration_survival_bce
        )
    else:
        total = (
            watch_loss
            + float(lambda_restore) * restore_huber
            + float(lambda_survival) * survival_bce
            + float(lambda_ratio) * (ratio_nll + float(lambda_restore) * ratio_restore_huber + ratio_survival_bce)
            + float(lambda_duration_survival) * duration_survival_bce
        )
    _log_once("obs_set_posterior_done", "DSI observation-set posterior training objective computed.")
    return {
        "total": total,
        "obs_set_nll": watch_loss.detach(),
        "watch_objective_nll": watch_loss.detach(),
        "watch_cif_nll": watch_cif_nll.detach(),
        "watch_exact_nll": watch_exact_nll.detach(),
        "restore_huber": restore_huber.detach(),
        "restore_log_huber": restore_log_huber.detach(),
        "restore_seconds_huber": restore_seconds_huber.detach(),
        "survival_bce": survival_bce.detach(),
        "ratio_nll": ratio_nll.detach(),
        "ratio_restore_huber": ratio_restore_huber.detach(),
        "ratio_survival_bce": ratio_survival_bce.detach(),
        "duration_survival_bce": duration_survival_bce.detach(),
        "seconds_aux_weight": watch_loss.new_tensor(float(lambda_seconds_aux)).detach(),
        "completion_prob_mean": event_prob[:, completion_id].mean().detach() if 0 <= completion_id < event_prob.shape[1] else watch_loss.new_tensor(0.0),
        "overplay_prob_mean": event_prob[:, overplay_id].mean().detach() if 0 <= overplay_id < event_prob.shape[1] else watch_loss.new_tensor(0.0),
    }


def time_only_posterior_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    lambda_restore: float,
    time_upper_bounds: torch.Tensor,
    restore_mode: str = "log",
) -> dict[str, torch.Tensor]:
    _log_once("time_only_posterior_start", "Computing the time-only posterior training objective.")

    logits = output["obs_set_logits"]
    time_mass = torch.softmax(logits[:, 0, :], dim=1)
    observed_bin = batch["time_bin"].to(device=time_mass.device).clamp(max=time_mass.shape[1] - 1)
    sample_idx = torch.arange(observed_bin.shape[0], device=time_mass.device)
    watch_loss = -torch.log(time_mass[sample_idx, observed_bin].clamp_min(1e-12)).mean()

    upper = time_upper_bounds.to(device=time_mass.device, dtype=time_mass.dtype)
    restored_time = (time_mass * upper.view(1, -1)).sum(dim=1)
    target_time = batch["play_time_sec"].to(device=time_mass.device, dtype=time_mass.dtype).clamp_min(0.0)
    restore_log_huber = F.smooth_l1_loss(
        torch.log1p(restored_time.clamp_min(0.0)),
        torch.log1p(target_time),
    )
    restore_seconds_huber = F.smooth_l1_loss(restored_time.clamp_min(0.0), target_time)
    if str(restore_mode) == "log":
        restore_huber = restore_log_huber
    elif str(restore_mode) == "seconds":
        restore_huber = restore_seconds_huber
    elif str(restore_mode) == "both":
        restore_huber = restore_log_huber + restore_seconds_huber
    else:
        raise ValueError(f"Unsupported time-only restore_mode: {restore_mode}")

    event_mass = time_only_posterior_event_mass_from_output(output, batch["duration_sec"], upper)
    event_prob = event_mass.sum(dim=2).clamp_min(1e-12)
    watch_cif_nll = competing_risk_nll_from_event_mass(event_mass, batch["watch_event"], batch["time_bin"])
    watch_exact_nll = exact_event_time_nll_from_event_mass(event_mass, batch["watch_event"], batch["time_bin"])
    total = watch_loss + float(lambda_restore) * restore_huber
    _log_once("time_only_posterior_done", "Time-only posterior training objective computed.")
    zero = watch_loss.new_tensor(0.0)
    return {
        "total": total,
        "obs_set_nll": watch_loss.detach(),
        "watch_objective_nll": watch_loss.detach(),
        "watch_cif_nll": watch_cif_nll.detach(),
        "watch_exact_nll": watch_exact_nll.detach(),
        "restore_huber": restore_huber.detach(),
        "restore_log_huber": restore_log_huber.detach(),
        "restore_seconds_huber": restore_seconds_huber.detach(),
        "survival_bce": zero,
        "ratio_nll": zero,
        "ratio_restore_huber": zero,
        "ratio_survival_bce": zero,
        "duration_survival_bce": zero,
        "seconds_aux_weight": zero,
        "completion_prob_mean": event_prob[:, 2].mean().detach(),
        "overplay_prob_mean": event_prob[:, 3].mean().detach(),
    }


def exact_event_time_nll_from_event_mass(
    event_mass: torch.Tensor,
    event: torch.Tensor,
    time_bin: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    batch_idx = torch.arange(event.shape[0], device=event.device)
    per_sample = -torch.log(event_mass[batch_idx, event, time_bin].clamp_min(1e-12))
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def window_mass_from_event_mass(event_mass: torch.Tensor, time_bin: torch.Tensor) -> torch.Tensor:
    device = event_mass.device
    batch_size, num_events, num_bins = event_mass.shape

    offsets = torch.tensor([-1, 0, 1], device=device, dtype=torch.long)
    base_kernel = torch.tensor([0.25, 0.50, 0.25], device=device, dtype=event_mass.dtype)
    window_idx = time_bin.view(-1, 1) + offsets.view(1, -1)
    valid = (window_idx >= 0) & (window_idx < num_bins)
    clipped_idx = window_idx.clamp(min=0, max=num_bins - 1)

    kernel = base_kernel.view(1, -1) * valid.to(event_mass.dtype)
    kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-12)

    gather_idx = clipped_idx.view(batch_size, 1, 3).expand(-1, num_events, -1)
    window_values = torch.gather(event_mass, dim=2, index=gather_idx)
    return (window_values * kernel.view(batch_size, 1, 3)).sum(dim=2).clamp_min(1e-12)


def windowed_competing_risk_nll_from_event_mass(
    event_mass: torch.Tensor,
    event: torch.Tensor,
    time_bin: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    cdf, survival = cdf_and_survival_from_event_mass(event_mass)
    batch_idx = torch.arange(event.shape[0], device=event.device)

    window_mass = window_mass_from_event_mass(event_mass, time_bin)
    target_window_mass = window_mass[batch_idx, event].clamp_min(1e-12)

    survival_at_t = survival[batch_idx, :, time_bin]
    event_mask = torch.nn.functional.one_hot(event, num_classes=event_mass.shape[1]).bool()
    other_survival = survival_at_t.masked_fill(event_mask, 1.0)

    per_sample = -torch.log(target_window_mass) - torch.log(other_survival.clamp_min(1e-12)).sum(dim=1)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")
