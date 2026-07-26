from __future__ import annotations

from typing import Any

import numpy as np
import torch

from dsi.losses import competing_risk_nll_from_event_mass


def apply_event_marginal_calibration(
    event_mass: torch.Tensor,
    calibration: dict[str, Any] | None,
) -> torch.Tensor:
    if calibration is None:
        return event_mass
    if calibration.get("type") != "event_marginal":
        return event_mass
    if bool(calibration.get("identity", False)):
        return event_mass

    event_prob = event_mass.sum(dim=2).clamp_min(1e-12)
    time_profile = event_mass / event_prob.unsqueeze(-1)

    temperature = torch.as_tensor(
        calibration["event_temperature"],
        device=event_mass.device,
        dtype=event_mass.dtype,
    ).view(1, -1).clamp(min=0.5, max=5.0)
    calibrated_event_prob = torch.softmax(torch.log(event_prob) / temperature, dim=1)

    return (calibrated_event_prob.unsqueeze(-1) * time_profile).clamp_min(1e-12)


def fit_event_marginal_calibration_from_arrays(
    arrays: dict[str, np.ndarray],
    max_steps: int,
    logger,
) -> dict[str, Any]:
    logger.info("开始拟合validation-only event-marginal calibration...")

    if "event_mass" in arrays:
        event_prob = np.asarray(arrays["event_mass"].sum(axis=2), dtype=np.float64)
    else:
        event_prob = np.asarray(arrays["event_prob"], dtype=np.float64)
    event = np.asarray(arrays["event"], dtype=np.int64)
    num_events = int(event_prob.shape[1])

    def binary_ece(pred: np.ndarray, label: np.ndarray, bins: int = 15) -> float:
        score = 0.0
        edges = np.linspace(0.0, 1.0, bins + 1)
        for idx in range(bins):
            left = edges[idx]
            right = edges[idx + 1]
            if idx + 1 == bins:
                mask = (pred >= left) & (pred <= right)
            else:
                mask = (pred >= left) & (pred < right)
            count = int(mask.sum())
            if count == 0:
                continue
            score += (count / max(len(pred), 1)) * abs(float(pred[mask].mean()) - float(label[mask].mean()))
        return float(score)

    def calibrated_event_prob(temperature: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(event_prob, 1e-12, None)) / temperature.reshape(1, -1)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-12, None)

    def macro_ece(temperature: np.ndarray) -> float:
        prob = calibrated_event_prob(temperature)
        scores = [
            binary_ece(prob[:, idx], (event == idx).astype(np.float64))
            for idx in range(num_events)
        ]
        return float(np.mean(scores))

    grid = np.asarray([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0], dtype=np.float64)
    max_rounds = max(1, min(int(max_steps), 3))
    temperature = np.ones(num_events, dtype=np.float64)
    before_ece = macro_ece(temperature)
    best_ece = before_ece
    for _ in range(max_rounds):
        improved = False
        for event_idx in range(num_events):
            best_local_temperature = float(temperature[event_idx])
            best_local_ece = best_ece
            for value in grid:
                candidate = temperature.copy()
                candidate[event_idx] = float(value)
                value_ece = macro_ece(candidate)
                if value_ece + 1e-12 < best_local_ece:
                    best_local_ece = value_ece
                    best_local_temperature = float(value)
            if best_local_ece + 1e-12 < best_ece:
                temperature[event_idx] = best_local_temperature
                best_ece = best_local_ece
                improved = True
        if not improved:
            break

    before_cr_nll = float(np.mean(arrays["per_sample_cumulative_nll"]))
    calibration = {
        "type": "event_marginal",
        "event_temperature": temperature.astype(np.float32).tolist(),
    }
    after_cr_nll: float | None = None
    if "event_mass" in arrays:
        time_bin = torch.as_tensor(arrays["time_bin"], dtype=torch.long)
        event_mass_torch = torch.as_tensor(arrays["event_mass"], dtype=torch.float32)
        event_torch = torch.as_tensor(event, dtype=torch.long)
        after_mass = apply_event_marginal_calibration(event_mass_torch, calibration)
        after_cr_nll = float(competing_risk_nll_from_event_mass(after_mass, event_torch, time_bin).detach())
    if (not np.isfinite(best_ece)) or best_ece > before_ece:
        logger.info("event-marginal calibration未改善validation macro ECE，回退到identity temperature。")
        temperature = np.ones(num_events, dtype=np.float32)
        calibration = {
            "type": "event_marginal",
            "event_temperature": temperature.astype(np.float32).tolist(),
            "identity": True,
        }
        best_ece = before_ece
        after_cr_nll = before_cr_nll
    else:
        calibration["identity"] = bool(np.allclose(temperature, 1.0))

    boundary_rate = float(((temperature <= 0.500001) | (temperature >= 4.999999)).mean())
    calibration["objective"] = "macro_ece"
    calibration["validation_macro_ece_before"] = before_ece
    calibration["validation_macro_ece_after"] = best_ece
    calibration["validation_cr_nll_before"] = before_cr_nll
    calibration["validation_cr_nll_after"] = after_cr_nll
    calibration["temperature_boundary_rate"] = boundary_rate
    logger.info(
        "validation-only event-marginal calibration完成。macro ECE %.6f -> %.6f, CR-NLL before %.6f, after %s, boundary_rate %.4f",
        before_ece,
        best_ece,
        before_cr_nll,
        "not_recomputed" if after_cr_nll is None else f"{after_cr_nll:.6f}",
        boundary_rate,
    )
    return calibration
