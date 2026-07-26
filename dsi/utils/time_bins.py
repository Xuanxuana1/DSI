from __future__ import annotations

import numpy as np


def build_time_bin_upper_bounds(
    linear_seconds: int,
    max_seconds: int,
    exponential_growth: float,
) -> np.ndarray:
    if linear_seconds < 1:
        raise ValueError("linear_seconds must be >= 1")
    if max_seconds <= linear_seconds:
        raise ValueError("max_seconds must be larger than linear_seconds")
    if exponential_growth <= 1.0:
        raise ValueError("exponential_growth must be > 1")

    bounds = list(range(1, linear_seconds + 1))
    current = float(linear_seconds)
    while current < max_seconds:
        current = min(float(max_seconds), np.ceil(current * exponential_growth))
        if int(current) <= bounds[-1]:
            current = bounds[-1] + 1
        bounds.append(int(current))
    return np.asarray(bounds, dtype=np.float32)


def assign_time_bins(event_time_seconds: np.ndarray, upper_bounds: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.ceil(event_time_seconds), 1, int(upper_bounds[-1]))
    return np.searchsorted(upper_bounds, clipped, side="left").astype(np.int64)
