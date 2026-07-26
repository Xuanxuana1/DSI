from __future__ import annotations

import torch


def _event_taxonomy_thresholds(output: dict[str, torch.Tensor]) -> tuple[int, float, float, float, str]:

    num_events = int(output.get("num_watch_events", 4))
    alpha = float(output.get("event_alpha", 0.3))
    mid_cut = float(output.get("event_mid_cut", 0.9))
    beta = float(output.get("event_beta", 1.1))
    support_mode = str(output.get("event_support_mode", "support_aware"))
    return num_events, alpha, mid_cut, beta, support_mode


def _observation_set_support_mask(
    logits: torch.Tensor,
    duration_sec: torch.Tensor,
    time_upper_bounds: torch.Tensor,
    num_events: int,
    alpha: float,
    mid_cut: float,
    beta: float,
    support_mode: str = "support_aware",
) -> torch.Tensor:

    dtype = logits.dtype
    device = logits.device
    upper = time_upper_bounds.to(device=device, dtype=dtype).view(1, 1, -1)
    duration = duration_sec.to(device=device, dtype=dtype).clamp_min(1e-3).view(-1, 1, 1)

    ratio = upper / duration
    support = torch.zeros_like(logits, dtype=torch.bool)
    if int(num_events) > 0:
        support[:, 0, :] = (ratio[:, 0, :] < float(alpha))
    if int(num_events) > 1:
        support[:, 1, :] = (ratio[:, 0, :] >= float(alpha)) & (ratio[:, 0, :] < float(mid_cut))
    if int(num_events) > 2:
        duration_bin = torch.bucketize(
            duration_sec.to(device=device, dtype=dtype).clamp_min(1e-3),
            time_upper_bounds.to(device=device, dtype=dtype),
            right=False,
        ).clamp(max=logits.shape[2] - 1)
        support[:, 2, :].scatter_(1, duration_bin.view(-1, 1), True)
    if int(num_events) > 3:
        if str(support_mode) == "cumulative":
            support[:, 3, :] = True
        elif str(support_mode) == "support_aware":
            support[:, 3, :] = ratio[:, 0, :] > float(beta)
        else:
            raise ValueError(f"Unsupported event_support_mode: {support_mode}")

    no_support = ~support.reshape(support.shape[0], -1).any(dim=1)
    if torch.any(no_support):
        support = support.clone()
        fallback_bin = torch.bucketize(
            duration_sec.to(device=device, dtype=dtype).clamp_min(1e-3),
            time_upper_bounds.to(device=device, dtype=dtype),
            right=False,
        ).clamp(max=logits.shape[2] - 1)
        fallback_event = min(max(int(num_events) - 1, 0), 2)
        support[no_support, fallback_event, fallback_bin[no_support]] = True

    return support


def _ratio_observation_set_support_mask(
    logits: torch.Tensor,
    ratio_upper_bounds: torch.Tensor,
    num_events: int,
    alpha: float,
    mid_cut: float,
    beta: float,
    support_mode: str = "support_aware",
) -> torch.Tensor:

    dtype = logits.dtype
    device = logits.device
    ratio = ratio_upper_bounds.to(device=device, dtype=dtype).view(1, -1)
    support = torch.zeros_like(logits, dtype=torch.bool)

    if int(num_events) > 0:
        support[:, 0, :] = ratio < float(alpha)
    if int(num_events) > 1:
        support[:, 1, :] = (ratio >= float(alpha)) & (ratio < float(mid_cut))
    if int(num_events) > 2:
        completion_bin = torch.argmin(torch.abs(ratio.view(-1) - ratio.new_tensor(1.0))).view(1, 1)
        support[:, 2, :].scatter_(1, completion_bin.expand(logits.shape[0], 1), True)
    if int(num_events) > 3:
        if str(support_mode) == "cumulative":
            support[:, 3, :] = True
        elif str(support_mode) == "support_aware":
            support[:, 3, :] = ratio > float(beta)
        else:
            raise ValueError(f"Unsupported event_support_mode: {support_mode}")

    no_support = ~support.reshape(support.shape[0], -1).any(dim=1)
    if torch.any(no_support):
        support = support.clone()
        fallback_event = min(max(int(num_events) - 1, 0), 2)
        fallback_bin = torch.argmin(torch.abs(ratio.view(-1) - ratio.new_tensor(1.0)))
        support[no_support, fallback_event, fallback_bin] = True

    return support


def observation_set_posterior_event_mass_from_output(
    output: dict[str, torch.Tensor],
    duration_sec: torch.Tensor,
    time_upper_bounds: torch.Tensor,
) -> torch.Tensor:

    if "obs_set_logits" not in output:
        raise KeyError("DSI provider output must contain obs_set_logits")
    num_events, alpha, mid_cut, beta, support_mode = _event_taxonomy_thresholds(output)

    logits = output["obs_set_logits"]
    support = _observation_set_support_mask(
        logits,
        duration_sec,
        time_upper_bounds,
        num_events,
        alpha,
        mid_cut,
        beta,
        support_mode,
    )
    masked_logits = logits.masked_fill(~support, -1e9)
    event_mass = torch.softmax(masked_logits.reshape(logits.shape[0], -1), dim=1).view_as(logits)
    return event_mass.clamp_min(1e-12)


def ratio_observation_set_posterior_event_mass_from_output(output: dict[str, torch.Tensor]) -> torch.Tensor:

    if "ratio_obs_set_logits" not in output:
        raise KeyError("DSI provider output must contain ratio_obs_set_logits")
    if "ratio_upper_bounds" not in output:
        raise KeyError("DSI provider output must contain ratio_upper_bounds")

    num_events, alpha, mid_cut, beta, support_mode = _event_taxonomy_thresholds(output)
    logits = output["ratio_obs_set_logits"]
    support = _ratio_observation_set_support_mask(
        logits,
        output["ratio_upper_bounds"],
        num_events,
        alpha,
        mid_cut,
        beta,
        support_mode,
    )

    masked_logits = logits.masked_fill(~support, -1e9)
    event_ratio_mass = torch.softmax(masked_logits.reshape(logits.shape[0], -1), dim=1).view_as(logits)
    return event_ratio_mass.clamp_min(1e-12)


def ratio_event_mass_to_time_mass(
    output: dict[str, torch.Tensor],
    duration_sec: torch.Tensor,
    time_upper_bounds: torch.Tensor,
) -> torch.Tensor:

    ratio_mass = ratio_observation_set_posterior_event_mass_from_output(output)
    dtype = ratio_mass.dtype
    device = ratio_mass.device
    ratio_upper = output["ratio_upper_bounds"].to(device=device, dtype=dtype)
    upper = time_upper_bounds.to(device=device, dtype=dtype)
    duration = duration_sec.to(device=device, dtype=dtype).clamp_min(1e-3).view(-1, 1)

    ratio_time = ratio_upper.view(1, -1) * duration
    time_bin = torch.bucketize(ratio_time, upper, right=False).clamp(max=int(upper.numel()) - 1)

    batch_size, num_events, _ = ratio_mass.shape
    time_mass = ratio_mass.new_zeros((batch_size, num_events, int(upper.numel())))
    scatter_index = time_bin.view(batch_size, 1, -1).expand(-1, num_events, -1)
    time_mass.scatter_add_(dim=2, index=scatter_index, src=ratio_mass)

    return time_mass.clamp_min(1e-12)


def posterior_event_mass_from_output(
    output: dict[str, torch.Tensor],
    duration_sec: torch.Tensor,
    time_upper_bounds: torch.Tensor,
) -> torch.Tensor:

    model_type = str(output.get("posterior_model_type", "obs_set_posterior"))
    primary_axis = str(output.get("posterior_primary_axis", "seconds"))

    if model_type == "time_only_posterior":
        return time_only_posterior_event_mass_from_output(output, duration_sec, time_upper_bounds)
    if primary_axis == "ratio":
        return ratio_event_mass_to_time_mass(output, duration_sec, time_upper_bounds)
    if primary_axis == "seconds":
        return observation_set_posterior_event_mass_from_output(output, duration_sec, time_upper_bounds)
    raise ValueError(f"Unsupported posterior_primary_axis: {primary_axis}")


def time_only_posterior_event_mass_from_output(
    output: dict[str, torch.Tensor],
    duration_sec: torch.Tensor,
    time_upper_bounds: torch.Tensor,
) -> torch.Tensor:

    if "obs_set_logits" not in output:
        raise KeyError("Time-only provider output must contain obs_set_logits")
    logits = output["obs_set_logits"]
    if logits.shape[1] != 1:
        raise ValueError(f"Time-only provider expects one event axis, got {logits.shape[1]}")
    time_mass = torch.softmax(logits[:, 0, :], dim=1)

    upper = time_upper_bounds.to(device=logits.device, dtype=logits.dtype).view(1, -1)
    duration = duration_sec.to(device=logits.device, dtype=logits.dtype).clamp_min(1e-3).view(-1, 1)
    ratio = upper / duration
    alpha = float(output.get("event_alpha", 0.3))
    mid_cut = float(output.get("event_mid_cut", 0.9))
    beta = float(output.get("event_beta", 1.1))
    masks = torch.stack(
        [
            ratio < alpha,
            (ratio >= alpha) & (ratio < mid_cut),
            (ratio >= mid_cut) & (ratio <= beta),
            ratio > beta,
        ],
        dim=1,
    )
    event_mass = time_mass.unsqueeze(1) * masks.to(time_mass.dtype)
    return event_mass.clamp_min(1e-12)
