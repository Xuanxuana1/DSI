from __future__ import annotations

from typing import Any

from numba import njit
import numpy as np


def binary_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:

    label = np.asarray(labels, dtype=np.int8).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.isfinite(score)
    label = label[valid]
    score = score[valid]
    positive = int(label.sum())
    negative = int(label.shape[0] - positive)
    if positive <= 0 or negative <= 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_label = label[order]
    ranks = np.empty(sorted_score.shape[0], dtype=np.float64)
    start = 0
    while start < sorted_score.shape[0]:
        end = start + 1
        while end < sorted_score.shape[0] and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + 1 + end)
        start = end

    positive_rank_sum = float(ranks[sorted_label == 1].sum())
    return float((positive_rank_sum - positive * (positive + 1) * 0.5) / max(positive * negative, 1))


@njit(cache=True)


def _count_descending_label_pairs(values: np.ndarray) -> int:
    count = 0
    n = values.shape[0]
    if n <= 1:
        return 0
    src = values.copy()
    dst = np.empty_like(src)
    width = 1
    while width < n:
        start = 0
        while start < n:
            mid = min(start + width, n)
            end = min(start + 2 * width, n)
            i = start
            j = mid
            k = start
            while i < mid and j < end:
                if src[i] <= src[j]:
                    dst[k] = src[i]
                    i += 1
                else:
                    dst[k] = src[j]
                    count += mid - i
                    j += 1
                k += 1
            while i < mid:
                dst[k] = src[i]
                i += 1
                k += 1
            while j < end:
                dst[k] = src[j]
                j += 1
                k += 1
            start += 2 * width
        tmp = src
        src = dst
        dst = tmp
        width *= 2
    return count


@njit(cache=True)


def _dcg_prefix(relevance: np.ndarray, indices: np.ndarray, start: int, k: int) -> float:
    value = 0.0
    limit = min(k, indices.shape[0])
    for pos in range(limit):
        value += relevance[start + indices[pos]] / np.log2(float(pos) + 2.0)
    return value


@njit(cache=True)


def _session_ndcg_single(session: np.ndarray, score: np.ndarray, relevance: np.ndarray, k: int) -> float:
    total = 0.0
    count = 0
    n = session.shape[0]
    start = 0
    while start < n:
        end = start + 1
        while end < n and session[end] == session[start]:
            end += 1
        size = end - start
        if size >= 2:
            max_rel = relevance[start]
            for idx in range(start + 1, end):
                if relevance[idx] > max_rel:
                    max_rel = relevance[idx]
            if max_rel > 0.0:
                local_score = score[start:end]
                local_rel = relevance[start:end]
                pred_order = np.argsort(-local_score, kind="mergesort")
                ideal_order = np.argsort(-local_rel, kind="mergesort")
                ideal = _dcg_prefix(relevance, ideal_order, start, k)
                if ideal > 0.0:
                    total += _dcg_prefix(relevance, pred_order, start, k) / ideal
                    count += 1
        start = end
    if count == 0:
        return np.nan
    return total / float(count)


@njit(cache=True)


def _session_topk_value_single(session: np.ndarray, score: np.ndarray, reward: np.ndarray, k: int) -> tuple[float, int, int]:
    selected_sum = 0.0
    selected_count = 0
    usable_sessions = 0
    n = session.shape[0]
    start = 0
    while start < n:
        end = start + 1
        while end < n and session[end] == session[start]:
            end += 1
        size = end - start
        if size > 0:
            kk = min(k, size)
            local_score = score[start:end]
            order = np.argsort(-local_score, kind="mergesort")
            for pos in range(kk):
                selected_sum += reward[start + order[pos]]
            selected_count += kk
            usable_sessions += 1
        start = end
    if selected_count == 0:
        return np.nan, usable_sessions, selected_count
    return selected_sum / float(selected_count), usable_sessions, selected_count


def continuous_auc_score(targets: np.ndarray, scores: np.ndarray) -> float:

    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.isfinite(target) & np.isfinite(score)
    target = target[valid]
    score = score[valid]
    if target.size < 2 or np.unique(target).size < 2:
        return float("nan")

    order = np.argsort(-score, kind="mergesort")
    sorted_target = target[order].astype(np.float64, copy=True)

    pairs = sorted_target.shape[0] * (sorted_target.shape[0] - 1) / 2.0
    if pairs <= 0.0:
        return float("nan")
    return float(_count_descending_label_pairs(sorted_target) / pairs)


def grouped_continuous_auc_score(groups: np.ndarray, targets: np.ndarray, scores: np.ndarray, min_group_size: int = 2) -> float:

    group = np.asarray(groups, dtype=np.int64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.isfinite(target) & np.isfinite(score)
    group = group[valid]
    target = target[valid]
    score = score[valid]
    if group.size == 0:
        return float("nan")
    order = np.argsort(group, kind="mergesort")
    group = group[order]
    target = target[order]
    score = score[order]
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    ends = np.r_[starts[1:], group.size]

    weighted_sum = 0.0
    weight_sum = 0.0
    for start, end in zip(starts, ends):
        if end - start < int(min_group_size):
            continue
        if np.unique(target[start:end]).size < 2:
            continue
        value = continuous_auc_score(target[start:end], score[start:end])
        if not np.isfinite(value):
            continue
        weight = float(end - start)
        weighted_sum += float(value) * weight
        weight_sum += weight
    return float(weighted_sum / weight_sum) if weight_sum > 0.0 else float("nan")


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:

    pred = np.asarray(prediction, dtype=np.float64).reshape(-1).clip(min=0.0)
    truth = np.asarray(target, dtype=np.float64).reshape(-1).clip(min=0.0)
    valid = np.isfinite(pred) & np.isfinite(truth)
    pred = pred[valid]
    truth = truth[valid]
    if pred.size == 0:
        return {
            "mae_sec": float("nan"),
            "rmse_sec": float("nan"),
            "log_mae": float("nan"),
            "log_rmse": float("nan"),
        }

    error = pred - truth
    log_error = np.log1p(pred) - np.log1p(truth)
    return {
        "mae_sec": float(np.mean(np.abs(error))),
        "rmse_sec": float(np.sqrt(np.mean(np.square(error)))),
        "log_mae": float(np.mean(np.abs(log_error))),
        "log_rmse": float(np.sqrt(np.mean(np.square(log_error)))),
    }


def event_auc_metrics(event_prob: np.ndarray, watch_event: np.ndarray, event_names: list[str]) -> dict[str, float]:

    prob = np.asarray(event_prob, dtype=np.float64)
    label = np.asarray(watch_event, dtype=np.int64).reshape(-1)
    count = min(prob.shape[1], len(event_names))
    output: dict[str, float] = {}

    for event_idx in range(count):
        name = event_names[event_idx]
        output[f"{name}_auc"] = binary_auc_score((label == event_idx).astype(np.int8), prob[:, event_idx])

    tail_ids = [
        idx
        for idx, name in enumerate(event_names[:count])
        if name in {"true_completion", "normal_completion", "overplay_replay"}
    ]
    if tail_ids:
        long_label = np.isin(label, np.asarray(tail_ids, dtype=np.int64)).astype(np.int8)
        long_score = prob[:, tail_ids].sum(axis=1)
        output["long_engagement_auc"] = binary_auc_score(long_label, long_score)
    return output


def _dcg_at_k(relevance: np.ndarray, k: int) -> float:

    rel = np.asarray(relevance, dtype=np.float64)[: int(k)]
    if rel.size == 0:
        return 0.0
    discount = 1.0 / np.log2(np.arange(rel.size, dtype=np.float64) + 2.0)
    return float(np.sum(rel * discount))


def session_ndcg_metrics(
    session_key: np.ndarray,
    score: np.ndarray,
    relevance: np.ndarray,
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:

    session = np.asarray(session_key, dtype=np.int64).reshape(-1)
    pred = np.asarray(score, dtype=np.float64).reshape(-1)
    rel = np.asarray(relevance, dtype=np.float64).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(rel)
    session = session[valid]
    pred = pred[valid]
    rel = rel[valid]
    if session.size == 0:
        return {f"ndcg@{k}": float("nan") for k in ks}
    order = np.argsort(session, kind="mergesort")
    session = session[order]
    pred = pred[order]
    rel = rel[order]
    starts = np.flatnonzero(np.r_[True, session[1:] != session[:-1]])
    ends = np.r_[starts[1:], session.size]

    return {
        f"ndcg@{k}": float(_session_ndcg_single(session, pred, rel, int(k)))
        for k in ks
    }


def session_topk_value(
    session_key: np.ndarray,
    score: np.ndarray,
    reward: np.ndarray,
    k: int,
) -> dict[str, float | int]:

    session = np.asarray(session_key, dtype=np.int64).reshape(-1)
    pred = np.asarray(score, dtype=np.float64).reshape(-1)
    rel = np.asarray(reward, dtype=np.float64).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(rel)
    session = session[valid]
    pred = pred[valid]
    rel = rel[valid]
    if session.size == 0:
        return {"value": float("nan"), "sessions": 0, "selected": 0}
    order = np.argsort(session, kind="mergesort")
    session = session[order]
    pred = pred[order]
    rel = rel[order]

    value, sessions, selected = _session_topk_value_single(session, pred, rel, int(k))
    return {"value": float(value), "sessions": int(sessions), "selected": int(selected)}


def neutral_metric_summary(
    expected_time_sec: np.ndarray,
    play_time_sec: np.ndarray,
    event_prob: np.ndarray,
    watch_event: np.ndarray,
    session_key: np.ndarray,
    event_names: list[str],
    ranking_score: np.ndarray | None = None,
) -> dict[str, Any]:

    output: dict[str, Any] = {
        "watch_time_error": regression_metrics(expected_time_sec, play_time_sec),
    }

    output["event_auc"] = event_auc_metrics(event_prob, watch_event, event_names)

    prob = np.asarray(event_prob, dtype=np.float64)
    label = np.asarray(watch_event, dtype=np.int64)
    tail_ids = [
        idx
        for idx, name in enumerate(event_names[: prob.shape[1]])
        if name in {"true_completion", "normal_completion", "overplay_replay"}
    ]
    long_label = np.isin(label, np.asarray(tail_ids, dtype=np.int64)).astype(np.float64)
    long_score = prob[:, tail_ids].sum(axis=1) if tail_ids else np.zeros_like(long_label)
    output["ranking"] = {
        "watch_time": session_ndcg_metrics(
            session_key,
            expected_time_sec,
            np.log1p(np.asarray(play_time_sec, dtype=np.float64).clip(min=0.0)),
        ),
        "long_engagement": session_ndcg_metrics(
            session_key,
            long_score,
            long_label,
        ),
    }

    output["watch_time_ranking"] = {
        "xauc_global": continuous_auc_score(play_time_sec, expected_time_sec),
        "session_xauc": grouped_continuous_auc_score(session_key, play_time_sec, expected_time_sec),
    }
    if ranking_score is not None:

        output["ranking_head"] = {
            "long_engagement": session_ndcg_metrics(
                session_key,
                np.asarray(ranking_score, dtype=np.float64),
                long_label,
            )
        }
    return output

from typing import Any

import numpy as np


def _ranking_metrics(score: np.ndarray, eval_arrays: dict[str, np.ndarray], topks: list[int]) -> dict[str, Any]:

    session = np.asarray(eval_arrays["session_key"], dtype=np.int64)
    rewards = {
        "watch_time_log": np.log1p(np.asarray(eval_arrays["play_time"], dtype=np.float64).clip(min=0.0)),
        "watch_time_sec": np.asarray(eval_arrays["play_time"], dtype=np.float64).clip(min=0.0),
        "long_label": np.asarray(eval_arrays["long_label"], dtype=np.float64),
        "overplay_label": np.asarray(eval_arrays["overplay_label"], dtype=np.float64),
    }
    ks = tuple(int(value) for value in topks)

    return {
        "ndcg": {
            name: session_ndcg_metrics(session, score, reward, ks=ks)
            for name, reward in rewards.items()
        },
        "topk_value": {
            name: {
                f"top{int(topk)}": session_topk_value(session, score, reward, int(topk))
                for topk in topks
            }
            for name, reward in rewards.items()
        },
    }


def _evaluate_decision(
    pred_time: np.ndarray,
    rank_score: np.ndarray,
    eval_arrays: dict[str, np.ndarray],
    topks: list[int],
    name: str,
    fast_eval: bool,
    fast_eval_rows: int,
    seed: int,
    logger,
) -> dict[str, Any]:
    logger.info("开始评估decision输出。name=%s", name)

    true_time = np.asarray(eval_arrays["play_time"], dtype=np.float64)
    session = np.asarray(eval_arrays["session_key"], dtype=np.int64)
    long_label = np.asarray(eval_arrays["long_label"], dtype=np.float64)
    overplay_label = np.asarray(eval_arrays["overplay_label"], dtype=np.float64)
    pred = np.asarray(pred_time, dtype=np.float64)
    score = np.asarray(rank_score, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(score)
    metric_valid = valid

    auc_sampled = False
    if bool(fast_eval) and int(fast_eval_rows) > 0 and int(valid.sum()) > int(fast_eval_rows):
        rng = np.random.default_rng(int(seed) + 9409)
        valid_index = np.flatnonzero(valid)
        sample_index = rng.choice(valid_index, size=int(fast_eval_rows), replace=False)
        metric_valid = np.zeros_like(valid, dtype=bool)
        metric_valid[sample_index] = True
        auc_sampled = True
        logger.info("fast-eval抽样排序指标。name=%s rows=%s", name, int(fast_eval_rows))

    output = {
        "finite_rows": int(valid.sum()),
        "total_rows": int(valid.shape[0]),
        "metric_rows": {
            "error_rows": int(valid.sum()),
            "auc_rows": int(metric_valid.sum()),
            "auc_sampled": bool(auc_sampled),
        },
        "watch_time_error": regression_metrics(pred[valid], true_time[valid]),
        "watch_time_ranking": {
            "xauc_global": continuous_auc_score(true_time[metric_valid], pred[metric_valid]),
            "session_xauc": None if bool(fast_eval) else grouped_continuous_auc_score(session[valid], true_time[valid], pred[valid]),
        },
        "rank_auc": {
            "long_auc": binary_auc_score(long_label[metric_valid], score[metric_valid]),
            "overplay_auc": binary_auc_score(overplay_label[metric_valid], score[metric_valid]),
        },
        "rank_head": None if bool(fast_eval) else _ranking_metrics(score, eval_arrays, topks),
        "pred_time_rank": None if bool(fast_eval) else _ranking_metrics(pred, eval_arrays, topks),
    }

    logger.info("decision输出评估完成。name=%s finite=%s", name, int(valid.sum()))
    return output
