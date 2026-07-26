from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def eval_mae(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean(np.abs(labels.reshape(-1) - scores.reshape(-1))))


def eval_rmse(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(labels.reshape(-1) - scores.reshape(-1)))))


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


def eval_xauc(labels: np.ndarray, scores: np.ndarray) -> float:
    logger.info("Computing XAUC. rows=%d", int(labels.reshape(-1).shape[0]))

    order = np.argsort(-scores.reshape(-1), kind="mergesort")
    sorted_labels = labels.reshape(-1)[order]

    work = sorted_labels.astype(np.float64, copy=True)
    pairs = len(work) * (len(work) - 1) / 2.0
    if pairs <= 0:
        logger.info("XAUC computed. xauc=0.000000")
        return 0.0
    xauc = float(_count_descending_label_pairs(work) / pairs)
    logger.info("XAUC computed. xauc=%.6f", xauc)
    return xauc


def eval_binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def get_playtime_percentiles_range(labels: np.ndarray, bucket_num: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    logger.info("Computing TPM quantile buckets. bucket_num=%d", int(bucket_num))

    percentiles = np.percentile(labels.reshape(-1), np.linspace(0.0, 100.0, int(bucket_num) + 1).astype(np.float32))

    begins = torch.tensor(percentiles[:-1], dtype=torch.float32, device=device).view(1, -1)
    ends = torch.tensor(percentiles[1:], dtype=torch.float32, device=device).view(1, -1)
    logger.info("TPM quantile buckets computed. min=%.4f max=%.4f", float(percentiles[0]), float(percentiles[-1]))
    return begins, ends


def get_tree_encoded_label(label: torch.Tensor, tree_num_intervals: int, begins: torch.Tensor, ends: torch.Tensor) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:

    label_dict: dict[int, torch.Tensor] = {}
    weight_dict: dict[int, torch.Tensor] = {}
    height = int(math.log2(int(tree_num_intervals)))
    target = label.view(-1, 1)

    for level in range(height):
        for node in range(2**level):
            temp_ind = max(int(tree_num_intervals * 1.0 / (2**level) * node) - 1, 0)
            if node == 0:
                weight = torch.where(target < begins[:, temp_ind].reshape(-1, 1), torch.zeros_like(target), torch.ones_like(target))
            else:
                weight = torch.where(target < ends[:, temp_ind].reshape(-1, 1), torch.zeros_like(target), torch.ones_like(target))
            temp_ind = max(int(tree_num_intervals * 1.0 / (2**level) * (node + 1)) - 1, 0)
            weight = torch.where(target < ends[:, temp_ind].reshape(-1, 1), weight, torch.zeros_like(target))
            temp_ind = max(int(tree_num_intervals * (1.0 / (2**level) * node + 1.0 / (2 ** (level + 1)))) - 1, 0)
            encoded = torch.where(target < ends[:, temp_ind].reshape(-1, 1), torch.zeros_like(target), torch.ones_like(target))
            label_dict[1000 * level + node] = encoded
            weight_dict[1000 * level + node] = weight
    return label_dict, weight_dict


def get_tree_classify_loss(label_dict: dict[int, torch.Tensor], weight_dict: dict[int, torch.Tensor], pred: torch.Tensor, tree_num_intervals: int) -> torch.Tensor:

    loss = pred.new_tensor(0.0)
    height = int(math.log2(int(tree_num_intervals)))
    for level in range(height):
        for node in range(2**level):
            labels = label_dict[1000 * level + node].reshape(-1, 1)
            weights = weight_dict[1000 * level + node].reshape(-1, 1)
            probs = pred[:, 2**level - 1 + node].view(-1, 1).clamp(1e-6, 1.0 - 1e-6)
            loss = loss + F.binary_cross_entropy(probs, labels, weight=weights)

    return loss / float(tree_num_intervals - 1)


def get_tree_encoded_value(pred: torch.Tensor, tree_num_intervals: int, begins: torch.Tensor, ends: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

    height = int(math.log2(int(tree_num_intervals)))
    leaf_values = (begins + ends) / 2.0
    batch_size = pred.size(0)
    encoded_prob_list = []

    for leaf_idx in range(int(tree_num_intervals)):
        log_prob = torch.zeros(batch_size, dtype=torch.float32, device=pred.device)
        cur_code = 2**height - 1 + leaf_idx
        for _ in range(1, height + 1):
            branch = cur_code % 2
            classifier_idx = (cur_code - 1) // 2
            probs = pred[:, classifier_idx].clamp(1e-6, 1.0 - 1e-6)
            log_prob = log_prob + (torch.log(1.0 - probs) if branch == 1 else torch.log(probs))
            cur_code = classifier_idx
        encoded_prob_list.append(log_prob)

    encoded_prob = torch.exp(torch.stack(encoded_prob_list, dim=1))
    encoded_playtime = torch.sum(leaf_values * encoded_prob, dim=-1, keepdim=True)
    e_x2 = torch.sum((leaf_values.pow(2)) * encoded_prob, dim=-1, keepdim=True)
    var = torch.sqrt(torch.abs(e_x2 - encoded_playtime.pow(2)) + 1e-8)
    return encoded_playtime.float(), torch.sum(var).float()


def duration_quantile_edges(durations: np.ndarray, bucket_num: int) -> np.ndarray:
    logger.info("Computing equal-frequency duration-bucket boundaries. bucket_num=%d", int(bucket_num))

    values = np.asarray(durations, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0 or int(bucket_num) <= 1:
        logger.info("Equal-frequency duration-bucket boundaries computed. actual_buckets=1")
        return np.zeros((0,), dtype=np.float32)

    probs = np.linspace(0.0, 1.0, int(bucket_num) + 1)
    raw_edges = np.quantile(values, probs)

    finite_edges = np.unique(raw_edges[1:-1][np.isfinite(raw_edges[1:-1])]).astype(np.float32)
    logger.info("Equal-frequency duration-bucket boundaries computed. requested=%d actual_buckets=%d", int(bucket_num), int(finite_edges.size + 1))
    return finite_edges


def build_quantile_table(labels: np.ndarray, durations: np.ndarray, bucket_num: int, quantile_bins: int) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Building the quantile auxiliary table. bucket_num=%d quantile_bins=%d", int(bucket_num), int(quantile_bins))

    finite_edges = duration_quantile_edges(durations, int(bucket_num))
    bucket_ids = np.searchsorted(finite_edges, durations.reshape(-1), side="left").astype(np.int64)
    actual_bucket_num = int(finite_edges.size + 1)

    quantiles = np.zeros((actual_bucket_num, int(quantile_bins)), dtype=np.float32)
    grid = np.linspace(0.0, 1.0, int(quantile_bins))
    for idx in range(actual_bucket_num):
        mask = bucket_ids == idx
        values = labels[mask]
        if len(values) == 0:
            values = labels
        quantiles[idx] = np.quantile(values.reshape(-1), grid).astype(np.float32)

    logger.info("Quantile auxiliary table built. buckets=%d", actual_bucket_num)
    return finite_edges, quantiles


def duration_bucket(duration: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    if edges.numel() == 0:
        return torch.zeros_like(duration, dtype=torch.long)
    return torch.bucketize(duration.view(-1), edges.to(duration.device), right=False).long()


def value_to_quantile(values: torch.Tensor, bucket_idx: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:

    value = values.view(-1).float()
    bucket = bucket_idx.view(-1).long()
    table = quantiles.to(values.device)
    selected = table.index_select(0, bucket)

    pos = torch.sum(selected < value.view(-1, 1), dim=1).clamp(min=1, max=selected.shape[1] - 1)
    row = torch.arange(value.shape[0], device=values.device)

    left = selected[row, pos - 1]
    right = selected[row, pos]
    delta = right - left
    fraction = torch.where(delta.abs() > 1e-6, (value - left) / delta.clamp(min=1e-6), torch.zeros_like(value))
    quantile_index = (pos - 1).float() + fraction
    quantile_index = torch.where(value <= selected[:, 0], torch.zeros_like(quantile_index), quantile_index)
    quantile_index = torch.where(value >= selected[:, -1], torch.full_like(quantile_index, float(selected.shape[1] - 1)), quantile_index)

    return (quantile_index / float(selected.shape[1] - 1)).clamp(0.0, 1.0)


def quantile_to_value(pred: torch.Tensor, bucket_idx: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:

    value = pred.view(-1).float().clamp(0.0, 1.0)
    bucket = bucket_idx.view(-1).long()
    table = quantiles.to(pred.device)
    selected = table.index_select(0, bucket)

    scaled = value * float(selected.shape[1] - 1)
    left_idx = torch.floor(scaled).long().clamp(min=0, max=selected.shape[1] - 2)
    right_idx = left_idx + 1
    weight = scaled - left_idx.float()
    row = torch.arange(value.shape[0], device=pred.device)

    left = selected[row, left_idx]
    right = selected[row, right_idx]
    return (left + weight * (right - left)).float().view(-1, 1)


def discretize_time_label(label: torch.Tensor, split_nodes: torch.Tensor) -> torch.Tensor:
    label_view = label.reshape(-1, 1, 1)
    nodes = split_nodes.reshape(1, 1, -1)
    return (label_view > nodes).float().squeeze(1)


def restore_time_label(preds: torch.Tensor, split_nodes: torch.Tensor) -> torch.Tensor:
    left_nodes = torch.cat([split_nodes.new_tensor([0.0]), split_nodes[:-1]])
    right_nodes = split_nodes
    bucket_size = right_nodes - left_nodes
    return torch.sum(preds * bucket_size.view(1, -1), dim=1)


def get_ord_criterion(preds: torch.Tensor) -> torch.Tensor:
    left = preds[:, :-1]
    right = preds[:, 1:]
    return torch.sum(torch.clamp(right - left, min=0.0))
