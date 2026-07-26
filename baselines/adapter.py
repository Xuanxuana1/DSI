from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from .utils import (
    discretize_time_label,
    duration_bucket,
    get_ord_criterion,
    get_tree_classify_loss,
    get_tree_encoded_label,
    get_tree_encoded_value,
    quantile_to_value,
    restore_time_label,
    value_to_quantile,
)

logger = logging.getLogger(__name__)


def _build_dynamic_vocab(values: np.ndarray, q_start: float, q_end: float, q_decay_rate: float, epsilon: float) -> list[int]:
    logger.info("开始构造GR动态词表。rows=%d", int(values.size))

    if values.size == 0:
        logger.info("GR动态词表构造完成。vocab_size=1")
        return [1]
    sorted_values = np.sort(values.astype(np.int64, copy=False))[::-1]
    updated = sorted_values.copy()
    output: list[int] = []
    error = float("inf")
    q_value = float(q_start)

    while error > float(epsilon):
        token = int(np.percentile(updated, q_value * 100.0))
        if token <= 0:
            break
        updated = np.where(updated < token, updated, updated - token)
        output.append(token)
        denom = np.maximum(sorted_values, 1e-10)
        error = float(np.max(np.abs(updated / denom)))
        q_value = max(float(q_end), q_value * float(q_decay_rate))
        if len(output) >= 512:
            break

    if not output:
        output = [1]
    logger.info("GR动态词表构造完成。vocab_size=%d", int(len(output)))
    return output


def _duration_ratio(labels: np.ndarray, durations: np.ndarray) -> np.ndarray:
    return (labels.astype(np.float64, copy=False) / np.maximum(durations.astype(np.float64, copy=False), 1e-6)).clip(0.0, 10.0)


def build_gr_numeric_vocab(
    labels: np.ndarray,
    durations: np.ndarray,
    q_start: float,
    q_end: float,
    q_decay_rate: float,
    epsilon: float,
    max_numeric_tokens: int | None = None,
) -> list[int]:
    logger.info("开始构造GR原仓库式动态数值词表。")

    ratio_int = (_duration_ratio(labels, durations) * 1000.0).astype(np.int64)

    numeric_vocab = _build_dynamic_vocab(ratio_int, q_start, q_end, q_decay_rate, epsilon)
    if max_numeric_tokens is not None and int(max_numeric_tokens) > 0:
        numeric_vocab = numeric_vocab[: int(max_numeric_tokens)]

    logger.info("GR原仓库式动态数值词表构造完成。numeric_tokens=%d", int(len(numeric_vocab)))
    return numeric_vocab


def _hsic_loss(representation: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:

    x = representation.flatten(start_dim=1)
    y = duration.flatten(start_dim=1)
    batch_size = int(x.shape[0])
    if batch_size <= 3:
        return x.new_zeros(())

    xx = x @ x.T
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    dxx = rx.T + rx - 2.0 * xx
    vx = torch.quantile(dxx, 0.5).clamp_min(1e-6)
    yy = y @ y.T
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    dyy = ry.T + ry - 2.0 * yy
    vy = torch.quantile(dyy, 0.5).clamp_min(1e-6)
    mask = 1.0 - torch.eye(batch_size, device=x.device)
    k_mat = torch.exp(-0.5 * dxx / vx) * mask
    l_mat = torch.exp(-0.5 * dyy / vy) * mask

    kl_mat = k_mat @ l_mat
    trace = kl_mat.trace()
    second = k_mat.sum() * l_mat.sum() / float((batch_size - 1) * (batch_size - 2))
    third = kl_mat.sum() / float(batch_size - 2)
    hsic = (trace + second - 2.0 * third) / float(batch_size * (batch_size - 3))
    return hsic.clamp_min(0.0)


def _micro_hsic_loss(
    representation: torch.Tensor,
    duration: torch.Tensor,
    pred: torch.Tensor,
    label: torch.Tensor,
    bucket: torch.Tensor,
    micro_bs: int,
    bucket_count: int,
) -> torch.Tensor:

    batch_size = int(representation.shape[0])
    if batch_size <= 0 or batch_size % int(micro_bs) != 0:
        return representation.new_zeros(())

    err = pred - label
    threshold = 0.1 + 0.3 * (bucket.float() + 1.0) / max(float(bucket_count), 1.0)
    active = err.abs() > threshold
    residual = ((err.abs() - threshold)[active] ** 2).sum() / 2.0 if active.any() else representation.new_zeros(())

    losses = []
    for rep_chunk, dur_chunk in zip(representation.split(int(micro_bs), dim=0), duration.split(int(micro_bs), dim=0)):
        losses.append(_hsic_loss(rep_chunk, dur_chunk))
    return torch.stack(losses).sum() * 100.0 - residual


class BaseAdapter:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def parameters(self):
        return self.model.parameters()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def to(self, device: torch.device | str):
        self.model.to(device)
        return self

    def stage1_loss(self, features: dict[str, torch.Tensor], label: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_base_pred(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_base_hidden(self, features: dict[str, torch.Tensor]) -> torch.Tensor | None:
        return None

    def get_base_pred_and_hidden(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:

        p, proxy = self.get_base_pred(features)

        hidden = self.get_base_hidden(features)
        return p, proxy, hidden

    @staticmethod
    def probability_from_proxy(proxy: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:

        return torch.sigmoid(torch.log(proxy.clamp(min=1e-7)) - torch.log1p(duration.view(-1, 1)))


class VRAdapter(BaseAdapter):
    def stage1_loss(self, features: dict[str, torch.Tensor], label: torch.Tensor) -> torch.Tensor:

        target = (label / features["duration"].clamp(min=1e-6)).clamp(0.0, 10.0)

        pred = self.model(features)
        return F.mse_loss(pred, target.float())

    def get_base_pred(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self.model(features).view(-1, 1).clamp(1e-6, 1.0 - 1e-6)
        proxy = pred * features["duration"].view(-1, 1).clamp(min=1e-6)
        return pred, proxy.clamp(min=0.001)

    def get_base_pred_and_hidden(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        pred, hidden = self.model.forward_with_hidden(features)
        p = pred.view(-1, 1).clamp(1e-6, 1.0 - 1e-6)
        proxy = p * features["duration"].view(-1, 1).clamp(min=1e-6)
        return p, proxy.clamp(min=0.001), hidden.detach()


class WLRAdapter(BaseAdapter):
    def __init__(self, model: torch.nn.Module) -> None:
        logger.info("开始初始化WLR adapter。")
        super().__init__(model)

        self.label_scale = 1.0
        logger.info("WLR adapter初始化完成。")

    def set_label_scale(self, label_scale: float) -> None:
        logger.info("开始绑定WLR标签缩放。label_scale=%.6f", float(label_scale))

        self.label_scale = max(float(label_scale), 1e-6)
        logger.info("WLR标签缩放绑定完成。")

    def stage1_loss(self, features: dict[str, torch.Tensor], label: torch.Tensor) -> torch.Tensor:

        scale = label.new_tensor(self.label_scale).clamp(min=1e-6)
        target = (label.float() / scale).clamp(min=0.0)

        pred = self.model(features).clamp(1e-6, 1.0 - 1e-6)
        return -(target * torch.log(pred) + torch.log(1.0 - pred)).mean()

    def _proxy_from_probability(self, pred: torch.Tensor, features: dict[str, torch.Tensor]) -> torch.Tensor:
        logger.debug("开始还原WLR proxy。")

        odds = pred / (1.0 - pred)

        scale = pred.new_tensor(self.label_scale).clamp(min=1e-6)
        proxy = odds.clamp(min=0.001) * scale
        logger.debug("WLR proxy还原完成。")
        return proxy

    def get_base_pred(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self.model(features).view(-1, 1).clamp(1e-6, 1.0 - 1e-6)
        proxy = self._proxy_from_probability(pred, features)
        return pred, proxy.clamp(min=0.001)

    def get_base_pred_and_hidden(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        pred, hidden = self.model.forward_with_hidden(features)
        p = pred.view(-1, 1).clamp(1e-6, 1.0 - 1e-6)
        proxy = self._proxy_from_probability(p, features)
        return p, proxy.clamp(min=0.001), hidden.detach()


class CWMAdapter(BaseAdapter):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        self.c_inv = 40.0
        self.sigma = 2.0

    def set_params(self, c_inv: float, sigma: float) -> None:
        logger.info("开始绑定CWM参数。c_inv=%.6f sigma=%.6f", float(c_inv), float(sigma))

        self.c_inv = float(c_inv)
        self.sigma = max(float(sigma), 1e-6)
        logger.info("CWM参数绑定完成。")

    @staticmethod
    def _inv_sigmoid(value: torch.Tensor) -> torch.Tensor:

        clipped = value.clamp(1e-6, 1.0 - 1e-6)
        return torch.log(clipped / (1.0 - clipped))

    def stage1_loss(self, features: dict[str, torch.Tensor], label: torch.Tensor) -> torch.Tensor:
        logger.debug("开始计算CWM loss。")

        duration = features["duration"].view(-1).float().clamp(min=0.0)
        wt = torch.minimum(label.view(-1).float().clamp(min=0.0), duration)
        rel_score = self.model(features).view(-1)
        c_inv = rel_score.new_tensor(self.c_inv)
        sigma = rel_score.new_tensor(self.sigma).clamp(min=1e-6)

        less_mask = wt < duration
        over_mask = wt >= duration
        loss = rel_score.new_zeros(())
        if less_mask.any():
            rel_true_less = torch.exp(-c_inv / (wt[less_mask] + 1.0))
            target_less = self._inv_sigmoid(rel_true_less)
            loss = loss + torch.square(rel_score[less_mask] - target_less).div(2.0 * sigma.pow(2)).mean()
        if over_mask.any():
            rel_true_over = torch.exp(-c_inv / (wt[over_mask] + 1.0))
            target_over = self._inv_sigmoid(rel_true_over)
            loss = loss - F.logsigmoid((rel_score[over_mask] - target_over) / sigma).mean()

        logger.debug("CWM loss计算完成。loss=%.6f", float(loss.detach().cpu().item()))
        return loss

    def get_base_pred(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logger.debug("开始读取CWM预测。")

        raw_score = self.model(features).view(-1, 1)
        pred_rel = -F.logsigmoid(raw_score)

        duration = features["duration"].view(-1, 1).float().clamp(min=0.0)
        c_inv = raw_score.new_tensor(self.c_inv)
        proxy = c_inv / pred_rel.clamp(min=1e-6) - 1.0
        proxy = proxy.clamp(min=0.0)
        proxy = torch.minimum(proxy, duration)

        p = torch.sigmoid(torch.log(proxy.clamp(min=1e-7)) - torch.log1p(duration.clamp(min=1e-7)))
        logger.debug("CWM预测读取完成。")
        return p.clamp(1e-6, 1.0 - 1e-6), proxy


class EGMNAdapter(BaseAdapter):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        self.label_scale = 1.0

    def set_label_scale(self, label_scale: float) -> None:
        logger.info("开始绑定EGMN标签缩放。label_scale=%.6f", float(label_scale))

        self.label_scale = max(float(label_scale), 1e-6)
        logger.info("EGMN标签缩放绑定完成。")

    def stage1_loss(self, features: dict[str, torch.Tensor], label: torch.Tensor) -> torch.Tensor:
        logger.debug("开始计算EGMN adapter loss。")

        scale = label.new_tensor(self.label_scale).clamp(min=1e-6)
        target = label.float() / scale

        pi, lambda_, mu, sigma = self.model(features)

        nll, reg, entropy = self.model.loss(target, pi, lambda_, mu, sigma)
        loss = nll - 0.1 * entropy + 1.0 * reg
        logger.debug("EGMN adapter loss计算完成。loss=%.6f", float(loss.detach().cpu().item()))
        return loss

    def get_base_pred(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logger.debug("开始读取EGMN预测。")

        normalized_proxy = self.model.predict(features).view(-1, 1).clamp(min=1e-6)
        scale = normalized_proxy.new_tensor(self.label_scale).clamp(min=1e-6)
        proxy = (normalized_proxy * scale).clamp(min=1e-6)

        p = torch.sigmoid(torch.log(proxy.clamp(min=1e-7)) - torch.log1p(features["duration"].view(-1, 1)))
        logger.debug("EGMN预测读取完成。")
        return p.clamp(1e-6, 1.0 - 1e-6), proxy.clamp(min=0.001)

    def get_base_pred_and_hidden(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        logger.debug("开始读取EGMN预测和hidden。")

        pi, lambda_, mu, _, hidden = self.model.forward_with_hidden(features)
        pi_soft = torch.softmax(pi, dim=1)
        means = torch.cat([1.0 / lambda_, mu], dim=1)
        normalized_proxy = (pi_soft * means).sum(dim=1, keepdim=True).clamp(min=1e-6)

        scale = normalized_proxy.new_tensor(self.label_scale).clamp(min=1e-6)
        proxy = (normalized_proxy * scale).clamp(min=1e-6)
        p = torch.sigmoid(torch.log(proxy.clamp(min=1e-7)) - torch.log1p(features["duration"].view(-1, 1)))
        logger.debug("EGMN预测和hidden读取完成。")
        return p.clamp(1e-6, 1.0 - 1e-6), proxy.clamp(min=0.001), hidden.detach()


class TPMAdapter(BaseAdapter):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        self.bucket_num = 32
        self.bucket_begins: torch.Tensor | None = None
        self.bucket_ends: torch.Tensor | None = None

    def set_buckets(self, begins: torch.Tensor, ends: torch.Tensor, bucket_num: int) -> None:

        self.bucket_begins = begins
        self.bucket_ends = ends
        self.bucket_num = int(bucket_num)

    def stage1_loss(self, features: dict[str, torch.Tensor], label: torch.Tensor) -> torch.Tensor:
        if self.bucket_begins is None or self.bucket_ends is None:
            raise RuntimeError("TPM buckets are not initialized")
        pred = self.model(features)
        encoded_y, variance = get_tree_encoded_value(pred, self.bucket_num, self.bucket_begins, self.bucket_ends)
        label_dict, weight_dict = get_tree_encoded_label(label, self.bucket_num, self.bucket_begins, self.bucket_ends)
        tree_loss = get_tree_classify_loss(label_dict, weight_dict, pred, self.bucket_num)
        mse_loss = F.mse_loss(encoded_y, label.view(-1, 1).float())
        return tree_loss + mse_loss + 0.0001 * variance

    def get_base_pred(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bucket_begins is None or self.bucket_ends is None:
            raise RuntimeError("TPM buckets are not initialized")
        pred = self.model(features)
        proxy, _ = get_tree_encoded_value(pred, self.bucket_num, self.bucket_begins, self.bucket_ends)
        p = self.probability_from_proxy(proxy, features["duration"])
        return p.clamp(1e-6, 1.0 - 1e-6), proxy.clamp(min=0.001)

    def get_base_pred_and_hidden(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.bucket_begins is None or self.bucket_ends is None:
            raise RuntimeError("TPM buckets are not initialized")
        pred, hidden = self.model.forward_with_hidden(features)
        proxy, _ = get_tree_encoded_value(pred, self.bucket_num, self.bucket_begins, self.bucket_ends)
        p = self.probability_from_proxy(proxy, features["duration"])
        return p.clamp(1e-6, 1.0 - 1e-6), proxy.clamp(min=0.001), hidden.detach()


class D2QAdapter(BaseAdapter):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        self.duration_edges: torch.Tensor | None = None
        self.quantiles: torch.Tensor | None = None

    def set_quantiles(self, duration_edges: torch.Tensor, quantiles: torch.Tensor) -> None:
        self.duration_edges = duration_edges
        self.quantiles = quantiles

    def stage1_loss(self, features: dict[str, torch.Tensor], label: torch.Tensor) -> torch.Tensor:
        if self.duration_edges is None or self.quantiles is None:
            raise RuntimeError("D2Q quantiles are not initialized")
        pred = self.model(features)
        bucket = duration_bucket(features["duration"], self.duration_edges)
        target = value_to_quantile(label, bucket, self.quantiles)
        return F.mse_loss(pred, target)

    def get_base_pred(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.duration_edges is None or self.quantiles is None:
            raise RuntimeError("D2Q quantiles are not initialized")
        pred = self.model(features)
        bucket = duration_bucket(features["duration"], self.duration_edges)
        proxy = quantile_to_value(pred, bucket, self.quantiles)
        p = self.probability_from_proxy(proxy, features["duration"])
        return p.clamp(1e-6, 1.0 - 1e-6), proxy.clamp(min=0.001)

    def get_base_pred_and_hidden(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.duration_edges is None or self.quantiles is None:
            raise RuntimeError("D2Q quantiles are not initialized")
        pred, hidden = self.model.forward_with_hidden(features)
        bucket = duration_bucket(features["duration"], self.duration_edges)
        proxy = quantile_to_value(pred, bucket, self.quantiles)
        p = self.probability_from_proxy(proxy, features["duration"])
        return p.clamp(1e-6, 1.0 - 1e-6), proxy.clamp(min=0.001), hidden.detach()


class CREADAdapter(BaseAdapter):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        self.split_nodes: torch.Tensor | None = None
        self.label_scale = 1.0
        self.lambda_ce = 1.0
        self.lambda_restore = 1.0
        self.lambda_ord = 0.00002

    def set_split_nodes(
        self,
        split_nodes: torch.Tensor,
        label_scale: float = 1.0,
        lambda_ce: float = 1.0,
        lambda_restore: float = 1.0,
        lambda_ord: float = 0.00002,
    ) -> None:
        logger.info(
            "开始绑定CREAD阈值。heads=%d label_space=normalized lambda_ce=%.6f lambda_restore=%.6f lambda_ord=%.6f",
            int(split_nodes.numel()),
            float(lambda_ce),
            float(lambda_restore),
            float(lambda_ord),
        )

        self.split_nodes = split_nodes
        self.label_scale = max(float(label_scale), 1e-6)
        self.lambda_ce = float(lambda_ce)
        self.lambda_restore = float(lambda_restore)
        self.lambda_ord = float(lambda_ord)
        logger.info("CREAD阈值绑定完成。")

    def stage1_loss(self, features: dict[str, torch.Tensor], label: torch.Tensor) -> torch.Tensor:
        logger.debug("开始计算CREAD loss。")
        if self.split_nodes is None:
            raise RuntimeError("CREAD split nodes are not initialized")

        scale = label.new_tensor(self.label_scale).clamp(min=1e-6)
        target = (label.float() / scale).clamp(min=0.0)

        pred = self.model(features)
        binary = discretize_time_label(target, self.split_nodes)
        restore = restore_time_label(pred, self.split_nodes)
        ce_loss = F.binary_cross_entropy(pred.clamp(1e-6, 1.0 - 1e-6), binary.float())
        restore_loss = F.huber_loss(restore, target)
        ord_loss = get_ord_criterion(pred)
        loss = self.lambda_ce * ce_loss + self.lambda_restore * restore_loss + self.lambda_ord * ord_loss
        logger.debug("CREAD loss计算完成。loss=%.6f", float(loss.detach().cpu().item()))
        return loss

    def get_base_pred(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logger.debug("开始读取CREAD预测。")
        if self.split_nodes is None:
            raise RuntimeError("CREAD split nodes are not initialized")

        pred = self.model(features)
        proxy = restore_time_label(pred, self.split_nodes).view(-1, 1).clamp(min=1e-6)

        scale = proxy.new_tensor(self.label_scale).clamp(min=1e-6)
        proxy = (proxy * scale).clamp(min=1e-6)

        p = self.probability_from_proxy(proxy, features["duration"])
        logger.debug("CREAD预测读取完成。")
        return p.clamp(1e-6, 1.0 - 1e-6), proxy.clamp(min=0.001)

    def get_base_pred_and_hidden(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.split_nodes is None:
            raise RuntimeError("CREAD split nodes are not initialized")
        pred, hidden = self.model.forward_with_hidden(features)
        proxy = restore_time_label(pred, self.split_nodes).view(-1, 1).clamp(min=1e-6)
        scale = proxy.new_tensor(self.label_scale).clamp(min=1e-6)
        proxy = (proxy * scale).clamp(min=1e-6)
        p = self.probability_from_proxy(proxy, features["duration"])
        return p.clamp(1e-6, 1.0 - 1e-6), proxy.clamp(min=0.001), hidden.detach()


class GRAdapter(BaseAdapter):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        self.duration_edges: torch.Tensor | None = None
        self.numeric_vocab: list[int] = [1]
        self.token_values: torch.Tensor | None = None
        self.max_len = 20
        self.window_size = 20
        self.cls_weight = 10.0
        self.huber_weight = 1.0
        self.teacher_force_ratio = 0.5

    def set_references(
        self,
        labels: np.ndarray,
        durations: np.ndarray,
        duration_edges: torch.Tensor,
        q_start: float,
        q_end: float,
        q_decay_rate: float,
        epsilon: float,
        max_len: int,
        window_size: int,
        cls_weight: float,
        huber_weight: float,
        teacher_force_ratio: float,
        numeric_vocab: list[int] | None = None,
    ) -> None:
        logger.info("开始绑定GR引用表。")

        output_dim = int(getattr(self.model.decoder, "output_dim"))
        if numeric_vocab is None:
            self.numeric_vocab = build_gr_numeric_vocab(
                labels,
                durations,
                q_start,
                q_end,
                q_decay_rate,
                epsilon,
                max_numeric_tokens=max(output_dim - 3, 1),
            )
        else:
            self.numeric_vocab = [int(value) for value in numeric_vocab[: max(output_dim - 3, 1)]]
        values = [0.0, 0.0, 0.0] + [float(value) for value in self.numeric_vocab]
        values.extend([0.0] * max(output_dim - len(values), 0))

        self.duration_edges = duration_edges
        self.token_values = torch.as_tensor(values, dtype=torch.float32, device=duration_edges.device)
        self.max_len = int(max_len)
        self.window_size = int(window_size)
        self.cls_weight = float(cls_weight)
        self.huber_weight = float(huber_weight)
        self.teacher_force_ratio = float(teacher_force_ratio)
        logger.info("GR引用表绑定完成。vocab_size=%d", int(len(values)))

    def _duration_bucket(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.duration_edges is None:
            raise RuntimeError("GR duration edges are not initialized")
        return duration_bucket(features["duration"], self.duration_edges)

    def _encode_tokens(self, label: torch.Tensor, duration: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logger.debug("开始编码GR目标token。")

        ratio = (label.detach().float() / duration.detach().float().clamp(min=1e-6)).clamp(0.0, 10.0)
        ratio_int = (ratio * 1000.0).long().cpu().numpy().reshape(-1)
        rows: list[list[int]] = []
        for value in ratio_int.tolist():
            remainder = int(value)
            row = [1]
            for token in self.numeric_vocab:
                while remainder >= int(token) and len(row) < self.max_len - 1:
                    remainder -= int(token)
                    row.append(3 + self.numeric_vocab.index(token))
            row.append(2)
            if len(row) > self.max_len:
                row = row[: self.max_len]
                row[-1] = 2
            row.extend([0] * max(self.max_len - len(row), 0))
            rows.append(row)

        device = label.device
        tokens = torch.as_tensor(rows, dtype=torch.long, device=device)
        mask = (tokens != 0).long()
        logger.debug("GR目标token编码完成。")
        return tokens, mask

    def _ratio_from_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.token_values is None:
            raise RuntimeError("GR token values are not initialized")

        probs = torch.softmax(logits, dim=-1)
        vocab_size = int(logits.shape[-1])
        argmax_idx = probs.argmax(dim=-1)
        indices = torch.arange(vocab_size, device=logits.device).view(1, 1, -1)
        active = (indices >= (argmax_idx.unsqueeze(-1) - self.window_size // 2)) & (
            indices < (argmax_idx.unsqueeze(-1) + self.window_size // 2 + 1)
        )
        values = self.token_values.to(device=logits.device, dtype=probs.dtype).view(1, 1, -1)
        weighted = torch.sum(probs * active.to(probs.dtype) * values, dim=-1)

        step_mask = mask[:, 1 : 1 + weighted.shape[1]].to(weighted.dtype)
        return (weighted * step_mask).sum(dim=1) / 1000.0

    def _ratio_from_autoregressive_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.token_values is None:
            raise RuntimeError("GR token values are not initialized")

        generated = logits.argmax(dim=-1)
        sequence = torch.cat([torch.ones((generated.shape[0], 1), dtype=torch.long, device=generated.device), generated], dim=1)
        eos_seen = sequence == 2
        first_eos = eos_seen.long().argmax(dim=1)
        first_eos = first_eos + (eos_seen.sum(dim=1) == 0).long() * sequence.shape[1]
        positions = torch.arange(sequence.shape[1], device=sequence.device).view(1, -1)
        mask = (positions <= first_eos.view(-1, 1)).long()
        return self._ratio_from_logits(logits, mask)

    def stage1_loss(self, features: dict[str, torch.Tensor], label: torch.Tensor) -> torch.Tensor:
        logger.debug("开始计算GR loss。")
        bucket = self._duration_bucket(features)
        tokens, mask = self._encode_tokens(label, features["duration"])
        logits = self.model(features, bucket, tokens[:, :-1], mask[:, :-1], self.teacher_force_ratio)
        target = tokens[:, 1:].contiguous().view(-1)
        loss_ce = F.cross_entropy(logits.contiguous().view(-1, logits.shape[-1]), target, ignore_index=0, reduction="sum")
        pred_ratio = self._ratio_from_logits(logits, mask)
        target_ratio = (label.float() / features["duration"].float().clamp(min=1e-6)).clamp(0.0, 10.0)
        loss_huber = F.huber_loss(pred_ratio, target_ratio, reduction="sum")
        loss = self.cls_weight * loss_ce + self.huber_weight * loss_huber
        logger.debug("GR loss计算完成。loss=%.6f", float(loss.detach().cpu().item()))
        return loss

    def get_base_pred(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logger.debug("开始读取GR预测。")
        bucket = self._duration_bucket(features)
        logits = self.model.autoregressive_logits(features, bucket, self.max_len)
        ratio = self._ratio_from_autoregressive_logits(logits).view(-1, 1).clamp(0.0, 10.0)
        proxy = ratio * features["duration"].view(-1, 1).clamp(min=1e-6)
        p = torch.sigmoid(torch.log(proxy.clamp(min=1e-7)) - torch.log1p(features["duration"].view(-1, 1)))
        logger.debug("GR预测读取完成。")
        return p.clamp(1e-6, 1.0 - 1e-6), proxy.clamp(min=0.001)


class DIFLAdapter(BaseAdapter):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        self.duration_edges: torch.Tensor | None = None
        self.quantiles: torch.Tensor | None = None
        self.micro_bs = 128

    def set_references(self, labels: np.ndarray, durations: np.ndarray, duration_edges: torch.Tensor, quantiles: torch.Tensor, micro_bs: int) -> None:
        logger.info("开始绑定DIFL引用表。")
        self.duration_edges = duration_edges
        self.quantiles = quantiles
        self.micro_bs = int(micro_bs)
        logger.info("DIFL引用表绑定完成。buckets=%d quantiles=%d", int(quantiles.shape[0]), int(quantiles.shape[1]))

    def _duration_bucket(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.duration_edges is None:
            raise RuntimeError("DIFL duration edges are not initialized")
        return duration_bucket(features["duration"], self.duration_edges)

    def _ratio_to_quantile(self, ratio: torch.Tensor, bucket: torch.Tensor) -> torch.Tensor:
        if self.quantiles is None:
            raise RuntimeError("DIFL quantiles are not initialized")

        table = self.quantiles.to(device=ratio.device, dtype=ratio.dtype)
        output = torch.zeros_like(ratio)
        denom = float(max(table.shape[1] - 1, 1))
        for bucket_id in range(table.shape[0]):
            mask = bucket == bucket_id
            if not mask.any():
                continue
            edges = table[bucket_id]
            values = ratio[mask].clamp(min=float(edges[0]), max=float(edges[-1]))
            right = torch.searchsorted(edges.contiguous(), values.contiguous(), right=True).clamp(1, table.shape[1] - 1)
            left = right - 1
            low = edges[left]
            high = edges[right]
            weight = torch.where((high - low).abs() > 1e-8, (values - low) / (high - low).clamp_min(1e-8), torch.zeros_like(values))
            output[mask] = (left.to(ratio.dtype) + weight) / denom
        return output.clamp(0.0, 1.0)

    def _quantile_to_ratio(self, pred: torch.Tensor, bucket: torch.Tensor) -> torch.Tensor:
        if self.quantiles is None:
            raise RuntimeError("DIFL quantiles are not initialized")

        table = self.quantiles.to(device=pred.device, dtype=pred.dtype)
        output = torch.zeros_like(pred)
        scale = float(max(table.shape[1] - 1, 1))
        for bucket_id in range(table.shape[0]):
            mask = bucket == bucket_id
            if not mask.any():
                continue
            pos = (pred[mask].clamp(0.0, 1.0) * scale).clamp(0.0, scale)
            left = torch.floor(pos).long().clamp(0, table.shape[1] - 1)
            right = (left + 1).clamp(max=table.shape[1] - 1)
            weight = pos - left.to(pred.dtype)
            output[mask] = table[bucket_id, left] + weight * (table[bucket_id, right] - table[bucket_id, left])
        return output.clamp(0.0, 10.0)

    def stage1_loss(self, features: dict[str, torch.Tensor], label: torch.Tensor) -> torch.Tensor:
        logger.debug("开始计算DIFL loss。")
        bucket = self._duration_bucket(features)
        ratio = (label.float() / features["duration"].float().clamp(min=1e-6)).clamp(0.0, 10.0)
        target = self._ratio_to_quantile(ratio, bucket)
        representation, pred = self.model(features, bucket)
        pred_flat = pred.view(-1)
        loss_huber = F.huber_loss(pred_flat, target, reduction="sum", delta=1.0)
        loss_hsic = _micro_hsic_loss(
            representation,
            features["duration"].view(-1, 1).float(),
            pred_flat,
            target,
            bucket,
            self.micro_bs,
            int(self.quantiles.shape[0]) if self.quantiles is not None else 1,
        )
        loss = loss_huber + loss_hsic
        logger.debug("DIFL loss计算完成。loss=%.6f", float(loss.detach().cpu().item()))
        return loss

    def get_base_pred(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logger.debug("开始读取DIFL预测。")
        bucket = self._duration_bucket(features)
        _, pred = self.model(features, bucket)
        ratio = self._quantile_to_ratio(pred.view(-1), bucket).view(-1, 1)
        proxy = ratio * features["duration"].view(-1, 1).clamp(min=1e-6)
        p = torch.sigmoid(torch.log(proxy.clamp(min=1e-7)) - torch.log1p(features["duration"].view(-1, 1)))
        logger.debug("DIFL预测读取完成。")
        return p.clamp(1e-6, 1.0 - 1e-6), proxy.clamp(min=0.001)

    def get_base_pred_and_hidden(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        bucket = self._duration_bucket(features)
        _, pred, hidden = self.model.forward_with_hidden(features, bucket)
        ratio = self._quantile_to_ratio(pred.view(-1), bucket).view(-1, 1)
        proxy = ratio * features["duration"].view(-1, 1).clamp(min=1e-6)
        p = self.probability_from_proxy(proxy, features["duration"])
        return p.clamp(1e-6, 1.0 - 1e-6), proxy.clamp(min=0.001), hidden.detach()

_REGISTRY = {
    "vr": VRAdapter,
    "wlr": WLRAdapter,
    "wd": WLRAdapter,
    "cwm": CWMAdapter,
    "egmn": EGMNAdapter,
    "tpm": TPMAdapter,
    "d2q": D2QAdapter,
    "cread": CREADAdapter,
    "gr": GRAdapter,
    "difl": DIFLAdapter,
}


def build_adapter(name: str, model: torch.nn.Module) -> BaseAdapter:

    if name not in _REGISTRY:
        raise ValueError(f"Unsupported baseline: {name}")

    logger.info("构造baseline adapter。name=%s", name)
    return _REGISTRY[name](model)


def list_supported_models() -> list[str]:
    return ["vr", "wlr", "cwm", "egmn", "tpm", "d2q", "cread", "gr", "difl"]
