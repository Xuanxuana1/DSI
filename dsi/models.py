from __future__ import annotations

from dataclasses import dataclass, replace
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)
_MODEL_LOG_ONCE_KEYS: set[str] = set()


def _model_log_once(key: str, message: str, *args: object) -> None:

    if key in _MODEL_LOG_ONCE_KEYS:
        return

    logger.info(message, *args)
    _MODEL_LOG_ONCE_KEYS.add(key)


def _embedding(num_embeddings: int, embedding_dim: int, sparse: bool) -> nn.Embedding:
    return nn.Embedding(max(int(num_embeddings), 1), int(embedding_dim), sparse=bool(sparse))


@dataclass(frozen=True)


class ModelShape:
    num_users: int
    num_items: int
    num_tabs: int
    num_watch_events: int
    num_time_bins: int
    num_numeric_features: int
    num_content_category_levels: int
    content_category_cardinalities: list[int]
    num_content_numeric_features: int
    raw_sparse_cardinalities: list[int]
    num_raw_numeric_features: int
    raw_sequence_cardinality: int
    raw_sequence_length: int
    unknown_prev_event: int


class RawFeatureEncoder(nn.Module):
    def __init__(self, shape: ModelShape, embedding_dim: int, sparse_embeddings: bool = False) -> None:
        super().__init__()
        logger.info(
            "开始初始化原始特征编码器。sparse=%d numeric=%d seq_len=%d",
            len(shape.raw_sparse_cardinalities),
            int(shape.num_raw_numeric_features),
            int(shape.raw_sequence_length),
        )

        self.embedding_dim = int(embedding_dim)
        self.raw_sparse_embeddings = nn.ModuleList(
            _embedding(cardinality, self.embedding_dim, sparse_embeddings)
            for cardinality in shape.raw_sparse_cardinalities
        )
        self.raw_seq_embedding = (
            _embedding(shape.raw_sequence_cardinality, self.embedding_dim, sparse_embeddings)
            if int(shape.raw_sequence_length) > 0 and int(shape.raw_sequence_cardinality) > 0
            else None
        )

        self.output_dim = (
            len(self.raw_sparse_embeddings) * self.embedding_dim
            + int(shape.num_raw_numeric_features)
            + (self.embedding_dim if self.raw_seq_embedding is not None else 0)
        )
        logger.info("原始特征编码器初始化完成。output_dim=%d", int(self.output_dim))

    def forward(self, batch: dict[str, torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
        _model_log_once("raw_feature_encoder_forward_start", "开始执行原始特征编码前向。")

        batch_size = int(reference.shape[0])
        if self.output_dim == 0:
            return reference.new_zeros((batch_size, 0))

        parts: list[torch.Tensor] = []
        raw_sparse = batch.get("raw_sparse")
        if raw_sparse is not None and raw_sparse.shape[1] > 0:
            for idx, embedding in enumerate(self.raw_sparse_embeddings):
                value = raw_sparse[:, idx].clamp_min(0).clamp_max(embedding.num_embeddings - 1)
                parts.append(embedding(value))
        raw_numeric = batch.get("raw_numeric")
        if raw_numeric is not None and raw_numeric.shape[1] > 0:
            parts.append(raw_numeric.float())

        raw_seq = batch.get("raw_seq")
        if self.raw_seq_embedding is not None and raw_seq is not None and raw_seq.shape[1] > 0:
            seq_value = raw_seq.clamp_min(0).clamp_max(self.raw_seq_embedding.num_embeddings - 1)
            seq_emb = self.raw_seq_embedding(seq_value)
            mask = batch.get("raw_seq_mask")
            if mask is None or mask.shape != raw_seq.shape:
                mask = (raw_seq > 0).to(seq_emb.dtype)
            else:
                mask = mask.to(seq_emb.dtype)
            pooled = (seq_emb * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            parts.append(pooled)

        output = torch.cat(parts, dim=-1) if parts else reference.new_zeros((batch_size, 0))
        _model_log_once("raw_feature_encoder_forward_done", "原始特征编码前向完成。")
        return output


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float, hidden_layers: int = 2) -> nn.Sequential:

    depth = max(int(hidden_layers), 1)

    layers: list[nn.Module] = []
    current_dim = int(input_dim)
    for _ in range(depth):
        layers.append(nn.Linear(current_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(float(dropout)))
        current_dim = int(hidden_dim)

    layers.append(nn.Linear(current_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _event_threshold_payload(alpha: float, mid_cut: float, beta: float, support_mode: str) -> dict[str, float | str]:
    logger.info(
        "开始构造事件边界载荷。alpha=%.4f mid_cut=%.4f beta=%.4f support_mode=%s",
        float(alpha),
        float(mid_cut),
        float(beta),
        str(support_mode),
    )

    payload = {
        "event_alpha": float(alpha),
        "event_mid_cut": float(mid_cut),
        "event_beta": float(beta),
        "event_support_mode": str(support_mode),
    }
    logger.info("事件边界载荷构造完成。")
    return payload


class ObservationSetPosteriorModel(nn.Module):
    def __init__(
        self,
        shape: ModelShape,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        sparse_embeddings: bool = False,
        alpha: float = 0.3,
        mid_cut: float = 0.9,
        beta: float = 1.1,
        support_mode: str = "support_aware",
        mlp_hidden_layers: int = 2,
        use_raw_features: bool = True,
        ratio_axis_bins: int = 0,
        duration_survival_bins: int = 0,
        duration_ratio_bucket_edges: list[float] | None = None,
        duration_prior_bucket_edges: list[float] | None = None,
        duration_prior_values: list[float] | None = None,
        posterior_primary_axis: str = "seconds",
    ) -> None:
        super().__init__()
        logger.info("开始初始化full joint observation-set provider。")

        self.shape = shape
        self.embedding_dim = int(embedding_dim)
        self.content_embedding_dim = max(4, int(embedding_dim) // 4)
        self.event_thresholds = _event_threshold_payload(alpha, mid_cut, beta, support_mode)
        self.use_raw_features = bool(use_raw_features)
        self.ratio_axis_bins = int(ratio_axis_bins)
        self.duration_survival_bins = int(duration_survival_bins)
        self.posterior_primary_axis = str(posterior_primary_axis)
        duration_ratio_edges = torch.as_tensor(duration_ratio_bucket_edges or [], dtype=torch.float32)
        duration_prior_edges = torch.as_tensor(duration_prior_bucket_edges or [], dtype=torch.float32)
        duration_prior_table = torch.as_tensor(duration_prior_values or [], dtype=torch.float32)
        self.duration_ratio_buckets = int(duration_ratio_edges.numel() + 1) if duration_ratio_edges.numel() > 0 else 0
        self.duration_prior_buckets = int(duration_prior_table.numel())

        self.user_embedding = _embedding(shape.num_users, embedding_dim, sparse_embeddings)
        self.item_embedding = _embedding(shape.num_items, embedding_dim, sparse_embeddings)
        self.tab_embedding = _embedding(shape.num_tabs, embedding_dim // 2, sparse_embeddings)
        self.prev_event_embedding = _embedding(shape.unknown_prev_event + 1, embedding_dim // 2, sparse_embeddings)
        self.content_embeddings = nn.ModuleList(
            _embedding(cardinality, self.content_embedding_dim, sparse_embeddings)
            for cardinality in shape.content_category_cardinalities
        )
        self.raw_feature_encoder = (
            RawFeatureEncoder(shape, self.content_embedding_dim, sparse_embeddings)
            if self.use_raw_features
            else None
        )

        content_dim = (
            shape.num_content_category_levels * self.content_embedding_dim
            + shape.num_content_numeric_features
            + (self.raw_feature_encoder.output_dim if self.raw_feature_encoder is not None else 0)
        )
        self.watch_dim = (
            int(embedding_dim) * 2
            + int(embedding_dim) // 2
            + int(embedding_dim) // 2
            + int(shape.num_numeric_features)
            + int(content_dim)
        )

        self.shared = _mlp(self.watch_dim, hidden_dim, hidden_dim, dropout, int(mlp_hidden_layers))
        self.event_time_head = nn.Linear(int(hidden_dim), int(shape.num_watch_events) * int(shape.num_time_bins))
        self.ratio_event_head = (
            nn.Linear(int(hidden_dim), int(shape.num_watch_events) * int(self.ratio_axis_bins))
            if int(self.ratio_axis_bins) > 0 and int(self.duration_ratio_buckets) == 0
            else None
        )
        self.duration_ratio_event_head = (
            nn.Linear(
                int(hidden_dim),
                int(self.duration_ratio_buckets) * int(shape.num_watch_events) * int(self.ratio_axis_bins),
            )
            if int(self.ratio_axis_bins) > 0 and int(self.duration_ratio_buckets) > 1
            else None
        )
        self.duration_survival_head = (
            nn.Linear(int(hidden_dim), int(self.duration_survival_bins))
            if int(self.duration_survival_bins) > 0
            else None
        )
        if int(self.ratio_axis_bins) > 0:
            self.register_buffer("ratio_upper_bounds", self._build_ratio_grid(int(self.ratio_axis_bins)), persistent=False)
        else:
            self.register_buffer("ratio_upper_bounds", torch.zeros(0, dtype=torch.float32), persistent=False)
        self.register_buffer("duration_ratio_bucket_edges", duration_ratio_edges, persistent=True)
        self.register_buffer("duration_prior_bucket_edges", duration_prior_edges, persistent=True)
        self.register_buffer("duration_prior_values", duration_prior_table, persistent=True)
        if int(self.duration_survival_bins) > 0:
            self.register_buffer("duration_survival_ratio_bounds", self._build_ratio_grid(int(self.duration_survival_bins)), persistent=False)
        else:
            self.register_buffer("duration_survival_ratio_bounds", torch.zeros(0, dtype=torch.float32), persistent=False)
        logger.info(
            "full joint observation-set provider初始化完成。watch_dim=%d use_raw=%s primary_axis=%s ratio_bins=%d duration_ratio_buckets=%d survival_bins=%d duration_prior_buckets=%d",
            int(self.watch_dim),
            self.use_raw_features,
            self.posterior_primary_axis,
            int(self.ratio_axis_bins),
            int(self.duration_ratio_buckets),
            int(self.duration_survival_bins),
            int(self.duration_prior_buckets),
        )

    @staticmethod
    def _build_ratio_grid(num_bins: int) -> torch.Tensor:
        logger.info("开始构造ratio网格。bins=%d", int(num_bins))

        if int(num_bins) <= 1:
            values = torch.tensor([10.0], dtype=torch.float32)
        else:
            dense_prefix = torch.linspace(1.0 / float(num_bins), 1.0, max(int(num_bins) // 2, 2))
            replay_tail = torch.expm1(torch.linspace(0.0, torch.log1p(torch.tensor(9.0)), max(int(num_bins) - dense_prefix.numel() + 1, 2))) + 1.0
            values = torch.unique(torch.cat([dense_prefix, replay_tail])).float().clamp(1e-3, 10.0)
            if values.numel() != int(num_bins):
                index = torch.linspace(0, values.numel() - 1, int(num_bins)).round().long()
                values = values[index]

        values = torch.sort(values.float().clamp(1e-3, 10.0)).values
        values[-1] = torch.tensor(10.0, dtype=torch.float32)
        logger.info("ratio网格构造完成。bins=%d", int(values.numel()))
        return values

    def _inputs(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _model_log_once("obs_set_inputs_start", "开始编码observation-set输入。")

        user_emb = self.user_embedding(batch["user_idx"])
        item_emb = self.item_embedding(batch["item_idx"])
        tab_emb = self.tab_embedding(batch["tab"])
        prev_event_emb = self.prev_event_embedding(batch["prev_event"])

        content_parts = [
            embedding(batch["content_category"][:, idx])
            for idx, embedding in enumerate(self.content_embeddings)
        ]
        if self.shape.num_content_numeric_features > 0:
            content_parts.append(batch["content_numeric"])
        if self.raw_feature_encoder is not None:
            raw_features = self.raw_feature_encoder(batch, item_emb)
            if raw_features.shape[1] > 0:
                content_parts.append(raw_features)
        content = torch.cat(content_parts, dim=-1) if content_parts else item_emb.new_zeros((item_emb.shape[0], 0))

        watch_input = torch.cat([user_emb, item_emb, tab_emb, prev_event_emb, batch["numeric"], content], dim=-1)
        _model_log_once("obs_set_inputs_done", "observation-set输入编码完成。")
        return {
            "watch_input": watch_input,
            "user_emb": user_emb,
            "posterior_context": watch_input,
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _model_log_once("full_joint_obs_set_forward_start", "开始执行full joint observation-set前向。")

        encoded = self._inputs(batch)
        hidden = self.shared(encoded["watch_input"])

        logits = self.event_time_head(hidden).view(
            -1,
            int(self.shape.num_watch_events),
            int(self.shape.num_time_bins),
        )

        output = {
            "obs_set_logits": logits,
            "num_watch_events": self.shape.num_watch_events,
            **self.event_thresholds,
            "user_emb": encoded["user_emb"],
            "posterior_hidden": hidden,
            "posterior_context": encoded["posterior_context"],
            "posterior_primary_axis": self.posterior_primary_axis,
        }

        if self.duration_ratio_event_head is not None:
            duration_bucket = torch.bucketize(
                batch["duration_sec"].to(device=hidden.device, dtype=self.duration_ratio_bucket_edges.dtype),
                self.duration_ratio_bucket_edges.to(device=hidden.device),
                right=False,
            ).clamp(max=int(self.duration_ratio_buckets) - 1)
            ratio_logits_all = self.duration_ratio_event_head(hidden).view(
                -1,
                int(self.duration_ratio_buckets),
                int(self.shape.num_watch_events),
                int(self.ratio_axis_bins),
            )
            gather_index = duration_bucket.view(-1, 1, 1, 1).expand(
                -1,
                1,
                int(self.shape.num_watch_events),
                int(self.ratio_axis_bins),
            )
            ratio_logits = torch.gather(ratio_logits_all, dim=1, index=gather_index).squeeze(1)
            output["ratio_obs_set_logits"] = ratio_logits
            output["ratio_upper_bounds"] = self.ratio_upper_bounds
            output["duration_ratio_bucket"] = duration_bucket
        elif self.ratio_event_head is not None:
            ratio_logits = self.ratio_event_head(hidden).view(
                -1,
                int(self.shape.num_watch_events),
                int(self.ratio_axis_bins),
            )
            output["ratio_obs_set_logits"] = ratio_logits
            output["ratio_upper_bounds"] = self.ratio_upper_bounds

        if self.duration_survival_head is not None:
            output["duration_survival_logits"] = self.duration_survival_head(hidden)
            output["duration_survival_ratio_bounds"] = self.duration_survival_ratio_bounds

        _model_log_once("full_joint_obs_set_forward_done", "full joint observation-set前向完成。")
        return output


class GenericLatentControlModel(ObservationSetPosteriorModel):
    def __init__(
        self,
        shape: ModelShape,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        sparse_embeddings: bool = False,
        latent_dim: int = 27,
        mlp_hidden_layers: int = 2,
        use_raw_features: bool = True,
    ) -> None:
        super().__init__(
            shape=shape,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            sparse_embeddings=sparse_embeddings,
            mlp_hidden_layers=mlp_hidden_layers,
            use_raw_features=use_raw_features,
            ratio_axis_bins=0,
            duration_survival_bins=0,
        )
        logger.info("开始初始化generic latent control。latent_dim=%d", int(latent_dim))

        generic_width = int(shape.num_watch_events) * int(shape.num_time_bins)
        self.generic_norm = nn.LayerNorm(generic_width)
        self.latent_head = nn.Linear(generic_width, int(latent_dim))

        self.event_head = nn.Linear(int(latent_dim), int(shape.num_watch_events))
        self.time_head = nn.Linear(int(latent_dim), 1)
        self.latent_dim = int(latent_dim)

        logger.info(
            "generic latent control初始化完成。generic_width=%d latent_dim=%d",
            generic_width,
            self.latent_dim,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _model_log_once("generic_latent_forward_start", "开始执行generic latent前向。")

        encoded = self._inputs(batch)
        hidden = self.shared(encoded["watch_input"])

        generic_hidden = torch.relu(self.generic_norm(self.event_time_head(hidden)))
        latent = self.latent_head(generic_hidden)

        output = {
            "latent": latent,
            "event_logits": self.event_head(latent),
            "pred_log_time": self.time_head(latent).squeeze(-1),
        }
        _model_log_once("generic_latent_forward_done", "generic latent前向完成。")
        return output


class TimeOnlyPosteriorModel(ObservationSetPosteriorModel):
    def __init__(
        self,
        shape: ModelShape,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        sparse_embeddings: bool = False,
        alpha: float = 0.3,
        mid_cut: float = 0.9,
        beta: float = 1.1,
        mlp_hidden_layers: int = 2,
        use_raw_features: bool = True,
    ) -> None:
        super().__init__(
            shape=shape,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            sparse_embeddings=sparse_embeddings,
            alpha=alpha,
            mid_cut=mid_cut,
            beta=beta,
            support_mode="support_aware",
            mlp_hidden_layers=mlp_hidden_layers,
            use_raw_features=use_raw_features,
            ratio_axis_bins=0,
            duration_survival_bins=0,
        )
        logger.info("开始初始化time-only posterior control。")
        self.time_only_alpha = float(alpha)
        self.time_only_mid_cut = float(mid_cut)
        self.time_only_beta = float(beta)
        logger.info("time-only posterior control初始化完成。")

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _model_log_once("time_only_posterior_forward_start", "开始执行time-only posterior前向。")
        output = super().forward(batch)
        logits = output["obs_set_logits"]
        time_logits = logits.mean(dim=1, keepdim=True)
        output["obs_set_logits"] = time_logits
        output["num_watch_events"] = 1
        output["posterior_model_type"] = "time_only_posterior"
        _model_log_once("time_only_posterior_forward_done", "time-only posterior前向完成。")
        return output


def build_shape(manifest: dict) -> ModelShape:

    content = manifest.get("content_features", {})
    raw_features = manifest.get("raw_features", {})

    return ModelShape(
        num_users=int(manifest["num_users"]),
        num_items=int(manifest["num_items"]),
        num_tabs=int(manifest["num_tabs"]),
        num_watch_events=len(manifest.get("watch_event_names", manifest["event_names"][:3])),
        num_time_bins=int(manifest["num_time_bins"]),
        num_numeric_features=int(manifest["num_numeric_features"]),
        num_content_category_levels=int(content.get("num_content_category_levels", 0)),
        content_category_cardinalities=[int(value) for value in content.get("content_category_cardinalities", [])],
        num_content_numeric_features=int(content.get("num_content_numeric_features", 0)),
        raw_sparse_cardinalities=[int(value) for value in raw_features.get("raw_sparse_cardinalities", [])],
        num_raw_numeric_features=int(raw_features.get("num_raw_numeric_features", 0)),
        raw_sequence_cardinality=int(raw_features.get("raw_sequence_cardinality", 0)),
        raw_sequence_length=int(raw_features.get("raw_sequence_length", 0)),
        unknown_prev_event=int(manifest["unknown_prev_event"]),
    )
