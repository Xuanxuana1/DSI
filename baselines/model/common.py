from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


def feature_sizes(description: list[tuple[str, Any, str]]) -> dict[str, tuple[Any, str]]:
    return {name: (size, kind) for name, size, kind in description}


class DictFeatureEncoder(nn.Module):
    def __init__(self, description: list[tuple[str, Any, str]], embed_dim: int, include_ctn_linear: bool) -> None:
        super().__init__()
        logger.info("Initializing the baseline feature encoder. embed_dim=%d include_ctn_linear=%s", int(embed_dim), include_ctn_linear)

        self.spec = feature_sizes(description)
        self.sparse_names = [name for name, (_, kind) in self.spec.items() if kind == "spr"]
        self.sparse_vec_names = [name for name, (_, kind) in self.spec.items() if kind == "spr_vec"]
        self.sequence_names = [name for name, (_, kind) in self.spec.items() if kind == "seq"]
        self.ctn_vec_names = [name for name, (_, kind) in self.spec.items() if kind == "ctn_vec"]
        self.ctn_names = [name for name, (_, kind) in self.spec.items() if kind == "ctn"]
        self.include_ctn_linear = bool(include_ctn_linear)

        self.emb_layer = nn.ModuleDict({
            name: nn.Embedding(int(self.spec[name][0]), int(embed_dim))
            for name in self.sparse_names
        })
        self.sparse_vec_layers = nn.ModuleDict({
            name: nn.ModuleList(
                nn.Embedding(max(int(cardinality), 1), int(embed_dim))
                for cardinality in list(self.spec[name][0])
            )
            for name in self.sparse_vec_names
        })
        self.sequence_layers = nn.ModuleDict({
            name: nn.Embedding(max(int(self.spec[name][0]), 1), int(embed_dim))
            for name in self.sequence_names
            if int(self.spec[name][0]) > 0
        })

        self.embedding_output_dim = len(self.sparse_names) * int(embed_dim)
        self.sparse_vec_output_dim = sum(len(list(self.spec[name][0])) * int(embed_dim) for name in self.sparse_vec_names)
        self.sequence_output_dim = len(self.sequence_layers) * int(embed_dim)
        self.ctn_vec_dim = sum(int(self.spec[name][0]) for name in self.ctn_vec_names)
        self.ctn_dim = len(self.ctn_names)
        self.output_dim = self.embedding_output_dim + self.sparse_vec_output_dim + self.sequence_output_dim + self.ctn_vec_dim + self.ctn_dim
        self.linear_dim = self.ctn_vec_dim + self.ctn_dim if self.include_ctn_linear else 0
        logger.info(
            "Baseline feature encoder initialized. sparse=%d sparse_vec_dim=%d seq_dim=%d ctn_vec_dim=%d ctn_dim=%d output_dim=%d",
            len(self.sparse_names),
            self.sparse_vec_output_dim,
            self.sequence_output_dim,
            self.ctn_vec_dim,
            self.ctn_dim,
            self.output_dim,
        )

    def forward(self, x_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        logger.debug("Encoding baseline input features.")

        parts = [
            self.emb_layer[name](x_dict[name].view(-1)).view(x_dict[name].shape[0], -1)
            for name in self.sparse_names
        ]

        for name in self.sparse_vec_names:
            values = x_dict[name].long()
            embeddings = self.sparse_vec_layers[name]
            if values.shape[1] == 0 or len(embeddings) == 0:
                continue
            for idx, embedding in enumerate(embeddings):
                value = values[:, idx].clamp_min(0).clamp_max(embedding.num_embeddings - 1)
                parts.append(embedding(value))

        for name in self.sequence_names:
            if name not in self.sequence_layers:
                continue
            values = x_dict[name].long()
            if values.shape[1] == 0:
                continue
            embedding = self.sequence_layers[name]
            sequence = values.clamp_min(0).clamp_max(embedding.num_embeddings - 1)
            sequence_emb = embedding(sequence)
            mask_name = f"{name}_mask"
            mask = x_dict.get(mask_name)
            if mask is None or mask.shape != values.shape:
                mask = (values > 0).to(sequence_emb.dtype)
            else:
                mask = mask.to(sequence_emb.dtype)
            pooled = (sequence_emb * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            parts.append(pooled)

        ctn_parts: list[torch.Tensor] = []
        for name in self.ctn_vec_names:
            ctn_parts.append(x_dict[name].view(x_dict[name].shape[0], -1).float())
        for name in self.ctn_names:
            ctn_parts.append(x_dict[name].view(-1, 1).float())
        if ctn_parts:
            ctn = torch.cat(ctn_parts, dim=1)
            parts.append(ctn)
        else:
            ctn = None

        dense = torch.cat(parts, dim=1) if parts else x_dict["duration"].new_zeros((x_dict["duration"].shape[0], 0))
        linear = ctn if self.include_ctn_linear else None
        logger.debug("Baseline input features encoded. dense_dim=%d", int(dense.shape[1]))
        return dense, linear


class FieldFeatureEncoder(nn.Module):
    def __init__(self, description: list[tuple[str, Any, str]], embed_dim: int, continuous_mode: str = "dense") -> None:
        super().__init__()
        logger.info("Initializing the field encoder. embed_dim=%d mode=%s", int(embed_dim), str(continuous_mode))

        if str(continuous_mode) not in {"dense", "wide"}:
            raise ValueError(f"unsupported continuous_mode={continuous_mode}")
        self.spec = feature_sizes(description)
        self.continuous_mode = str(continuous_mode)
        self.features = {
            name: (size, kind)
            for name, (size, kind) in self.spec.items()
            if kind in {"spr", "seq", "ctn", "spr_vec", "ctn_vec"}
        }

        self.emb_layer = nn.ModuleDict()
        self.sparse_vec_layers = nn.ModuleDict()
        self.ctn_linear_layer = nn.ModuleDict()
        self.ctn_vec_linear_layer = nn.ModuleDict()
        dense_dim = 0
        linear_dim = 0
        for name, (size, kind) in self.features.items():
            if kind == "spr":
                self.emb_layer[name] = nn.Embedding(max(int(size), 1), int(embed_dim))
                dense_dim += int(embed_dim)
            elif kind == "seq":
                if int(size) > 0:
                    self.emb_layer[name] = nn.Embedding(max(int(size), 1), int(embed_dim))
                    dense_dim += int(embed_dim)
            elif kind == "spr_vec":
                embeddings = nn.ModuleList(
                    nn.Embedding(max(int(cardinality), 1), int(embed_dim))
                    for cardinality in list(size)
                )
                self.sparse_vec_layers[name] = embeddings
                dense_dim += len(embeddings) * int(embed_dim)
            elif kind == "ctn":
                self.ctn_linear_layer[name] = nn.Linear(1, 1, bias=False)
                if self.continuous_mode == "dense":
                    dense_dim += 1
                else:
                    linear_dim += 1
            elif kind == "ctn_vec":
                layers = nn.ModuleList(nn.Linear(1, 1, bias=False) for _ in range(int(size)))
                self.ctn_vec_linear_layer[name] = layers
                if self.continuous_mode == "dense":
                    dense_dim += len(layers)
                else:
                    linear_dim += len(layers)

        self.output_dim = int(dense_dim)
        self.linear_dim = int(linear_dim)
        logger.info("Field encoder initialized. dense_dim=%d linear_dim=%d", self.output_dim, self.linear_dim)

    def forward(self, x_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        logger.debug("Encoding baseline field inputs.")

        dense_parts: list[torch.Tensor] = []
        linear_parts: list[torch.Tensor] = []
        for name, (_, kind) in self.features.items():
            if kind == "spr":
                value = x_dict[name].view(-1).long().clamp_min(0).clamp_max(self.emb_layer[name].num_embeddings - 1)
                dense_parts.append(self.emb_layer[name](value))
            elif kind == "seq":
                if name not in self.emb_layer:
                    continue
                values = x_dict[name].long()
                embedding = self.emb_layer[name]
                sequence = values.clamp_min(0).clamp_max(embedding.num_embeddings - 1)
                seq_emb = embedding(sequence)
                mask = self._sequence_mask(x_dict, name, values).to(seq_emb.dtype)
                pooled = (seq_emb * mask.unsqueeze(2)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1e-9)
                dense_parts.append(pooled)
            elif kind == "spr_vec":
                values = x_dict[name].long()
                for idx, embedding in enumerate(self.sparse_vec_layers[name]):
                    value = values[:, idx].clamp_min(0).clamp_max(embedding.num_embeddings - 1)
                    dense_parts.append(embedding(value))

        for name, (_, kind) in self.features.items():
            if kind == "ctn":
                part = self.ctn_linear_layer[name](x_dict[name].float().view(-1, 1))
                if self.continuous_mode == "dense":
                    dense_parts.append(part)
                else:
                    linear_parts.append(part)
            elif kind == "ctn_vec":
                values = x_dict[name].float()
                for idx, layer in enumerate(self.ctn_vec_linear_layer[name]):
                    part = layer(values[:, idx].view(-1, 1))
                    if self.continuous_mode == "dense":
                        dense_parts.append(part)
                    else:
                        linear_parts.append(part)

        batch_size = int(x_dict["duration"].shape[0])
        dense = torch.cat(dense_parts, dim=1) if dense_parts else x_dict["duration"].new_zeros((batch_size, 0))
        linear = torch.cat(linear_parts, dim=1) if linear_parts else None
        logger.debug("Baseline field inputs encoded. dense_dim=%d", int(dense.shape[1]))
        return dense, linear

    @staticmethod
    def _sequence_mask(x_dict: dict[str, torch.Tensor], name: str, values: torch.Tensor) -> torch.Tensor:

        mask = x_dict.get(f"{name}mask")
        if mask is None:
            mask = x_dict.get(f"{name}_mask")

        if mask is None or mask.shape != values.shape:
            return (values > 0).float()
        return mask.float()
