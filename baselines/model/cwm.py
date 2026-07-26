from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from .common import feature_sizes

logger = logging.getLogger(__name__)


class CWM(nn.Module):
    def __init__(self, description: list[tuple[str, Any, str]], embed_dim: int) -> None:
        super().__init__()
        logger.info("Initializing CWM baseline. embed_dim=%d", int(embed_dim))

        self.spec = feature_sizes(description)
        self.sparse_names = [name for name, (_, kind) in self.spec.items() if kind == "spr"]
        self.sparse_vec_names = [name for name, (_, kind) in self.spec.items() if kind == "spr_vec"]

        self.scalar_embeddings = nn.ModuleDict({
            name: nn.Embedding(max(int(self.spec[name][0]), 1), int(embed_dim))
            for name in self.sparse_names
        })
        self.scalar_linears = nn.ModuleDict({
            name: nn.Embedding(max(int(self.spec[name][0]), 1), 1)
            for name in self.sparse_names
        })

        self.vector_embeddings = nn.ModuleDict({
            name: nn.ModuleList(
                nn.Embedding(max(int(cardinality), 1), int(embed_dim))
                for cardinality in list(self.spec[name][0])
            )
            for name in self.sparse_vec_names
        })
        self.vector_linears = nn.ModuleDict({
            name: nn.ModuleList(
                nn.Embedding(max(int(cardinality), 1), 1)
                for cardinality in list(self.spec[name][0])
            )
            for name in self.sparse_vec_names
        })
        self.bias = nn.Parameter(torch.zeros(1))
        logger.info(
            "CWM baseline initialized. scalar_fields=%d vector_groups=%d",
            len(self.sparse_names),
            len(self.sparse_vec_names),
        )

    def forward(self, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        logger.debug("Running the CWM forward pass.")

        emb_parts: list[torch.Tensor] = []
        linear_parts: list[torch.Tensor] = []
        for name in self.sparse_names:
            values = x_dict[name].view(-1).long().clamp_min(0).clamp_max(self.scalar_embeddings[name].num_embeddings - 1)
            emb_parts.append(self.scalar_embeddings[name](values))
            linear_parts.append(self.scalar_linears[name](values))

        for name in self.sparse_vec_names:
            values = x_dict[name].long()
            if values.shape[1] == 0:
                continue
            for idx, embedding in enumerate(self.vector_embeddings[name]):
                value = values[:, idx].clamp_min(0).clamp_max(embedding.num_embeddings - 1)
                emb_parts.append(embedding(value))
                linear_parts.append(self.vector_linears[name][idx](value))

        if not emb_parts:
            batch_size = int(x_dict["duration"].shape[0])
            logger.debug("CWM forward pass complete. No categorical fields.")
            return self.bias.expand(batch_size)
        emb = torch.stack(emb_parts, dim=1)
        linear = torch.stack(linear_parts, dim=1).sum(dim=1).squeeze(1)
        square_of_sum = torch.sum(emb, dim=1).pow(2)
        sum_of_square = torch.sum(emb.pow(2), dim=1)
        fm = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)

        score = linear + fm + self.bias
        logger.debug("CWM forward pass complete.")
        return score.view(-1)
