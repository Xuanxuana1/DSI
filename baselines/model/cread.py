from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from .common import FieldFeatureEncoder
from .layers import MultiLayerPerceptron

logger = logging.getLogger(__name__)


class Cread(nn.Module):
    def __init__(
        self,
        description: list[tuple[str, int, str]],
        embed_dim: int,
        share_mlp_dims: tuple[int, ...],
        output_mlp_dims: tuple[int, ...],
        head_num: int,
        dropout: float,
    ) -> None:
        super().__init__()
        logger.info("Initializing CREAD baseline. head_num=%d", int(head_num))

        self.build(
            description,
            int(embed_dim),
            tuple(int(v) for v in share_mlp_dims),
            tuple(int(v) for v in output_mlp_dims),
            int(head_num),
            float(dropout),
        )
        logger.info("CREAD baseline initialized. input_dim=%d", int(self.embed_output_dim))

    def build(
        self,
        description: list[tuple[str, int, str]],
        embed_dim: int,
        share_mlp_dims: tuple[int, ...],
        output_mlp_dims: tuple[int, ...],
        head_num: int,
        dropout: float,
    ) -> None:
        logger.info("Building CREAD layers.")

        self.encoder = FieldFeatureEncoder(description, int(embed_dim), continuous_mode="dense")

        if not share_mlp_dims:
            raise ValueError("CREAD share_mlp_dims must not be empty")
        self.embed_output_dim = int(self.encoder.output_dim)
        self.share_mlp = MultiLayerPerceptron(
            int(self.encoder.output_dim),
            tuple(int(v) for v in share_mlp_dims),
            float(dropout),
            output_layer=False,
        )
        self.output_mlps = nn.ModuleList(
            MultiLayerPerceptron(
                int(share_mlp_dims[-1]),
                tuple(int(v) for v in output_mlp_dims),
                float(dropout),
                output_layer=True,
            )
            for _ in range(int(head_num))
        )
        logger.info("CREAD layers built.")

    def init(self) -> None:
        logger.info("Initializing CREAD parameters.")

        for parameter in self.parameters():
            nn.init.uniform_(parameter, -0.01, 0.01)
        logger.info("CREAD parameters initialized.")

    def forward_with_hidden(self, x_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logger.debug("Running the CREAD forward pass.")

        encoded, _ = self.encoder(x_dict)
        shared = self.share_mlp(encoded)

        outputs = [output_mlp(shared) for output_mlp in self.output_mlps]
        result = torch.sigmoid(torch.cat(outputs, dim=1))
        logger.debug("CREAD forward pass complete. heads=%d", int(result.shape[1]))
        return result, shared

    def forward(self, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        pred, _ = self.forward_with_hidden(x_dict)
        return pred
