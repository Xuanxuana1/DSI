from __future__ import annotations

import logging

import torch
from torch import nn

from .common import FieldFeatureEncoder
from .layers import MultiLayerPerceptronD2Q

logger = logging.getLogger(__name__)


class D2Q(nn.Module):
    def __init__(
        self,
        description: list[tuple[str, int, str]],
        embed_dim: int,
        mlp_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        logger.info("开始初始化D2Q baseline。")

        self.encoder = FieldFeatureEncoder(description, int(embed_dim), continuous_mode="dense")

        self.mlp = MultiLayerPerceptronD2Q(self.encoder.output_dim, mlp_dims, dropout)
        logger.info("D2Q baseline初始化完成。input_dim=%d", self.encoder.output_dim)

    def forward_with_hidden(self, x_dict: dict[str, torch.Tensor]):

        dense, _ = self.encoder(x_dict)

        hidden = self.mlp.mlp[:-1](dense)
        output = self.mlp.mlp[-1](hidden)
        return torch.sigmoid(output.squeeze(1)), hidden

    def forward(self, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        pred, _ = self.forward_with_hidden(x_dict)
        return pred
