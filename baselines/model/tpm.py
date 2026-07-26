from __future__ import annotations

import logging

import torch
from torch import nn

from .common import FieldFeatureEncoder
from .layers import MultiLayerPerceptronTPM

logger = logging.getLogger(__name__)


class TPM(nn.Module):
    def __init__(
        self,
        description: list[tuple[str, int, str]],
        class_num: int,
        embed_dim: int,
        mlp_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        logger.info("Initializing TPM baseline. class_num=%d", int(class_num))

        self.encoder = FieldFeatureEncoder(description, embed_dim, continuous_mode="dense")

        self.mlp = MultiLayerPerceptronTPM(self.encoder.output_dim, mlp_dims, dropout, class_num)
        logger.info("TPM baseline initialized. input_dim=%d", self.encoder.output_dim)

    def forward_with_hidden(self, x_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:

        dense, _ = self.encoder(x_dict)

        hidden = self.mlp.mlp[:-1](dense)
        output = self.mlp.mlp[-1](hidden)
        return torch.sigmoid(output), hidden

    def forward(self, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        pred, _ = self.forward_with_hidden(x_dict)
        return pred
