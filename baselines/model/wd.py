from __future__ import annotations

import logging

import torch
from torch import nn

from .common import FieldFeatureEncoder
from .layers import MultiLayerPerceptron

logger = logging.getLogger(__name__)


class WideAndDeep(nn.Module):
    def __init__(self, description: list[tuple[str, int, str]], embed_dim: int, mlp_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        logger.info("Initializing Wide&Deep baseline. embed_dim=%d mlp_dims=%s", int(embed_dim), mlp_dims)

        self.encoder = FieldFeatureEncoder(description, embed_dim, continuous_mode="wide")

        self.mlp = MultiLayerPerceptron(self.encoder.output_dim, mlp_dims, dropout)
        logger.info("Wide&Deep baseline initialized. input_dim=%d", self.encoder.output_dim)

    def forward_with_hidden(self, x_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:

        dense, linear = self.encoder(x_dict)

        hidden = self.mlp.mlp[:-1](dense)
        res = self.mlp.mlp[-1](hidden)
        if linear is not None:
            res = res + linear.sum(dim=1, keepdims=True)

        pred = torch.sigmoid(res.squeeze(1))
        return pred, hidden

    def forward(self, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        pred, _ = self.forward_with_hidden(x_dict)
        return pred
