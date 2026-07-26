from __future__ import annotations

import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim: int, embed_dims: tuple[int, ...], dropout: float, output_layer: bool = True) -> None:
        super().__init__()
        logger = __import__("logging").getLogger(__name__)
        logger.info("Initializing MLP. input_dim=%d hidden=%s output_layer=%s", int(input_dim), embed_dims, output_layer)

        layers: list[nn.Module] = []
        current_dim = int(input_dim)
        for embed_dim in embed_dims:
            layers.append(nn.Linear(current_dim, int(embed_dim)))
            layers.append(nn.BatchNorm1d(int(embed_dim)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=float(dropout)))
            current_dim = int(embed_dim)

        if output_layer:
            layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)
        logger.info("MLP initialized.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class MultiLayerPerceptronD2Q(nn.Module):
    def __init__(self, input_dim: int, embed_dims: tuple[int, ...], dropout: float, output_layer: bool = True) -> None:
        super().__init__()
        logger = __import__("logging").getLogger(__name__)
        logger.info("Initializing D2Q MLP. input_dim=%d hidden=%s", int(input_dim), embed_dims)

        layers: list[nn.Module] = []
        current_dim = int(input_dim)
        for embed_dim in embed_dims:
            layers.append(nn.Linear(current_dim, int(embed_dim)))
            layers.append(nn.BatchNorm1d(int(embed_dim)))
            layers.append(Swish())
            layers.append(nn.Dropout(p=float(dropout)))
            current_dim = int(embed_dim)

        if output_layer:
            layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)
        logger.info("D2Q MLP initialized.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MultiLayerPerceptronTPM(nn.Module):
    def __init__(self, input_dim: int, embed_dims: tuple[int, ...], dropout: float, class_num: int, output_layer: bool = True) -> None:
        super().__init__()
        logger = __import__("logging").getLogger(__name__)
        logger.info("Initializing TPM MLP. input_dim=%d hidden=%s class_num=%d", int(input_dim), embed_dims, int(class_num))

        layers: list[nn.Module] = []
        current_dim = int(input_dim)
        for embed_dim in embed_dims:
            layers.append(nn.Linear(current_dim, int(embed_dim)))
            layers.append(nn.BatchNorm1d(int(embed_dim)))
            layers.append(Swish())
            layers.append(nn.Dropout(p=float(dropout)))
            current_dim = int(embed_dim)

        if output_layer:
            layers.append(nn.Linear(current_dim, int(class_num)))
        self.mlp = nn.Sequential(*layers)
        logger.info("TPM MLP initialized.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
