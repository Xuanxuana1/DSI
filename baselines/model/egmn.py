from __future__ import annotations

import logging

import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F

from .common import FieldFeatureEncoder
from .layers import MultiLayerPerceptron

logger = logging.getLogger(__name__)


class EGMN(nn.Module):
    def __init__(
        self,
        description: list[tuple[str, int, str]],
        embed_dim: int,
        share_mlp_dims: tuple[int, ...],
        dropout: float,
        components: int = 10,
    ) -> None:
        super().__init__()
        logger.info("Initializing EGMN baseline. components=%d", int(components))

        self.components = int(components)
        self.encoder = FieldFeatureEncoder(description, embed_dim, continuous_mode="dense")
        self.share_mlp = MultiLayerPerceptron(self.encoder.output_dim, share_mlp_dims, dropout, output_layer=False)
        hidden_dim = int(share_mlp_dims[-1])

        self.lambda_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus(beta=0.5))
        self.mixture_logits = nn.Linear(hidden_dim, self.components + 1)
        self.gauss_mu = nn.Sequential(nn.Linear(hidden_dim, self.components), nn.Softplus())
        self.gauss_sigma = nn.Sequential(nn.Linear(hidden_dim, self.components), nn.Softplus())
        logger.info("EGMN baseline initialized. input_dim=%d", self.encoder.output_dim)

    def forward_with_hidden(self, x_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        dense, _ = self.encoder(x_dict)
        hidden = self.share_mlp(dense)

        lambda_ = self.lambda_layer(hidden) + 1e-6
        pi = self.mixture_logits(hidden)
        mu = self.gauss_mu(hidden) + 1.0 / lambda_
        sigma = self.gauss_sigma(hidden) + 1e-6
        return pi, lambda_, mu, sigma, hidden

    def forward(self, x_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pi, lambda_, mu, sigma, _ = self.forward_with_hidden(x_dict)
        return pi, lambda_, mu, sigma

    def loss(self, y_true: torch.Tensor, pi: torch.Tensor, lambda_: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logger.debug("Computing EGMN loss.")

        batch_size = y_true.shape[0]
        target = y_true.view(-1, 1)

        exp_dist = D.Exponential(rate=lambda_.view(-1))
        log_prob_short = exp_dist.log_prob(target.view(-1)).view(batch_size, 1)

        log_prob_all = []
        for comp_idx in range(mu.shape[1]):
            normal_dist = D.Normal(loc=mu[:, comp_idx], scale=sigma[:, comp_idx])
            trunc_min = torch.zeros_like(mu[:, comp_idx])
            prob_long = 1.0 - normal_dist.cdf(trunc_min)
            target_clamped = target.view(-1).clamp(min=1e-6)
            log_prob = normal_dist.log_prob(target_clamped) - torch.log(prob_long.clamp(min=1e-6))
            log_prob_all.append(log_prob.view(-1, 1))
        log_prob = torch.cat([log_prob_short] + log_prob_all, dim=1)

        log_mix_probs = torch.log_softmax(pi, dim=1)
        nll = -torch.logsumexp(log_mix_probs + log_prob, dim=1).mean()
        mix_probs = torch.softmax(pi, dim=1)
        pred = torch.sum(mix_probs * torch.cat([1.0 / lambda_, mu], dim=1), dim=1, keepdim=True)
        reg = F.l1_loss(pred, target.float())
        entropy = -torch.sum(mix_probs * torch.log(mix_probs + 1e-6), dim=1).mean()
        logger.debug("EGMN loss computed. nll=%.6f reg=%.6f", float(nll.detach().cpu().item()), float(reg.detach().cpu().item()))
        return nll, reg, entropy

    def predict(self, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:

        pi, lambda_, mu, _ = self.forward(x_dict)

        pi_soft = torch.softmax(pi, dim=1)
        means = torch.cat([1.0 / lambda_, mu], dim=1)
        return (pi_soft * means).sum(dim=1, keepdim=True)
