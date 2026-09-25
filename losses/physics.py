# losses/physics.py

import torch
import torch.nn as nn


class AtmosphericConsistencyLoss(nn.Module):
    """
    Enforces:

        I = J*t + A*(1-t)

    """

    def forward(
        self,
        hazy,
        restored,
        transmission,
        atmospheric_light,
    ):

        reconstructed_hazy = (
            restored * transmission
            + atmospheric_light
            * (1.0 - transmission)
        )

        return torch.mean(
            torch.abs(
                hazy - reconstructed_hazy
            )
        )


class RoutingBalanceLoss(nn.Module):

    def forward(self, routing_probs):

        # [B,K,H,W]
        mean_probs = routing_probs.mean(
            dim=(0, 2, 3)
        )

        K = routing_probs.shape[1]

        target = torch.full_like(
            mean_probs,
            1.0 / K,
        )

        return torch.mean(
            (mean_probs - target) ** 2
        )


class RoutingSmoothnessLoss(nn.Module):

    def forward(self, routing_probs):

        dx = torch.abs(
            routing_probs[:, :, :, 1:]
            - routing_probs[:, :, :, :-1]
        )

        dy = torch.abs(
            routing_probs[:, :, 1:, :]
            - routing_probs[:, :, :-1, :]
        )

        return (
            dx.mean()
            + dy.mean()
        )
