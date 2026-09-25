# models/gating.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutingGating(nn.Module):
    """
    Pixel-wise expert gating network.

    Input:
        routing_features [B, C, H, W]

    Output:
        routing_logits [B, 3, H, W]
        routing_probs  [B, 3, H, W]

    Expert order:

        0 -> ASM
        1 -> MH-Former
        2 -> HazeDiff-CPG
    """

    def __init__(
        self,
        routing_dim=128,
        hidden_dim=64,
        num_experts=3,
    ):
        super().__init__()

        self.num_experts = num_experts

        self.gate = nn.Sequential(
            nn.Conv2d(
                routing_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),

            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),

            nn.Conv2d(
                hidden_dim,
                num_experts,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(self, routing_features):
        """
        Args:
            routing_features: [B,C,H,W]

        Returns:
            logits: [B,3,H,W]
            probs:  [B,3,H,W]
        """

        logits = self.gate(routing_features)

        probs = F.softmax(
            logits,
            dim=1,
        )

        return logits, probs
    
