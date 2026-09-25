# models/routing.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention between semantic features and haze-aware features.

    Query  : semantic features
    Key/Value : haze-aware features

    Input:
        semantic_features: [B, Cs, H, W]
        haze_features:     [B, Ch, H, W]

    Output:
        routing_features:  [B, Cr, H, W]
    """

    def __init__(
        self,
        semantic_dim,
        haze_dim,
        routing_dim=128,
        num_heads=4,
        dropout=0.0,
    ):
        super().__init__()

        if routing_dim % num_heads != 0:
            raise ValueError(
                "routing_dim must be divisible by num_heads."
            )

        self.semantic_proj = nn.Conv2d(
            semantic_dim,
            routing_dim,
            kernel_size=1,
            bias=False,
        )

        self.haze_proj = nn.Conv2d(
            haze_dim,
            routing_dim,
            kernel_size=1,
            bias=False,
        )

        self.norm_q = nn.LayerNorm(routing_dim)
        self.norm_kv = nn.LayerNorm(routing_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=routing_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(routing_dim, routing_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(routing_dim * 4, routing_dim),
        )

        self.norm_out = nn.LayerNorm(routing_dim)

    def forward(self, semantic_features, haze_features):
        """
        Args:
            semantic_features: [B, Cs, H, W]
            haze_features:     [B, Ch, H, W]

        Returns:
            routing_features: [B, routing_dim, H, W]
        """

        if semantic_features.ndim != 4:
            raise ValueError(
                "semantic_features must have shape [B,C,H,W]"
            )

        if haze_features.ndim != 4:
            raise ValueError(
                "haze_features must have shape [B,C,H,W]"
            )

        B, _, H, W = semantic_features.shape

        # Match spatial resolution.
        if haze_features.shape[-2:] != (H, W):
            haze_features = F.interpolate(
                haze_features,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        # Project to common routing dimension.
        q = self.semantic_proj(semantic_features)
        kv = self.haze_proj(haze_features)

        # [B,C,H,W] -> [B,HW,C]
        q = q.flatten(2).transpose(1, 2)
        kv = kv.flatten(2).transpose(1, 2)

        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        # Semantic query attends to haze representation.
        attn_out, attn_weights = self.attention(
            query=q,
            key=kv,
            value=kv,
            need_weights=False,
        )

        # Residual cross-attention.
        x = q + attn_out

        # Feed-forward refinement.
        x = x + self.ffn(self.norm_out(x))

        # [B,HW,C] -> [B,C,H,W]
        x = x.transpose(1, 2).reshape(
            B,
            -1,
            H,
            W,
        )

        return x


class SemanticHazeRouting(nn.Module):
    """
    Semantic-haze routing module.

    Combines:
        SAM-3 semantic features
        +
        haze-aware features

    using cross-attention.

    The output is subsequently passed to the gating network.
    """

    def __init__(
        self,
        semantic_dim,
        haze_dim,
        routing_dim=128,
        num_heads=4,
        dropout=0.0,
    ):
        super().__init__()

        self.cross_attention = CrossAttentionBlock(
            semantic_dim=semantic_dim,
            haze_dim=haze_dim,
            routing_dim=routing_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, semantic_features, haze_features):
        return self.cross_attention(
            semantic_features,
            haze_features,
        )
