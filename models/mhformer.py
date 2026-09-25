# models/mhformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConvBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.block(x)


class WindowMHSA(nn.Module):
    """
    Window-based multi-head self-attention.

    For the first implementation, attention is applied
    on flattened spatial tokens. A windowed version can
    be substituted later without changing the expert API.
    """

    def __init__(
        self,
        channels,
        num_heads=4,
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                "channels must be divisible by num_heads"
            )

        self.norm = nn.LayerNorm(channels)

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

        self.proj = nn.Linear(
            channels,
            channels,
        )

    def forward(self, x):

        B, C, H, W = x.shape

        tokens = (
            x.flatten(2)
            .transpose(1, 2)
        )

        residual = tokens

        tokens = self.norm(tokens)

        attn_out, _ = self.attn(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )

        tokens = residual + self.proj(attn_out)

        x = (
            tokens
            .transpose(1, 2)
            .reshape(B, C, H, W)
        )

        return x


class SelectiveKernelFusion(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.branch1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
        )

        self.branch2 = nn.Conv2d(
            channels,
            channels,
            5,
            padding=2,
            groups=channels,
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                channels,
                channels,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):

        b1 = self.branch1(x)
        b2 = self.branch2(x)

        weight = self.attention(x)

        return weight * b1 + (1.0 - weight) * b2


class MHFormerBlock(nn.Module):

    def __init__(
        self,
        channels,
        num_heads=4,
    ):
        super().__init__()

        self.dwconv = DWConvBlock(
            channels
        )

        self.attention = WindowMHSA(
            channels,
            num_heads=num_heads,
        )

        self.sk = SelectiveKernelFusion(
            channels
        )

    def forward(self, x):

        x = self.dwconv(x)
        x = self.attention(x)
        x = self.sk(x)

        return x


class MHFormer(nn.Module):
    """
    Medium-haze restoration expert.

    Output:
        restored image
        K scaling map
        B atmospheric compensation map
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=48,
        num_blocks=4,
        num_heads=4,
    ):
        super().__init__()

        self.input_proj = nn.Conv2d(
            in_channels,
            base_channels,
            3,
            padding=1,
        )

        self.blocks = nn.Sequential(
            *[
                MHFormerBlock(
                    base_channels,
                    num_heads=num_heads,
                )
                for _ in range(num_blocks)
            ]
        )

        self.output = nn.Conv2d(
            base_channels,
            6,
            3,
            padding=1,
        )

    def forward(self, x):

        f = self.input_proj(x)

        f = self.blocks(f)

        params = self.output(f)

        K_raw = params[:, 0:3]
        B_raw = params[:, 3:6]

        # Keep K around 1.
        K = 1.0 + 0.5 * torch.tanh(K_raw)

        # B is bounded.
        B = 0.5 * torch.tanh(B_raw)

        restored = K * x + B

        restored = torch.clamp(
            restored,
            0.0,
            1.0,
        )

        return restored, K, B
