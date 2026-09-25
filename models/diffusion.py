# models/diffusion.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.dim = dim

    def forward(self, t):

        half = self.dim // 2

        emb_scale = math.log(10000) / (half - 1)

        emb = torch.exp(
            torch.arange(
                half,
                device=t.device,
                dtype=torch.float32,
            ) * -emb_scale
        )

        emb = t.float()[:, None] * emb[None, :]

        emb = torch.cat(
            [
                torch.sin(emb),
                torch.cos(emb),
            ],
            dim=1,
        )

        return emb


class FiLMBlock(nn.Module):

    def __init__(
        self,
        channels,
        condition_dim,
    ):
        super().__init__()

        self.norm = nn.GroupNorm(
            num_groups=8,
            num_channels=channels,
        )

        self.conv = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
        )

        self.condition = nn.Linear(
            condition_dim,
            channels * 2,
        )

    def forward(self, x, condition):

        scale_shift = self.condition(
            condition
        )

        scale, shift = torch.chunk(
            scale_shift,
            2,
            dim=1,
        )

        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        x = self.norm(x)

        x = x * (
            1.0 + scale
        ) + shift

        x = F.silu(x)

        return self.conv(x)


class ConditionalDiffusionUNet(nn.Module):

    def __init__(
        self,
        image_channels=3,
        base_channels=64,
        condition_dim=128,
    ):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(
                base_channels
            ),
            nn.Linear(
                base_channels,
                condition_dim,
            ),
            nn.SiLU(),
        )

        self.input = nn.Conv2d(
            image_channels,
            base_channels,
            3,
            padding=1,
        )

        self.block1 = FiLMBlock(
            base_channels,
            condition_dim,
        )

        self.block2 = FiLMBlock(
            base_channels,
            condition_dim,
        )

        self.output = nn.Conv2d(
            base_channels,
            image_channels,
            3,
            padding=1,
        )

    def forward(
        self,
        x,
        timestep,
        condition=None,
    ):

        t_emb = self.time_embedding(
            timestep
        )

        if condition is not None:
            t_emb = t_emb + condition

        f = self.input(x)

        f = self.block1(
            f,
            t_emb,
        )

        f = self.block2(
            f,
            t_emb,
        )

        return self.output(f)
