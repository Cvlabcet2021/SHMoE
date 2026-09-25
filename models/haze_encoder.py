# models/haze_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

        self.act = nn.GELU()

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.act(out)

        return out


class HazeEncoder(nn.Module):
    """
    Degradation-aware haze feature encoder.

    Input:
        RGB hazy image
        [B, 3, H, W]

    Output:
        Haze-aware feature map
        [B, embed_dim, H, W]

    The encoder is intentionally lightweight because its main
    purpose is to provide degradation information to the
    semantic-haze routing module.
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=32,
        embed_dim=128
    ):
        super().__init__()

        # Initial feature extraction
        self.stem = nn.Sequential(
            ConvBNAct(
                in_channels,
                base_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            ConvBNAct(
                base_channels,
                base_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        # Multi-scale feature extraction
        self.down1 = ConvBNAct(
            base_channels,
            base_channels * 2,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.res1 = nn.Sequential(
            ResidualBlock(base_channels * 2),
            ResidualBlock(base_channels * 2)
        )

        self.down2 = ConvBNAct(
            base_channels * 2,
            base_channels * 4,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.res2 = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4)
        )

        # Project multi-scale representation
        self.projection = nn.Sequential(
            nn.Conv2d(
                base_channels * 4,
                embed_dim,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        # Refine after upsampling
        self.refine = nn.Sequential(
            ConvBNAct(
                embed_dim,
                embed_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            ResidualBlock(embed_dim)
        )

    def forward(self, x):

        if x.dim() != 4:
            raise ValueError(
                "Input must have shape [B, C, H, W]"
            )

        # Original resolution
        x1 = self.stem(x)

        # 1/2 resolution
        x2 = self.down1(x1)
        x2 = self.res1(x2)

        # 1/4 resolution
        x3 = self.down2(x2)
        x3 = self.res2(x3)

        # Channel projection
        features = self.projection(x3)

        # Restore spatial resolution
        features = F.interpolate(
            features,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Final haze-aware representation
        features = self.refine(features)

        return features


if __name__ == "__main__":

    model = HazeEncoder(
        in_channels=3,
        base_channels=32,
        embed_dim=128
    )

    x = torch.randn(2, 3, 256, 256)

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("Parameters:", parameters)
