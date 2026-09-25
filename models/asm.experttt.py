# models/asm_expert.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransmissionEstimator(nn.Module):
    """
    Lightweight transmission estimation network.
    """

    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                base_channels,
                base_channels,
                3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                base_channels,
                1,
                3,
                padding=1,
            ),
        )

    def forward(self, x):
        # Transmission is constrained to [0,1].
        return torch.sigmoid(self.net(x))


class AtmosphericLightEstimator(nn.Module):
    """
    Estimates global atmospheric light A.
    """

    def __init__(self, in_channels=3, feature_channels=32):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels,
                feature_channels,
                3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                feature_channels,
                feature_channels,
                3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Conv2d(
                feature_channels,
                3,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        f = self.features(x)
        a = self.pool(f)
        a = self.head(a)

        # [B,3,1,1]
        return a


class ASMExpert(nn.Module):
    """
    Atmospheric Scattering Model expert.

    Intended for lightly degraded regions.
    """

    def __init__(
        self,
        in_channels=3,
        t0=0.12,
    ):
        super().__init__()

        self.t0 = t0

        self.transmission = TransmissionEstimator(
            in_channels=in_channels
        )

        self.atmospheric_light = AtmosphericLightEstimator(
            in_channels=in_channels
        )

    def forward(self, x):
        """
        Args:
            x: [B,3,H,W]

        Returns:
            restored: [B,3,H,W]
            transmission: [B,1,H,W]
            atmospheric_light: [B,3,1,1]
        """

        t = self.transmission(x)

        A = self.atmospheric_light(x)

        # Numerical stability.
        t_safe = torch.clamp(
            t,
            min=self.t0,
            max=1.0,
        )

        # I = J*t + A*(1-t)
        #
        # J = (I - A*(1-t))/t
        restored = (
            x - A * (1.0 - t_safe)
        ) / t_safe

        restored = torch.clamp(
            restored,
            0.0,
            1.0,
        )

        return restored, t, A
