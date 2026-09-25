# losses/reconstruction.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier_loss(
    pred,
    target,
    eps=1e-3,
):
    diff = pred - target
    loss = torch.sqrt(
        diff * diff + eps * eps
    )

    return loss.mean()


def l1_loss(pred, target):
    return F.l1_loss(
        pred,
        target,
    )


def mse_loss(pred, target):
    return F.mse_loss(
        pred,
        target,
    )


class ReconstructionLoss(nn.Module):

    def __init__(
        self,
        l1_weight=1.0,
        mse_weight=0.0,
        charbonnier_weight=0.0,
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.charbonnier_weight = (
            charbonnier_weight
        )

    def forward(
        self,
        pred,
        target,
    ):

        loss = 0.0

        if self.l1_weight > 0:
            loss = loss + (
                self.l1_weight
                * l1_loss(pred, target)
            )

        if self.mse_weight > 0:
            loss = loss + (
                self.mse_weight
                * mse_loss(pred, target)
            )

        if self.charbonnier_weight > 0:
            loss = loss + (
                self.charbonnier_weight
                * charbonnier_loss(
                    pred,
                    target,
                )
            )

        return loss
