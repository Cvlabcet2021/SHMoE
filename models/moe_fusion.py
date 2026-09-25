# models/moe_fusion.py

import torch
import torch.nn as nn


class MoEFusion(nn.Module):
    """
    Soft mixture-of-experts fusion.

    Expert order:
        0 = ASM
        1 = MH-Former
        2 = HazeDiff-CPG
    """

    def __init__(self):
        super().__init__()

        # Small refinement after weighted fusion.
        self.refinement = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                3,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                32,
                3,
                3,
                padding=1,
            ),
        )

    def forward(
        self,
        asm_output,
        mhformer_output,
        diffusion_output,
        routing_probs,
    ):

        p_asm = routing_probs[:, 0:1]

        p_mh = routing_probs[:, 1:2]

        p_diff = routing_probs[:, 2:3]

        fused = (
            p_asm * asm_output
            + p_mh * mhformer_output
            + p_diff * diffusion_output
        )

        # Residual refinement.
        residual = self.refinement(
            fused
        )

        output = fused + 0.1 * residual

        output = torch.clamp(
            output,
            0.0,
            1.0,
        )

        return output
