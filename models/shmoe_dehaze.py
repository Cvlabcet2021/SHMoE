# models/shmoe_dehaze.py

import torch
import torch.nn as nn

from .sam3_encoder import SAM3Encoder
from .haze_encoder import HazeEncoder
from .routing import SemanticHazeRouting
from .gating import RoutingGating
from .asm_expert import ASMExpert
from .mhformer import MHFormer
from .hazediff_cpg import HazeDiffCPG
from .moe_fusion import MoEFusion


class SHMoEDehaze(nn.Module):

    def __init__(
        self,
        semantic_dim,
        haze_dim,
        routing_dim=128,
        routing_heads=4,
        mh_channels=48,
        diffusion_condition_dim=128,
        freeze_sam=True,
    ):
        super().__init__()

        # ------------------------------------------------
        # 1. Semantic encoder
        # ------------------------------------------------

        self.sam_encoder = SAM3Encoder()

        if freeze_sam:
            for p in self.sam_encoder.parameters():
                p.requires_grad = False

        # ------------------------------------------------
        # 2. Haze encoder
        # ------------------------------------------------

        self.haze_encoder = HazeEncoder()

        # ------------------------------------------------
        # 3. Semantic-haze routing
        # ------------------------------------------------

        self.routing = SemanticHazeRouting(
            semantic_dim=semantic_dim,
            haze_dim=haze_dim,
            routing_dim=routing_dim,
            num_heads=routing_heads,
        )

        # ------------------------------------------------
        # 4. Gating
        # ------------------------------------------------

        self.gating = RoutingGating(
            routing_dim=routing_dim,
            hidden_dim=routing_dim // 2,
            num_experts=3,
        )

        # ------------------------------------------------
        # 5. Experts
        # ------------------------------------------------

        self.asm_expert = ASMExpert()

        self.mhformer = MHFormer(
            base_channels=mh_channels,
        )

        self.hazediff = HazeDiffCPG(
            condition_dim=diffusion_condition_dim,
        )

        # ------------------------------------------------
        # 6. Fusion
        # ------------------------------------------------

        self.fusion = MoEFusion()

    def forward(
        self,
        x,
        timestep=None,
    ):

        # ================================================
        # Semantic features
        # ================================================

        semantic_features = self.sam_encoder(x)

        # ================================================
        # Haze features
        # ================================================

        haze_features = self.haze_encoder(x)

        # ================================================
        # Semantic-haze cross attention
        # ================================================

        routing_features = self.routing(
            semantic_features,
            haze_features,
        )

        # ================================================
        # Pixel-wise expert probabilities
        # ================================================

        routing_logits, routing_probs = (
            self.gating(
                routing_features
            )
        )

        # ================================================
        # ASM expert
        # ================================================

        asm_output, transmission, atmospheric_light = (
            self.asm_expert(x)
        )

        # ================================================
        # MH-Former expert
        # ================================================

        mh_output, K, B = self.mhformer(x)

        # ================================================
        # HazeDiff-CPG expert
        # ================================================

        diffusion_result = self.hazediff(
            x,
            timestep=timestep,
        )

        diffusion_output = diffusion_result[
            "restored"
        ]

        # ================================================
        # Soft MoE fusion
        # ================================================

        output = self.fusion(
            asm_output=asm_output,
            mhformer_output=mh_output,
            diffusion_output=diffusion_output,
            routing_probs=routing_probs,
        )

        return {
            "output": output,

            "asm_output": asm_output,
            "mhformer_output": mh_output,
            "diffusion_output": diffusion_output,

            "routing_logits": routing_logits,
            "routing_probs": routing_probs,

            "transmission": transmission,
            "atmospheric_light": atmospheric_light,

            "K": K,
            "B": B,

            "diffusion": diffusion_result,

            "semantic_features": semantic_features,
            "haze_features": haze_features,
            "routing_features": routing_features,
        }
