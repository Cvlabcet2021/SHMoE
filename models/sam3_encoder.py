import torch
import torch.nn as nn
import torch.nn.functional as F


class SAM3Encoder(nn.Module):
    """
    Frozen SAM-3 semantic feature extractor.

    The SHMoE-Dehaze paper uses pretrained SAM-3 semantic
    features and keeps the SAM-3 parameters frozen.

    This class is intentionally implemented as an adapter.
    The exact SAM-3 loading/API depends on the SAM-3 package
    and checkpoint being used.

    Expected output:
        [B, feature_dim, H', W']
    """

    def __init__(
        self,
        sam3_model=None,
        feature_dim=256,
        output_dim=256
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.output_dim = output_dim

        # -----------------------------------------------------
        # SAM-3 model
        # -----------------------------------------------------

        self.sam3 = sam3_model

        # -----------------------------------------------------
        # Feature projection
        # -----------------------------------------------------

        if feature_dim != output_dim:

            self.projection = nn.Sequential(
                nn.Conv2d(
                    feature_dim,
                    output_dim,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(output_dim),
                nn.GELU()
            )

        else:

            self.projection = nn.Identity()

        # -----------------------------------------------------
        # Freeze SAM-3
        # -----------------------------------------------------

        if self.sam3 is not None:

            for parameter in self.sam3.parameters():
                parameter.requires_grad = False

            self.sam3.eval()

    # =========================================================
    # Prevent SAM-3 from entering training mode
    # =========================================================

    def train(self, mode=True):

        super().train(mode)

        # SAM-3 must remain frozen and in evaluation mode.

        if self.sam3 is not None:
            self.sam3.eval()

        return self

    # =========================================================
    # Extract raw SAM-3 features
    # =========================================================

    @torch.no_grad()
    def _extract_sam3_features(self, x):

        if self.sam3 is None:

            raise RuntimeError(
                "SAM-3 model has not been supplied. "
                "Load the appropriate SAM-3 checkpoint/API "
                "and pass the model to SAM3Encoder."
            )

        """
        IMPORTANT:

        The exact call below depends on the SAM-3 implementation.

        Possible implementations expose an image encoder,
        vision encoder, backbone, or feature extractor.

        Replace this section with the feature extraction call
        corresponding to the SAM-3 release/checkpoint used.
        """

        if hasattr(self.sam3, "image_encoder"):

            features = self.sam3.image_encoder(x)

        elif hasattr(self.sam3, "vision_encoder"):

            features = self.sam3.vision_encoder(x)

        elif hasattr(self.sam3, "encode_image"):

            features = self.sam3.encode_image(x)

        elif hasattr(self.sam3, "get_image_embeddings"):

            features = self.sam3.get_image_embeddings(x)

        else:

            raise AttributeError(
                "The supplied SAM-3 object does not expose a "
                "recognized image-feature extraction interface. "
                "Connect the official SAM-3 image encoder here."
            )

        return features

    # =========================================================
    # Convert token features to spatial feature map
    # =========================================================

    @staticmethod
    def _tokens_to_feature_map(features):

        """
        Converts common token layouts into:

            [B, C, H, W]

        Supported layouts:

            [B, C, H, W]
            [B, H, W, C]
            [B, N, C]

        For [B, N, C], N must correspond to a square
        spatial token grid.
        """

        # Already a spatial feature map
        if features.ndim == 4:

            # Assume BCHW if channel dimension is reasonably small
            # relative to spatial dimensions.

            B, D1, D2, D3 = features.shape

            if D1 <= D2 and D1 <= D3:
                return features

            # Otherwise assume BHWC
            return features.permute(
                0, 3, 1, 2
            ).contiguous()

        # Token representation
        if features.ndim == 3:

            B, N, C = features.shape

            grid_size = int(N ** 0.5)

            if grid_size * grid_size != N:

                raise ValueError(
                    f"Cannot reshape {N} tokens into a square "
                    "spatial feature map."
                )

            features = features.transpose(
                1,
                2
            )

            features = features.reshape(
                B,
                C,
                grid_size,
                grid_size
            )

            return features

        raise ValueError(
            "Unsupported SAM-3 feature shape: "
            f"{features.shape}"
        )

    # =========================================================
    # Forward
    # =========================================================

    def forward(self, x):

        # SAM-3 is frozen
        with torch.no_grad():

            features = self._extract_sam3_features(x)

        # Convert to BCHW
        features = self._tokens_to_feature_map(
            features
        )

        # Project semantic representation
        features = self.projection(
            features
        )

        return features


# =============================================================
# Dummy SAM-3 adapter for testing
# =============================================================

class DummySAM3(nn.Module):
    """
    Dummy backbone used ONLY for testing the rest of the
    SHMoE-Dehaze pipeline.

    This is NOT SAM-3.
    """

    def __init__(
        self,
        in_channels=3,
        feature_dim=256
    ):

        super().__init__()

        self.image_encoder = nn.Sequential(

            nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=4,
                padding=3
            ),

            nn.GELU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                128,
                feature_dim,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.GELU()
        )


# =============================================================
# Test
# =============================================================

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------
    # Dummy backbone
    # ---------------------------------------------------------

    dummy_sam3 = DummySAM3(
        in_channels=3,
        feature_dim=256
    )

    encoder = SAM3Encoder(
        sam3_model=dummy_sam3,
        feature_dim=256,
        output_dim=256
    )

    encoder = encoder.to(device)

    # ---------------------------------------------------------
    # Test input
    # ---------------------------------------------------------

    x = torch.randn(
        2,
        3,
        512,
        512,
        device=device
    )

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    with torch.no_grad():

        features = encoder(x)

    print(
        "Input:",
        x.shape
    )

    print(
        "SAM-3 semantic features:",
        features.shape
    )

    # ---------------------------------------------------------
    # Verify freezing
    # ---------------------------------------------------------

    trainable = sum(
        p.numel()
        for p in encoder.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in encoder.parameters()
    )

    print(
        f"Total parameters: {total:,}"
    )

    print(
        f"Trainable parameters: {trainable:,}"
    )
