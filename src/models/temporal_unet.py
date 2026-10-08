
import torch
from src.models.unet import UNet


class TemporalUNet(UNet):
    """
    Early-fusion U-Net for binary building-change detection.

    Input:
        T1 image: [B, 3, H, W]
        T2 image: [B, 3, H, W]

    Output:
        Change logits: [B, 1, H, W]
    """

    def __init__(self):
        super().__init__(
            in_channels=6,
            num_classes=1,
        )

    def forward(self, image_t1, image_t2):
        if image_t1.shape != image_t2.shape:
            raise ValueError(
                "T1 and T2 images must have matching shapes."
            )

        combined = torch.cat(
            [image_t1, image_t2],
            dim=1,
        )

        return super().forward(combined)
