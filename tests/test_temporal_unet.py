
import torch

from src.models.temporal_unet import TemporalUNet


def test_temporal_unet():
    model = TemporalUNet()
    model.eval()

    t1 = torch.randn(1, 3, 256, 256)
    t2 = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        output = model(t1, t2)

    assert output.shape == (1, 1, 256, 256)

    print("Temporal U-Net output:", output.shape)
    print("Temporal U-Net architecture test passed!")


if __name__ == "__main__":
    test_temporal_unet()
