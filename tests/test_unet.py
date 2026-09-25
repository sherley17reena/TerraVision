import torch

from src.models.unet import UNet


def test_unet_output_shape():

    model = UNet(
        in_channels=3,
        num_classes=7
    )

    x = torch.randn(
        1,
        3,
        256,
        256
    )

    output = model(x)

    assert output.shape == (
        1,
        7,
        256,
        256
    )


if __name__ == "__main__":
    test_unet_output_shape()
    print("U-Net test passed!")