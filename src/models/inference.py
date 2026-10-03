from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.models.device import get_device
from src.models.unet import UNet


IMAGE_SIZE = 256

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_unet_model(
    model_path="outputs/models/unet_experiment.pth",
):
    """
    Load the trained TerraVision U-Net model.
    """

    device = get_device()

    model = UNet(
        in_channels=3,
        num_classes=7,
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
        )
    )

    model = model.to(device)
    model.eval()

    return model, device


def preprocess_image(
    image_path,
    image_size=IMAGE_SIZE,
):
    """
    Prepare a satellite image for U-Net inference.
    """

    image_path = Path(image_path)

    image = Image.open(
        image_path
    ).convert("RGB")

    original_size = image.size

    image = TF.resize(
        image,
        [image_size, image_size],
        interpolation=TF.InterpolationMode.BILINEAR,
    )

    image = TF.to_tensor(image)

    image = TF.normalize(
        image,
        mean=MEAN,
        std=STD,
    )

    # Add batch dimension:
    # [3, H, W] -> [1, 3, H, W]
    image = image.unsqueeze(0)

    return image, original_size


def predict_mask(
    model,
    device,
    image_path,
):
    """
    Generate a land-cover segmentation mask
    for a satellite image.
    """

    image, original_size = preprocess_image(
        image_path
    )

    image = image.to(device)

    with torch.no_grad():

        output = model(image)

        prediction = torch.argmax(
            output,
            dim=1,
        )

    mask = (
        prediction
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    return mask, original_size