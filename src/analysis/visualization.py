import numpy as np
from PIL import Image


# RGB color for each TerraVision land-cover class.
CLASS_COLORS = {
    0: (128, 128, 128),  # Background
    1: (220, 20, 60),    # Building
    2: (255, 215, 0),    # Road
    3: (30, 144, 255),   # Water
    4: (210, 180, 140),  # Barren
    5: (34, 139, 34),    # Forest
    6: (154, 205, 50),   # Agriculture
}


def colorize_segmentation_mask(mask):
    """
    Convert a class-ID segmentation mask into an RGB image.
    """

    if mask.ndim != 2:
        raise ValueError(
            "Segmentation mask must be a 2D array."
        )

    height, width = mask.shape

    color_mask = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color

    return Image.fromarray(
        color_mask,
        mode="RGB",
    )


def create_change_map(change_mask):
    """
    Create a visual map showing changed and unchanged pixels.

    Black = unchanged
    Red   = changed
    """

    if change_mask.ndim != 2:
        raise ValueError(
            "Change mask must be a 2D array."
        )

    height, width = change_mask.shape

    visual = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    visual[change_mask] = (
        255,
        0,
        0,
    )

    return Image.fromarray(
        visual,
        mode="RGB",
    )