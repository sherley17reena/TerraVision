from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = [
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agriculture",
]


def main():

    mask_dirs = [
        Path("data/raw/LoveDA/Train/Urban/masks_png"),
        Path("data/raw/LoveDA/Train/Rural/masks_png"),
    ]

    class_counts = np.zeros(7, dtype=np.int64)
    ignored_pixels = 0

    for mask_dir in mask_dirs:

        for mask_path in mask_dir.glob("*.png"):

            mask = np.array(
                Image.open(mask_path),
                dtype=np.uint8
            )

            ignored_pixels += np.sum(mask == 0)

            # Raw LoveDA classes are 1-7
            for raw_class in range(1, 8):
                class_counts[raw_class - 1] += np.sum(
                    mask == raw_class
                )

    valid_pixels = class_counts.sum()

    print("\nLoveDA Training Class Distribution\n")

    for class_id, class_name in enumerate(CLASS_NAMES):

        percentage = (
            class_counts[class_id]
            / valid_pixels
            * 100
        )

        print(
            f"{class_id}: "
            f"{class_name:<12} "
            f"{percentage:6.2f}%"
        )

    print(
        f"\nIgnored pixels: "
        f"{ignored_pixels:,}"
    )


if __name__ == "__main__":
    main()