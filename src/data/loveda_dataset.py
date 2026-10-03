from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class LoveDADataset(Dataset):
    """
    PyTorch Dataset for LoveDA satellite images and segmentation masks.

    Raw LoveDA labels:
        0   = ignore / no-data
        1-7 = semantic classes

    Model labels:
        0-6 = semantic classes
        255 = ignore
    """

    def __init__(
        self,
        root_dir,
        split="Train",
        image_size=256,
        augment=False,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment

        self.samples = []

        # LoveDA contains Urban and Rural regions.
        for region in ["Urban", "Rural"]:
            image_dir = (
                self.root_dir
                / split
                / region
                / "images_png"
            )

            mask_dir = (
                self.root_dir
                / split
                / region
                / "masks_png"
            )

            # Pair every satellite image with its corresponding mask.
            for image_path in sorted(image_dir.glob("*.png")):
                mask_path = mask_dir / image_path.name

                if not mask_path.exists():
                    raise FileNotFoundError(
                        f"Mask not found for {image_path}"
                    )

                self.samples.append(
                    (image_path, mask_path)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        # Load image and segmentation mask.
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize image.
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=TF.InterpolationMode.BILINEAR,
        )

        # IMPORTANT:
        # Masks use nearest-neighbour interpolation.
        # Bilinear interpolation would create invalid class values.
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )

        # ---------------------------------------------------------
        # DATA AUGMENTATION
        # ---------------------------------------------------------
        # Only enabled for training data.
        #
        # The exact same transformation must be applied to both
        # the satellite image and segmentation mask.
        # ---------------------------------------------------------

        if self.augment:

            # Random horizontal flip.
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flip.
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # ---------------------------------------------------------
        # IMAGE PREPROCESSING
        # ---------------------------------------------------------

        # Convert:
        # PIL [H, W, C]
        #       ↓
        # Tensor [C, H, W]
        #
        # Pixel values also change from 0-255 to 0-1.
        image = TF.to_tensor(image)

        # Normalize RGB channels.
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # ---------------------------------------------------------
        # MASK PREPROCESSING
        # ---------------------------------------------------------

        mask = np.array(
            mask,
            dtype=np.int64,
        )

        # LoveDA value 0 represents pixels that should be ignored.
        ignore_pixels = mask == 0

        # Convert LoveDA labels:
        #
        # 1, 2, 3, 4, 5, 6, 7
        # ↓
        # 0, 1, 2, 3, 4, 5, 6
        #
        # These now correspond to the seven U-Net output channels.
        mask = mask - 1

        # CrossEntropyLoss(ignore_index=255)
        # will ignore these pixels during training.
        mask[ignore_pixels] = 255

        mask = torch.from_numpy(mask).long()

        return image, mask