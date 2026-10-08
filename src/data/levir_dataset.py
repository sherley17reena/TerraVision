
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class LEVIRDataset(Dataset):
    def __init__(self, root_dir, image_size=256, augment=False):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.augment = augment

        a_dir = self.root_dir / "A"
        b_dir = self.root_dir / "B"
        label_dir = self.root_dir / "label"

        for directory in [a_dir, b_dir, label_dir]:
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Required directory not found: {directory}"
                )

        self.samples = []

        for image_a in sorted(a_dir.glob("*.png")):
            image_b = b_dir / image_a.name
            label = label_dir / image_a.name

            if not image_b.exists() or not label.exists():
                raise FileNotFoundError(
                    f"Missing matched image or label: {image_a.name}"
                )

            self.samples.append((image_a, image_b, label))

        if not self.samples:
            raise ValueError("No matching PNG image pairs found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path_a, path_b, path_label = self.samples[index]

        image_a = Image.open(path_a).convert("RGB")
        image_b = Image.open(path_b).convert("RGB")
        label = Image.open(path_label).convert("L")

        if image_a.size != image_b.size or image_a.size != label.size:
            raise ValueError(
                f"Image dimensions do not match: {path_a.name}"
            )

        image_a = TF.resize(
            image_a,
            [self.image_size, self.image_size],
            interpolation=TF.InterpolationMode.BILINEAR,
        )

        image_b = TF.resize(
            image_b,
            [self.image_size, self.image_size],
            interpolation=TF.InterpolationMode.BILINEAR,
        )

        label = TF.resize(
            label,
            [self.image_size, self.image_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )

        # Apply identical transformations to both images and the label.
        if self.augment:
            if random.random() < 0.5:
                image_a = TF.hflip(image_a)
                image_b = TF.hflip(image_b)
                label = TF.hflip(label)

            if random.random() < 0.5:
                image_a = TF.vflip(image_a)
                image_b = TF.vflip(image_b)
                label = TF.vflip(label)

        image_a = TF.to_tensor(image_a)
        image_b = TF.to_tensor(image_b)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        image_a = TF.normalize(image_a, mean=mean, std=std)
        image_b = TF.normalize(image_b, mean=mean, std=std)

        label = np.asarray(label, dtype=np.uint8)
        label = (label > 127).astype(np.float32)
        label = torch.from_numpy(label).unsqueeze(0)

        return image_a, image_b, label
