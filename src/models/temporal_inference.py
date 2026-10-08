
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.device import get_device
from src.models.temporal_unet import TemporalUNet


class TemporalChangeDetector:
    def __init__(
        self,
        model_path="outputs/models/temporal_unet_30epochs_best.pth",
        image_size=256,
        threshold=0.5,
    ):
        self.device = get_device()
        self.image_size = image_size
        self.threshold = threshold

        checkpoint_path = Path(model_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Temporal U-Net model not found: {checkpoint_path}"
            )

        self.model = TemporalUNet().to(self.device)

        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"Temporal U-Net loaded on {self.device}")

    def preprocess(self, image):
        image = image.convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )

        array = np.asarray(image, dtype=np.float32) / 255.0

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

        normalized = (array - mean) / std

        tensor = torch.from_numpy(
            normalized.transpose(2, 0, 1).copy()
        ).float().unsqueeze(0)

        return tensor.to(self.device)

    @torch.inference_mode()
    def predict(self, image_t1, image_t2):
        tensor_t1 = self.preprocess(image_t1)
        tensor_t2 = self.preprocess(image_t2)

        logits = self.model(tensor_t1, tensor_t2)

        probabilities = torch.sigmoid(logits)

        mask = (
            probabilities >= self.threshold
        ).squeeze().cpu().numpy().astype(np.uint8)

        changed_pixels = int(mask.sum())
        total_pixels = int(mask.size)

        change_percentage = (
            changed_pixels / total_pixels * 100
        )

        return {
            "change_mask": mask,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "change_percentage": round(change_percentage, 2),
        }
