
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.models.device import get_device
from src.models.temporal_unet import TemporalUNet


def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((256, 256))

    array = np.asarray(image, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    normalized = (array - mean) / std

    tensor = torch.from_numpy(
        normalized.transpose(2, 0, 1)
    ).float().unsqueeze(0)

    return array, tensor


def visualize():
    device = get_device()

    test_dir = Path.home() / "Downloads/test"
    model_path = Path(
        "outputs/models/temporal_unet_30epochs_best.pth"
    )

    filenames = sorted(
        set(p.name for p in (test_dir / "A").glob("*.png"))
        & set(p.name for p in (test_dir / "B").glob("*.png"))
        & set(p.name for p in (test_dir / "label").glob("*.png"))
    )

    if not filenames:
        raise FileNotFoundError(
            "No matching LEVIR-CD test image pairs found."
        )

    filename = filenames[0]

    before, before_tensor = load_image(test_dir / "A" / filename)
    after, after_tensor = load_image(test_dir / "B" / filename)

    ground_truth = Image.open(
        test_dir / "label" / filename
    ).convert("L")

    ground_truth = ground_truth.resize(
        (256, 256),
        resample=Image.Resampling.NEAREST
    )

    ground_truth = np.asarray(ground_truth) > 127

    model = TemporalUNet().to(device)
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

    with torch.inference_mode():
        logits = model(
            before_tensor.to(device),
            after_tensor.to(device)
        )

        prediction = (
            torch.sigmoid(logits) >= 0.5
        ).squeeze().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    panels = [
        (before, "Before (T1)", None),
        (after, "After (T2)", None),
        (ground_truth, "Ground Truth Changes", "gray"),
        (prediction, "Temporal U-Net Prediction", "gray"),
    ]

    for ax, (image, title, cmap) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap, vmin=0 if cmap else None,
                  vmax=1 if cmap else None)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    fig.suptitle(
        f"TerraVision — Building Change Detection\n{filename}",
        fontsize=15
    )

    plt.tight_layout()

    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "temporal_comparison.png"

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    visualize()
