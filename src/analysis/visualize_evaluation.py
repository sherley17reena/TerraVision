
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.analysis.pipeline import analyze_land_change


def visualize_evaluation():
    data_dir = Path("data/sample_pairs/levir_cd")
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    before_path = data_dir / "before.png"
    after_path = data_dir / "after.png"
    label_path = data_dir / "ground_truth.png"

    for path in (before_path, after_path, label_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    # Run the existing U-Net inference pipeline.
    results = analyze_land_change(before_path, after_path)

    # Building class in the LoveDA-trained model.
    building_class = 1

    building_t1 = results["mask_t1"] == building_class
    building_t2 = results["mask_t2"] == building_class

    predicted_change = np.logical_xor(
        building_t1, building_t2
    )

    # Resize the reference mask using nearest-neighbor interpolation.
    height, width = predicted_change.shape

    ground_truth = Image.open(label_path).convert("L")
    ground_truth = ground_truth.resize(
        (width, height),
        resample=Image.Resampling.NEAREST,
    )
    ground_truth = np.asarray(ground_truth) > 127

    # Display the original images at the evaluation resolution.
    before = Image.open(before_path).convert("RGB").resize(
        (width, height)
    )
    after = Image.open(after_path).convert("RGB").resize(
        (width, height)
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    axes[0, 0].imshow(before)
    axes[0, 0].set_title("Earlier Satellite Image (T1)")

    axes[0, 1].imshow(after)
    axes[0, 1].set_title("Later Satellite Image (T2)")

    axes[1, 0].imshow(predicted_change, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Predicted Building-Class Changes")

    axes[1, 1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("LEVIR-CD Ground Truth (Building Changes)")

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        "TerraVision — Building-Change Diagnostic Comparison",
        fontsize=15,
    )
    plt.tight_layout()

    output_path = output_dir / "building_change_comparison.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Comparison saved to: {output_path}")


if __name__ == "__main__":
    visualize_evaluation()
