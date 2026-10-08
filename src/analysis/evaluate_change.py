
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
)

from src.analysis.pipeline import analyze_land_change


def evaluate_change_detection():
    data_dir = Path("data/sample_pairs/levir_cd")

    before_path = data_dir / "before.png"
    after_path = data_dir / "after.png"
    label_path = data_dir / "ground_truth.png"

    for path in [before_path, after_path, label_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    # Run the trained TerraVision model.
    results = analyze_land_change(before_path, after_path)

    # LoveDA Building class ID = 1.
    BUILDING_CLASS = 1

    # Identify pixels classified as buildings at each time.
    building_t1 = results["mask_t1"] == BUILDING_CLASS
    building_t2 = results["mask_t2"] == BUILDING_CLASS

    # Detect pixels entering or leaving the Building class.
    predicted_mask = np.logical_xor(
        building_t1,
        building_t2,
    ).astype(np.uint8)

    # LEVIR-CD label: pixels marked as building changes.
    ground_truth = Image.open(label_path).convert("L")
    ground_truth = ground_truth.resize(
        (predicted_mask.shape[1], predicted_mask.shape[0]),
        resample=Image.Resampling.NEAREST,
    )
    true_mask = (np.array(ground_truth) > 127).astype(np.uint8)

    y_true = true_mask.flatten()
    y_pred = predicted_mask.flatten()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    iou = jaccard_score(y_true, y_pred, zero_division=0)

    print("\nTerraVision — Building-Change Diagnostic Evaluation")
    print("-" * 48)
    print(f"Ground-truth building-change pixels: {int(y_true.sum()):,}")
    print(f"Predicted building-class change :    {int(y_pred.sum()):,}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"IoU:       {iou:.4f}")
    print("-" * 48)
    print(
        "Note: This is a building-specific diagnostic using "
        "a LoveDA-trained segmentation model on LEVIR-CD imagery. "
        "It is not a validated LEVIR-CD benchmark."
    )


if __name__ == "__main__":
    evaluate_change_detection()
