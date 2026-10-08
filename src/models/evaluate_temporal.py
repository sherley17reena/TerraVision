
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.levir_dataset import LEVIRDataset
from src.models.device import get_device
from src.models.temporal_unet import TemporalUNet
from src.models.train_temporal import calculate_metrics


def evaluate_model():
    device = get_device()
    print(f"Evaluation device: {device}")

    model_path = Path(
        "outputs/models/temporal_unet_30epochs_best.pth"
    )

    test_dir = Path.home() / "Downloads/test"

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    test_dataset = LEVIRDataset(
        root_dir=test_dir,
        image_size=256,
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    model = TemporalUNet().to(device)

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    with torch.inference_mode():
        for image_t1, image_t2, targets in test_loader:
            image_t1 = image_t1.to(device)
            image_t2 = image_t2.to(device)
            targets = targets.to(device)

            logits = model(image_t1, image_t2)

            tp, fp, fn = calculate_metrics(logits, targets)

            predictions = torch.sigmoid(logits) >= 0.5
            actual = targets >= 0.5

            tn = ((~predictions) & (~actual)).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = (2 * total_tp) / max(
        2 * total_tp + total_fp + total_fn, 1
    )
    iou = total_tp / max(
        total_tp + total_fp + total_fn, 1
    )

    total_pixels = total_tp + total_fp + total_fn + total_tn
    accuracy = (total_tp + total_tn) / max(total_pixels, 1)

    print("\nTerraVision — Temporal U-Net Test Evaluation")
    print("-" * 48)
    print(f"Test image pairs: {len(test_dataset)}")
    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")
    print(f"F1 Score:         {f1:.4f}")
    print(f"IoU:              {iou:.4f}")
    print(f"Pixel Accuracy:   {accuracy:.4f}")
    print("-" * 48)


if __name__ == "__main__":
    evaluate_model()

