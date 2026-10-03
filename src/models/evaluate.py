import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.loveda_dataset import LoveDADataset
from src.models.unet import UNet
from src.models.device import get_device
from src.models.train import get_class_weights


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

    # ---------------------------------------------------------
    # DEVICE
    # ---------------------------------------------------------

    device = get_device()
    print(f"Evaluating on: {device}")

    # ---------------------------------------------------------
    # VALIDATION DATASET
    # ---------------------------------------------------------

    val_dataset = LoveDADataset(
        root_dir="data/raw/LoveDA",
        split="Val",
        image_size=256,
        augment=False,
    )

    # Same reproducible validation subset used during training.
    generator = torch.Generator().manual_seed(42)

    val_indices = torch.randperm(
        len(val_dataset),
        generator=generator,
    )[:300].tolist()

    val_subset = Subset(
        val_dataset,
        val_indices,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    print(f"Validation samples: {len(val_subset)}")

    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------

    model = UNet(
        in_channels=3,
        num_classes=7,
    ).to(device)

    model.load_state_dict(
        torch.load(
            "outputs/models/unet_experiment.pth",
            map_location=device,
        )
    )

    model.eval()

    print("Model loaded successfully.")

    # ---------------------------------------------------------
    # LOSS
    # ---------------------------------------------------------

    class_weights = get_class_weights(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=255,
    )

    # ---------------------------------------------------------
    # METRIC COUNTERS
    # ---------------------------------------------------------

    num_classes = 7

    intersections = torch.zeros(
        num_classes,
        dtype=torch.float64,
    )

    prediction_totals = torch.zeros(
        num_classes,
        dtype=torch.float64,
    )

    target_totals = torch.zeros(
        num_classes,
        dtype=torch.float64,
    )

    unions = torch.zeros(
        num_classes,
        dtype=torch.float64,
    )

    total_loss = 0.0

    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model(images)

            loss = criterion(outputs, masks)
            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            # Completely exclude ignored pixels.
            valid_mask = masks != 255

            for class_id in range(num_classes):

                pred_class = (
                    (predictions == class_id)
                    & valid_mask
                )

                target_class = (
                    (masks == class_id)
                    & valid_mask
                )

                intersection = (
                    pred_class & target_class
                ).sum()

                union = (
                    pred_class | target_class
                ).sum()

                intersections[class_id] += (
                    intersection.cpu().item()
                )

                unions[class_id] += (
                    union.cpu().item()
                )

                prediction_totals[class_id] += (
                    pred_class.sum().cpu().item()
                )

                target_totals[class_id] += (
                    target_class.sum().cpu().item()
                )

    # ---------------------------------------------------------
    # CALCULATE RESULTS
    # ---------------------------------------------------------

    average_loss = total_loss / len(val_loader)

    class_ious = []
    class_dice = []

    print("\nPer-Class Evaluation")
    print("-" * 50)
    print(
        f"{'Class':<15}"
        f"{'IoU':>12}"
        f"{'Dice':>12}"
    )
    print("-" * 50)

    for class_id, class_name in enumerate(CLASS_NAMES):

        intersection = intersections[class_id]
        union = unions[class_id]

        pred_total = prediction_totals[class_id]
        target_total = target_totals[class_id]

        if union > 0:
            iou = intersection / union
        else:
            iou = torch.tensor(0.0)

        denominator = pred_total + target_total

        if denominator > 0:
            dice = (
                2.0 * intersection
            ) / denominator
        else:
            dice = torch.tensor(0.0)

        iou_value = float(iou)
        dice_value = float(dice)

        class_ious.append(iou_value)
        class_dice.append(dice_value)

        print(
            f"{class_name:<15}"
            f"{iou_value:>12.4f}"
            f"{dice_value:>12.4f}"
        )

    mean_iou = sum(class_ious) / len(class_ious)
    mean_dice = sum(class_dice) / len(class_dice)

    # ---------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------

    print("-" * 50)

    print("\nOverall Evaluation")
    print("-" * 50)

    print(
        f"Validation Loss: {average_loss:.4f}"
    )

    print(
        f"Mean IoU:        {mean_iou:.4f}"
    )

    print(
        f"Mean Dice:       {mean_dice:.4f}"
    )


if __name__ == "__main__":
    main()