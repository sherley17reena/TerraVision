from pathlib import Path

import joblib
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

    print("Loading Random Forest...")

    model = joblib.load(
        "outputs/models/random_forest.joblib"
    )

    print("Model loaded successfully.")

    root_dir = Path("data/raw/LoveDA")

    rng = np.random.default_rng(42)

    num_classes = 7

    intersections = np.zeros(
        num_classes,
        dtype=np.float64,
    )

    unions = np.zeros(
        num_classes,
        dtype=np.float64,
    )

    prediction_totals = np.zeros(
        num_classes,
        dtype=np.float64,
    )

    target_totals = np.zeros(
        num_classes,
        dtype=np.float64,
    )

    # ---------------------------------------------------------
    # COLLECT VALIDATION IMAGES
    # ---------------------------------------------------------

    image_pairs = []

    for region in ["Urban", "Rural"]:

        image_dir = (
            root_dir
            / "Val"
            / region
            / "images_png"
        )

        mask_dir = (
            root_dir
            / "Val"
            / region
            / "masks_png"
        )

        for image_path in sorted(
            image_dir.glob("*.png")
        ):

            mask_path = (
                mask_dir
                / image_path.name
            )

            if mask_path.exists():

                image_pairs.append(
                    (
                        image_path,
                        mask_path,
                    )
                )

    # ---------------------------------------------------------
    # RANDOM VALIDATION SUBSET
    # ---------------------------------------------------------

    sample_count = min(
        300,
        len(image_pairs),
    )

    selected_indices = rng.choice(
        len(image_pairs),
        size=sample_count,
        replace=False,
    )

    print(
        f"Validation images: {sample_count}"
    )

    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------

    for number, index in enumerate(
        selected_indices,
        start=1,
    ):

        image_path, mask_path = (
            image_pairs[index]
        )

        image = np.array(
            Image.open(
                image_path
            ).convert("RGB"),
            dtype=np.float32,
        )

        mask = np.array(
            Image.open(
                mask_path
            ).convert("L"),
            dtype=np.int64,
        )

        # Flatten image.
        pixels = image.reshape(-1, 3)

        labels = mask.reshape(-1)

        # Remove ignored pixels.
        valid = labels != 0

        pixels = pixels[valid]
        labels = labels[valid]

        # LoveDA labels:
        # 1-7 -> 0-6
        labels = labels - 1

        # Predict in chunks to avoid unnecessary
        # memory usage on large 1024x1024 images.
        chunk_size = 100000

        predictions = []

        for start in range(
            0,
            len(pixels),
            chunk_size,
        ):

            end = start + chunk_size

            chunk_predictions = (
                model.predict(
                    pixels[start:end]
                )
            )

            predictions.append(
                chunk_predictions
            )

        predictions = np.concatenate(
            predictions
        )

        # -----------------------------------------------------
        # UPDATE CLASS METRICS
        # -----------------------------------------------------

        for class_id in range(
            num_classes
        ):

            pred_class = (
                predictions == class_id
            )

            target_class = (
                labels == class_id
            )

            intersection = np.logical_and(
                pred_class,
                target_class,
            ).sum()

            union = np.logical_or(
                pred_class,
                target_class,
            ).sum()

            intersections[class_id] += (
                intersection
            )

            unions[class_id] += union

            prediction_totals[class_id] += (
                pred_class.sum()
            )

            target_totals[class_id] += (
                target_class.sum()
            )

        if number % 25 == 0:

            print(
                f"Processed "
                f"{number}/{sample_count} images"
            )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    class_ious = []
    class_dice = []

    print("\nRandom Forest Per-Class Evaluation")
    print("-" * 50)

    print(
        f"{'Class':<15}"
        f"{'IoU':>12}"
        f"{'Dice':>12}"
    )

    print("-" * 50)

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        if unions[class_id] > 0:

            iou = (
                intersections[class_id]
                / unions[class_id]
            )

        else:

            iou = 0.0

        denominator = (
            prediction_totals[class_id]
            + target_totals[class_id]
        )

        if denominator > 0:

            dice = (
                2
                * intersections[class_id]
                / denominator
            )

        else:

            dice = 0.0

        class_ious.append(iou)
        class_dice.append(dice)

        print(
            f"{class_name:<15}"
            f"{iou:>12.4f}"
            f"{dice:>12.4f}"
        )

    mean_iou = np.mean(
        class_ious
    )

    mean_dice = np.mean(
        class_dice
    )

    print("-" * 50)

    print("\nOverall Evaluation")
    print("-" * 50)

    print(
        f"Mean IoU:  {mean_iou:.4f}"
    )

    print(
        f"Mean Dice: {mean_dice:.4f}"
    )


if __name__ == "__main__":
    main()