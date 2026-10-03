from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


CLASS_NAMES = [
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agriculture",
]


def load_training_pixels(
    root_dir="data/raw/LoveDA",
    images_per_region=50,
    pixels_per_image=2000,
    random_seed=42,
):
    """
    Sample labelled pixels from LoveDA images.

    Features:
        R, G, B

    Target:
        LoveDA semantic class 0-6
    """

    root_dir = Path(root_dir)

    rng = np.random.default_rng(random_seed)

    all_features = []
    all_labels = []

    for region in ["Urban", "Rural"]:

        image_dir = (
            root_dir
            / "Train"
            / region
            / "images_png"
        )

        mask_dir = (
            root_dir
            / "Train"
            / region
            / "masks_png"
        )

        image_paths = sorted(
            image_dir.glob("*.png")
        )

        # Randomly choose images.
        selected_count = min(
            images_per_region,
            len(image_paths),
        )

        selected_indices = rng.choice(
            len(image_paths),
            size=selected_count,
            replace=False,
        )

        for index in selected_indices:

            image_path = image_paths[index]
            mask_path = mask_dir / image_path.name

            image = np.array(
                Image.open(image_path).convert("RGB"),
                dtype=np.float32,
            )

            mask = np.array(
                Image.open(mask_path).convert("L"),
                dtype=np.int64,
            )

            # Flatten image:
            # H × W × 3 -> pixels × 3
            pixels = image.reshape(-1, 3)
            labels = mask.reshape(-1)

            # Ignore LoveDA label 0.
            valid = labels != 0

            pixels = pixels[valid]
            labels = labels[valid]

            # LoveDA 1-7 -> model classes 0-6.
            labels = labels - 1

            sample_count = min(
                pixels_per_image,
                len(labels),
            )

            sample_indices = rng.choice(
                len(labels),
                size=sample_count,
                replace=False,
            )

            all_features.append(
                pixels[sample_indices]
            )

            all_labels.append(
                labels[sample_indices]
            )

    X = np.concatenate(
        all_features,
        axis=0,
    )

    y = np.concatenate(
        all_labels,
        axis=0,
    )

    return X, y


def main():

    print("Preparing Random Forest training data...")

    X, y = load_training_pixels()

    print(f"Training pixels: {len(y):,}")
    print(f"Feature shape: {X.shape}")

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X, y)

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions,
    )

    print("\nRandom Forest training complete.")
    print(f"Training accuracy: {accuracy:.4f}")

    Path(
        "outputs/models"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        "outputs/models/random_forest.joblib",
    )

    print(
        "Model saved to "
        "outputs/models/random_forest.joblib"
    )


if __name__ == "__main__":
    main()