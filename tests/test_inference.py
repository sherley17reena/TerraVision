from pathlib import Path

from src.models.inference import (
    load_unet_model,
    predict_mask,
)


def main():

    model, device = load_unet_model()

    print(
        f"Model loaded on: {device}"
    )

    image_path = Path(
        "data/raw/LoveDA/Val/Urban/images_png"
    )

    image_path = sorted(
        image_path.glob("*.png")
    )[0]

    print(
        f"Testing image: {image_path.name}"
    )

    mask, original_size = predict_mask(
        model,
        device,
        image_path,
    )

    print(
        f"Original image size: {original_size}"
    )

    print(
        f"Predicted mask shape: {mask.shape}"
    )

    print(
        f"Predicted classes: {sorted(set(mask.flatten()))}"
    )

    assert mask.shape == (256, 256)

    assert mask.min() >= 0

    assert mask.max() <= 6

    print(
        "\nInference test passed!"
    )


if __name__ == "__main__":
    main()