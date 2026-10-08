
from pathlib import Path

from PIL import Image

from src.models.temporal_inference import TemporalChangeDetector


def main():
    test_dir = Path.home() / "Downloads/test"

    filenames = sorted(
        set(p.name for p in (test_dir / "A").glob("*.png"))
        & set(p.name for p in (test_dir / "B").glob("*.png"))
    )

    if not filenames:
        raise FileNotFoundError(
            "No matching LEVIR-CD test image pairs found."
        )

    filename = filenames[0]

    image_t1 = Image.open(test_dir / "A" / filename)
    image_t2 = Image.open(test_dir / "B" / filename)

    detector = TemporalChangeDetector()

    result = detector.predict(image_t1, image_t2)

    assert result["change_mask"].shape == (256, 256)
    assert result["total_pixels"] == 256 * 256
    assert 0 <= result["change_percentage"] <= 100

    print("\nTemporal inference test passed!")
    print(f"Image pair: {filename}")
    print(f"Changed pixels: {result['changed_pixels']}")
    print(f"Total pixels: {result['total_pixels']}")
    print(f"Predicted building change: {result['change_percentage']}%")


if __name__ == "__main__":
    main()
