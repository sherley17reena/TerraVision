from pathlib import Path

from src.analysis.pipeline import analyze_land_change
from src.analysis.visualization import (
    colorize_segmentation_mask,
    create_change_map,
)


def main():

    # ---------------------------------------------------------
    # INPUT IMAGES
    # ---------------------------------------------------------

    image_dir = Path(
        "data/raw/LoveDA/Val/Urban/images_png"
    )

    images = sorted(
        image_dir.glob("*.png")
    )

    image_t1 = images[0]
    image_t2 = images[1]

    print(f"T1 image: {image_t1.name}")
    print(f"T2 image: {image_t2.name}")

    # ---------------------------------------------------------
    # RUN TERRAVISION PIPELINE
    # ---------------------------------------------------------

    results = analyze_land_change(
        image_t1,
        image_t2,
    )

    # ---------------------------------------------------------
    # CREATE VISUAL MAPS
    # ---------------------------------------------------------

    segmentation_t1 = colorize_segmentation_mask(
        results["mask_t1"]
    )

    segmentation_t2 = colorize_segmentation_mask(
        results["mask_t2"]
    )

    change_map = create_change_map(
        results["change_mask"]
    )

    # ---------------------------------------------------------
    # SAVE OUTPUTS
    # ---------------------------------------------------------

    output_dir = Path(
        "outputs/figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    t1_output = (
        output_dir / "segmentation_t1.png"
    )

    t2_output = (
        output_dir / "segmentation_t2.png"
    )

    change_output = (
        output_dir / "change_map.png"
    )

    segmentation_t1.save(t1_output)
    segmentation_t2.save(t2_output)
    change_map.save(change_output)

    print("\nSaved:")
    print(t1_output)
    print(t2_output)
    print(change_output)

    # ---------------------------------------------------------
    # TESTS
    # ---------------------------------------------------------

    assert segmentation_t1.size == (256, 256)
    assert segmentation_t2.size == (256, 256)
    assert change_map.size == (256, 256)

    assert t1_output.exists()
    assert t2_output.exists()
    assert change_output.exists()

    print(
        "\nVisualization test passed!"
    )


if __name__ == "__main__":
    main()