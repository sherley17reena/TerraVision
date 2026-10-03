from pathlib import Path

from src.analysis.pipeline import analyze_land_change


def main():

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

    results = analyze_land_change(
        image_t1,
        image_t2,
    )

    print(
        f"\nT1 mask shape: "
        f"{results['mask_t1'].shape}"
    )

    print(
        f"T2 mask shape: "
        f"{results['mask_t2'].shape}"
    )

    print(
        f"Changed pixels: "
        f"{results['changed_pixels']:,}"
    )

    print(
        f"Change percentage: "
        f"{results['change_percentage']:.2f}%"
    )

    print("\nTop transitions:")

    for transition, data in list(
        results["transitions"].items()
    )[:5]:

        print(
            f"{transition}: "
            f"{data['percentage']:.2f}%"
        )

    print("\nLand-Cover Change:")

    for class_name, data in (
        results["land_cover"].items()
    ):

        print(
            f"{class_name}: "
            f"{data['t1_percentage']:.2f}% -> "
            f"{data['t2_percentage']:.2f}% "
            f"({data['net_change']:+.2f} pp)"
        )

    # Basic pipeline checks.
    assert (
        results["mask_t1"].shape
        == (256, 256)
    )

    assert (
        results["mask_t2"].shape
        == (256, 256)
    )

    assert (
        results["change_mask"].shape
        == (256, 256)
    )

    assert (
        0
        <= results["change_percentage"]
        <= 100
    )

    print(
        "\nTerraVision pipeline test passed!"
    )


if __name__ == "__main__":
    main()