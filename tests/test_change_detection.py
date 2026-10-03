import numpy as np

from src.analysis.change_detection import (
    detect_changes,
    calculate_land_cover_change,
)


def main():

    # Earlier land-cover map.
    mask_t1 = np.array([
        [6, 6, 6, 0],
        [6, 6, 0, 0],
        [5, 5, 3, 3],
        [5, 5, 3, 3],
    ])

    # Later land-cover map.
    mask_t2 = np.array([
        [6, 1, 1, 0],
        [6, 1, 0, 0],
        [5, 5, 3, 1],
        [5, 5, 3, 1],
    ])

    # ---------------------------------------------------------
    # CHANGE DETECTION
    # ---------------------------------------------------------

    results = detect_changes(
        mask_t1,
        mask_t2,
    )

    print(
        "Total pixels:",
        results["total_pixels"],
    )

    print(
        "Changed pixels:",
        results["changed_pixels"],
    )

    print(
        "Change percentage:",
        f"{results['change_percentage']:.2f}%",
    )

    print("\nTransitions:")

    for transition, data in results["transitions"].items():

        print(
            f"{transition}: "
            f"{data['pixels']} pixels "
            f"({data['percentage']:.2f}%)"
        )

    # ---------------------------------------------------------
    # LAND-COVER STATISTICS
    # ---------------------------------------------------------

    land_cover = calculate_land_cover_change(
        mask_t1,
        mask_t2,
    )

    print("\nLand-Cover Change:")

    for class_name, data in land_cover.items():

        print(
            f"{class_name}: "
            f"{data['t1_percentage']:.2f}% -> "
            f"{data['t2_percentage']:.2f}% "
            f"(change: {data['net_change']:+.2f} pp)"
        )

    # ---------------------------------------------------------
    # TESTS
    # ---------------------------------------------------------

    assert results["total_pixels"] == 16

    assert results["changed_pixels"] == 5

    assert abs(
        results["change_percentage"] - 31.25
    ) < 1e-6

    assert (
        results["transitions"]
        ["Agriculture -> Building"]
        ["pixels"]
        == 3
    )

    assert (
        results["transitions"]
        ["Water -> Building"]
        ["pixels"]
        == 2
    )

    print(
        "\nChange detection test passed!"
    )


if __name__ == "__main__":
    main()