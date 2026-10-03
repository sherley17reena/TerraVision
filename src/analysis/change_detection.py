import numpy as np


CLASS_NAMES = [
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agriculture",
]


def detect_changes(mask_t1, mask_t2):
    """
    Compare two land-cover segmentation masks.

    Parameters
    ----------
    mask_t1 : np.ndarray
        Segmentation mask from the earlier image.

    mask_t2 : np.ndarray
        Segmentation mask from the later image.

    Returns
    -------
    dict
        Change statistics and land-cover transitions.
    """

    if mask_t1.shape != mask_t2.shape:
        raise ValueError(
            "T1 and T2 masks must have the same dimensions."
        )

    # A pixel changed when its predicted land-cover
    # class differs between T1 and T2.
    change_mask = mask_t1 != mask_t2

    total_pixels = mask_t1.size
    changed_pixels = int(change_mask.sum())

    change_percentage = (
        changed_pixels / total_pixels
    ) * 100

    # ---------------------------------------------------------
    # LAND-COVER TRANSITIONS
    # ---------------------------------------------------------

    transitions = {}

    for from_class in range(len(CLASS_NAMES)):

        for to_class in range(len(CLASS_NAMES)):

            # We only care about actual changes.
            if from_class == to_class:
                continue

            transition_pixels = np.logical_and(
                mask_t1 == from_class,
                mask_t2 == to_class,
            ).sum()

            if transition_pixels == 0:
                continue

            transition_percentage = (
                transition_pixels / total_pixels
            ) * 100

            transition_name = (
                f"{CLASS_NAMES[from_class]}"
                f" -> "
                f"{CLASS_NAMES[to_class]}"
            )

            transitions[transition_name] = {
                "pixels": int(transition_pixels),
                "percentage": float(
                    transition_percentage
                ),
            }

    # Sort largest transitions first.
    transitions = dict(
        sorted(
            transitions.items(),
            key=lambda item: item[1]["pixels"],
            reverse=True,
        )
    )

    return {
        "total_pixels": int(total_pixels),
        "changed_pixels": changed_pixels,
        "change_percentage": float(
            change_percentage
        ),
        "change_mask": change_mask,
        "transitions": transitions,
    }

def calculate_land_cover_change(mask_t1, mask_t2):
    """
    Calculate land-cover percentages at T1 and T2
    and the net change for each class.
    """

    if mask_t1.shape != mask_t2.shape:
        raise ValueError(
            "T1 and T2 masks must have the same dimensions."
        )

    total_pixels = mask_t1.size

    land_cover_stats = {}

    for class_id, class_name in enumerate(CLASS_NAMES):

        t1_pixels = int(
            (mask_t1 == class_id).sum()
        )

        t2_pixels = int(
            (mask_t2 == class_id).sum()
        )

        t1_percentage = (
            t1_pixels / total_pixels
        ) * 100

        t2_percentage = (
            t2_pixels / total_pixels
        ) * 100

        # Percentage-point change.
        net_change = (
            t2_percentage - t1_percentage
        )

        land_cover_stats[class_name] = {
            "t1_pixels": t1_pixels,
            "t2_pixels": t2_pixels,
            "t1_percentage": float(
                t1_percentage
            ),
            "t2_percentage": float(
                t2_percentage
            ),
            "net_change": float(
                net_change
            ),
        }

    return land_cover_stats