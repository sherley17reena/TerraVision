from src.models.inference import (
    load_unet_model,
    predict_mask,
)

from src.analysis.change_detection import (
    detect_changes,
    calculate_land_cover_change,
)


def analyze_land_change(
    image_t1_path,
    image_t2_path,
    model=None,
    device=None,
):
    """
    Run the complete TerraVision land-change pipeline.

    Pipeline:
        Image T1
            -> U-Net
            -> segmentation mask T1

        Image T2
            -> U-Net
            -> segmentation mask T2

        Masks
            -> change detection
            -> land-cover statistics
    """

    # Load the U-Net only if it has not already
    # been supplied by the caller.
    if model is None or device is None:
        model, device = load_unet_model()

    # ---------------------------------------------------------
    # SEGMENT T1
    # ---------------------------------------------------------

    mask_t1, size_t1 = predict_mask(
        model,
        device,
        image_t1_path,
    )

    # ---------------------------------------------------------
    # SEGMENT T2
    # ---------------------------------------------------------

    mask_t2, size_t2 = predict_mask(
        model,
        device,
        image_t2_path,
    )

    # ---------------------------------------------------------
    # CHANGE DETECTION
    # ---------------------------------------------------------

    change_results = detect_changes(
        mask_t1,
        mask_t2,
    )

    # ---------------------------------------------------------
    # LAND-COVER STATISTICS
    # ---------------------------------------------------------

    land_cover = calculate_land_cover_change(
        mask_t1,
        mask_t2,
    )

    return {
        "image_t1_size": size_t1,
        "image_t2_size": size_t2,
        "mask_t1": mask_t1,
        "mask_t2": mask_t2,
        "change_mask": change_results["change_mask"],
        "total_pixels": change_results["total_pixels"],
        "changed_pixels": change_results["changed_pixels"],
        "change_percentage": change_results["change_percentage"],
        "transitions": change_results["transitions"],
        "land_cover": land_cover,
    }