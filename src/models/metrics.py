import torch


def dice_score(
    predictions,
    targets,
    num_classes=7,
    ignore_index=255,
):
    """
    Calculate mean Dice score across semantic classes.

    Pixels with the ignore_index value are completely excluded
    from metric calculations.
    """

    predictions = torch.argmax(predictions, dim=1)

    # Only evaluate valid labelled pixels.
    valid_mask = targets != ignore_index

    dice_scores = []

    for class_id in range(num_classes):

        pred_class = (
            (predictions == class_id)
            & valid_mask
        )

        target_class = (
            (targets == class_id)
            & valid_mask
        )

        intersection = (
            pred_class & target_class
        ).sum().float()

        total = (
            pred_class.sum().float()
            + target_class.sum().float()
        )

        # Skip classes absent from both prediction and target.
        if total == 0:
            continue

        dice = (2.0 * intersection) / total

        dice_scores.append(dice)

    if not dice_scores:
        return 0.0

    return torch.stack(
        dice_scores
    ).mean().item()


def iou_score(
    predictions,
    targets,
    num_classes=7,
    ignore_index=255,
):
    """
    Calculate mean Intersection over Union across semantic classes.

    Pixels with the ignore_index value are completely excluded
    from metric calculations.
    """

    predictions = torch.argmax(predictions, dim=1)

    # Only evaluate valid labelled pixels.
    valid_mask = targets != ignore_index

    iou_scores = []

    for class_id in range(num_classes):

        pred_class = (
            (predictions == class_id)
            & valid_mask
        )

        target_class = (
            (targets == class_id)
            & valid_mask
        )

        intersection = (
            pred_class & target_class
        ).sum().float()

        union = (
            pred_class | target_class
        ).sum().float()

        if union == 0:
            continue

        iou = intersection / union

        iou_scores.append(iou)

    if not iou_scores:
        return 0.0

    return torch.stack(
        iou_scores
    ).mean().item()