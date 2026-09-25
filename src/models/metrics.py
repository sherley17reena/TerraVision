import torch


def dice_score(predictions, targets, num_classes=7):
    """
    Calculate mean Dice score across all classes.
    """

    predictions = torch.argmax(predictions, dim=1)

    dice_scores = []

    for class_id in range(num_classes):
        pred_class = predictions == class_id
        target_class = targets == class_id

        intersection = (pred_class & target_class).sum().float()

        total = pred_class.sum().float() + target_class.sum().float()

        if total == 0:
            continue

        dice = (2 * intersection) / total
        dice_scores.append(dice)

    if not dice_scores:
        return 0.0

    return torch.stack(dice_scores).mean().item()


def iou_score(predictions, targets, num_classes=7):
    """
    Calculate mean Intersection over Union (IoU).
    """

    predictions = torch.argmax(predictions, dim=1)

    iou_scores = []

    for class_id in range(num_classes):
        pred_class = predictions == class_id
        target_class = targets == class_id

        intersection = (pred_class & target_class).sum().float()
        union = (pred_class | target_class).sum().float()

        if union == 0:
            continue

        iou = intersection / union
        iou_scores.append(iou)

    if not iou_scores:
        return 0.0

    return torch.stack(iou_scores).mean().item()