import torch

from src.models.metrics import dice_score, iou_score


def test_perfect_prediction():

    targets = torch.tensor([
        [
            [0, 0],
            [1, 1]
        ]
    ])

    predictions = torch.zeros(
        1,
        7,
        2,
        2
    )

    predictions[0, 0, 0, 0] = 10
    predictions[0, 0, 0, 1] = 10

    predictions[0, 1, 1, 0] = 10
    predictions[0, 1, 1, 1] = 10

    dice = dice_score(
        predictions,
        targets,
        num_classes=7
    )

    iou = iou_score(
        predictions,
        targets,
        num_classes=7
    )

    assert abs(dice - 1.0) < 1e-6
    assert abs(iou - 1.0) < 1e-6


if __name__ == "__main__":
    test_perfect_prediction()
    print("Metrics test passed!")