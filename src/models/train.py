import torch
import torch.nn as nn

from src.models.device import get_device

from src.models.metrics import dice_score, iou_score


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one complete epoch.
    """

    model.train()

    total_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device).long()

        # Clear gradients from previous iteration
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, masks)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in dataloader:

            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model(images)

            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # Calculate metrics while completely ignoring
            # pixels labelled 255.
            total_dice += dice_score(
                outputs,
                masks,
                num_classes=7,
                ignore_index=255,
            )

            total_iou += iou_score(
                outputs,
                masks,
                num_classes=7,
                ignore_index=255,
            )

    num_batches = len(dataloader)

    average_loss = total_loss / num_batches
    average_dice = total_dice / num_batches
    average_iou = total_iou / num_batches

    return (
        average_loss,
        average_dice,
        average_iou,
    )

def get_class_weights(device):
    """
    Class weights derived from LoveDA training-set pixel frequencies.
    Uses square-root inverse frequency to reduce class imbalance
    without excessively weighting rare classes.
    """

    frequencies = torch.tensor(
        [
            0.3579,  # Background
            0.1106,  # Building
            0.0527,  # Road
            0.0639,  # Water
            0.0522,  # Barren
            0.1607,  # Forest
            0.2020,  # Agriculture
        ],
        dtype=torch.float32,
        device=device,
    )

    weights = 1.0 / torch.sqrt(frequencies)

    # Normalize so average weight = 1
    weights = weights / weights.mean()

    return weights

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=10,
    learning_rate=1e-4,
    model_path="outputs/models/unet_best.pth",
):
    """
    Complete TerraVision U-Net training pipeline.
    """

    device = get_device()

    print(f"Training on: {device}")

    model = model.to(device)

    # Suitable loss for multi-class segmentation
    class_weights = get_class_weights(device)

    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=255,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss, val_dice, val_iou = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Dice: {val_dice:.4f} | "
            f"IoU: {val_iou:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                model_path
            )

            print(f"Best model saved to {model_path}")

    return model