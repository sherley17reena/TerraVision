import torch
import torch.nn as nn

from src.models.device import get_device


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


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Evaluate the model without updating its weights.
    """

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model(images)

            loss = criterion(outputs, masks)

            total_loss += loss.item()

    return total_loss / len(dataloader)


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
    criterion = nn.CrossEntropyLoss()

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

        val_loss = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
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