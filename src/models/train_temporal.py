
from pathlib import Path
import csv
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.levir_dataset import LEVIRDataset
from src.models.device import get_device
from src.models.temporal_unet import TemporalUNet


def dice_loss(logits, targets, smooth=1.0):
    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (2 * intersection + smooth) / (denominator + smooth)
    return 1 - dice.mean()


def calculate_metrics(logits, targets):
    predictions = torch.sigmoid(logits) >= 0.5
    targets = targets >= 0.5

    tp = (predictions & targets).sum().item()
    fp = (predictions & ~targets).sum().item()
    fn = (~predictions & targets).sum().item()

    return tp, fp, fn


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for image_t1, image_t2, targets in loader:
        image_t1 = image_t1.to(device)
        image_t2 = image_t2.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(image_t1, image_t2)

            loss = (
                criterion(logits, targets)
                + dice_loss(logits, targets)
            )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()

        total_loss += loss.item() * image_t1.size(0)

        with torch.no_grad():
            tp, fp, fn = calculate_metrics(logits, targets)
            total_tp += tp
            total_fp += fp
            total_fn += fn

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * total_tp / max(
        2 * total_tp + total_fp + total_fn, 1
    )
    iou = total_tp / max(
        total_tp + total_fp + total_fn, 1
    )

    return {
        "loss": total_loss / len(loader.dataset),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def train_temporal_model(
    train_dir,
    val_dir,
    epochs=30,
    batch_size=2,
    learning_rate=1e-4,
    patience=7,
    resume=False,
):
    torch.manual_seed(42)
    random.seed(42)

    device = get_device()
    print(f"Training device: {device}")

    train_dataset = LEVIRDataset(
        train_dir, image_size=256, augment=True
    )
    val_dataset = LEVIRDataset(
        val_dir, image_size=256, augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = TemporalUNet().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = output_dir / "temporal_unet_best.pth"
    checkpoint_path = output_dir / "temporal_unet_checkpoint.pth"

    history_path = Path("outputs/temporal_training_history.csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "epoch", "train_loss", "val_loss",
        "precision", "recall", "f1", "iou",
        "learning_rate",
    ]

    start_epoch = 0
    best_f1 = -1.0
    no_improvement = 0
    min_delta = 1e-4

    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])

        start_epoch = checkpoint["epoch"]
        best_f1 = checkpoint["best_f1"]
        no_improvement = checkpoint["no_improvement"]

        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])

        if device.type == "mps" and checkpoint.get("mps_rng_state") is not None:
            torch.mps.set_rng_state(checkpoint["mps_rng_state"])

        print(f"Resuming from epoch {start_epoch}")

    else:
        # A fresh run must not accidentally overwrite the previous
        # best model before the user has chosen to replace it.
        if best_model_path.exists():
            print(
                "Warning: A fresh run will replace the existing "
                "best-model file when validation F1 improves."
            )

        with history_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()

    if resume and not history_path.exists():
        raise FileNotFoundError(
            f"Training history not found: {history_path}"
        )

    for epoch in range(start_epoch, epochs):
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_metrics = run_epoch(
            model, val_loader, criterion, None, device
        )

        scheduler.step(val_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1": val_metrics["f1"],
            "iou": val_metrics["iou"],
            "learning_rate": current_lr,
        }

        with history_path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writerow(row)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"LR: {current_lr:.6f}",
            flush=True,
        )

        if val_metrics["f1"] > best_f1 + min_delta:
            best_f1 = val_metrics["f1"]
            no_improvement = 0

            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model: {best_model_path}")
        else:
            no_improvement += 1

        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_f1": best_f1,
            "no_improvement": no_improvement,
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "mps_rng_state": (
                torch.mps.get_rng_state()
                if device.type == "mps"
                else None
            ),
        }

        torch.save(checkpoint, checkpoint_path)

        if no_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"No F1 improvement for {patience} epochs."
            )
            break

    print(
        f"Training completed. "
        f"Best validation F1: {best_f1:.4f}"
    )


if __name__ == "__main__":
    train_temporal_model(
        train_dir=str(Path.home() / "Downloads/train"),
        val_dir=str(Path.home() / "Downloads/val"),
        epochs=30,
        batch_size=2,
        resume=False,
    )
