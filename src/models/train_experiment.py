import torch
from torch.utils.data import DataLoader, Subset

from src.data.loveda_dataset import LoveDADataset
from src.models.unet import UNet
from src.models.train import train_model


def main():

    # ---------------------------------------------------------
    # LOAD DATASETS
    # ---------------------------------------------------------

    train_dataset = LoveDADataset(
        root_dir="data/raw/LoveDA",
        split="Train",
        image_size=256,
        augment=True,
    )

    val_dataset = LoveDADataset(
        root_dir="data/raw/LoveDA",
        split="Val",
        image_size=256,
        augment=False,
    )

    # ---------------------------------------------------------
    # CREATE REPRODUCIBLE RANDOM SUBSETS
    # ---------------------------------------------------------

    # Using a fixed seed means we get the same random samples
    # every time we run the experiment.
    generator = torch.Generator().manual_seed(42)

    train_indices = torch.randperm(
        len(train_dataset),
        generator=generator,
    )[:1000].tolist()

    val_indices = torch.randperm(
        len(val_dataset),
        generator=generator,
    )[:300].tolist()

    train_subset = Subset(
        train_dataset,
        train_indices,
    )

    val_subset = Subset(
        val_dataset,
        val_indices,
    )

    # ---------------------------------------------------------
    # CREATE DATALOADERS
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_subset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")

    # ---------------------------------------------------------
    # CREATE U-NET
    # ---------------------------------------------------------

    model = UNet(
        in_channels=3,
        num_classes=7,
    )

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=8,
        learning_rate=1e-4,
        model_path="outputs/models/unet_experiment.pth",
    )


if __name__ == "__main__":
    main()