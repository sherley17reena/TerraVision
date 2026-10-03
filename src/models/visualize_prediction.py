import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.loveda_dataset import LoveDADataset
from src.models.device import get_device
from src.models.unet import UNet
from src.data.utils import denormalize_image


def main():
    device = get_device()

    print(f"Using device: {device}")

    # Load validation dataset
    dataset = LoveDADataset(
        root_dir="data/raw/LoveDA",
        split="Val",
        image_size=256,
    )

    # Create the same model architecture used during training
    model = UNet(
        in_channels=3,
        num_classes=7,
    )

    # Load our trained weights
    checkpoint_path = "outputs/models/unet_experiment.pth"

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    model = model.to(device)
    model.eval()

    # Pick one validation image
    image, ground_truth = dataset[0]

    # Add batch dimension:
    # [3, 256, 256] -> [1, 3, 256, 256]
    input_image = image.unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(input_image)

        prediction = torch.argmax(
            output,
            dim=1
        ).squeeze(0)

    # Move data back to CPU for visualization
    image_display = (
        denormalize_image(image)
        .permute(1, 2, 0)
        .numpy()
    )
    ground_truth_display = ground_truth.numpy()
    prediction_display = prediction.cpu().numpy()

    # Hide ignored pixels in ground truth
    ground_truth_display = np.ma.masked_where(
        ground_truth_display == 255,
        ground_truth_display
    )

    # Plot
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 6)
    )

    axes[0].imshow(image_display)
    axes[0].set_title("Satellite Image")
    axes[0].axis("off")

    axes[1].imshow(
        ground_truth_display,
        cmap="tab10",
        vmin=0,
        vmax=6,
    )
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(
        prediction_display,
        cmap="tab10",
        vmin=0,
        vmax=6,
    )
    axes[2].set_title("U-Net Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()