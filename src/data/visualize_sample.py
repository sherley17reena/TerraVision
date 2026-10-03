import matplotlib.pyplot as plt
import numpy as np

from src.data.loveda_dataset import LoveDADataset
from src.data.utils import denormalize_image

dataset = LoveDADataset(
    root_dir="data/raw/LoveDA",
    split="Train",
    image_size=256,
)

image, mask = dataset[0]

# PyTorch image format: [C, H, W]
# Matplotlib format: [H, W, C]
image_display = (
    denormalize_image(image)
    .permute(1, 2, 0)
    .numpy()
)

mask_display = mask.numpy()

# Hide ignore pixels for visualization
mask_display = np.ma.masked_where(
    mask_display == 255,
    mask_display
)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image_display)
axes[0].set_title("LoveDA Satellite Image")
axes[0].axis("off")

mask_plot = axes[1].imshow(
    mask_display,
    cmap="tab10",
    vmin=0,
    vmax=6
)

axes[1].set_title("Ground-Truth Segmentation Mask")
axes[1].axis("off")

fig.colorbar(
    mask_plot,
    ax=axes[1],
    ticks=range(7),
    fraction=0.046,
    pad=0.04
)

plt.tight_layout()
plt.show()