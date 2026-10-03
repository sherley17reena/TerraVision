import torch

from src.data.loveda_dataset import LoveDADataset


def test_loveda_dataset():

    dataset = LoveDADataset(
        root_dir="data/raw/LoveDA",
        split="Train",
        image_size=256,
    )

    print("Dataset size:", len(dataset))

    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Image range:", image.min().item(), image.max().item())

    print("Mask shape:", mask.shape)
    print("Mask dtype:", mask.dtype)
    print("Mask values:", torch.unique(mask).tolist())

    assert len(dataset) == 2522
    assert image.shape == (3, 256, 256)
    assert mask.shape == (256, 256)

    valid_values = set(range(7)) | {255}
    actual_values = set(torch.unique(mask).tolist())

    assert actual_values.issubset(valid_values)


if __name__ == "__main__":
    test_loveda_dataset()
    print("LoveDA dataset test passed!")