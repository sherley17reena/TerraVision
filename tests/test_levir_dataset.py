
import sys

from src.data.levir_dataset import LEVIRDataset


def test_dataset(root_dir):
    dataset = LEVIRDataset(
        root_dir=root_dir,
        image_size=256,
        augment=False,
    )

    print("Number of image pairs:", len(dataset))

    image_a, image_b, label = dataset[0]

    print("T1 image shape:", image_a.shape)
    print("T2 image shape:", image_b.shape)
    print("Label shape:", label.shape)
    print("Unique label values:", label.unique().tolist())

    assert image_a.shape == (3, 256, 256)
    assert image_b.shape == (3, 256, 256)
    assert label.shape == (1, 256, 256)

    assert set(label.unique().tolist()).issubset({0.0, 1.0})

    print("LEVIR-CD dataset test passed!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m tests.test_levir_dataset "
            "/path/to/extracted/dataset"
        )

    test_dataset(sys.argv[1])
