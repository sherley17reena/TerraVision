
import numpy as np
from PIL import Image

from src.data.levir_dataset import LEVIRDataset


def test_dataset(tmp_path):
    """
    Test LEVIR-CD dataset loading using temporary sample images.
    No external dataset download is required.
    """

    # Create the LEVIR-CD folder structure.
    dataset_dir = tmp_path / "sample_levir"

    for folder in ("A", "B", "label"):
        (dataset_dir / folder).mkdir(parents=True)

    # Create two synthetic RGB satellite images.
    image_a = np.zeros((256, 256, 3), dtype=np.uint8)
    image_b = np.zeros((256, 256, 3), dtype=np.uint8)

    image_a[:, :] = [70, 110, 80]
    image_b[:, :] = [70, 110, 80]

    # Simulate a newly constructed building in T2.
    image_b[80:150, 90:160] = [200, 200, 200]

    # Binary ground-truth change mask.
    label = np.zeros((256, 256), dtype=np.uint8)
    label[80:150, 90:160] = 255

    # Save images in LEVIR-CD format.
    Image.fromarray(image_a).save(
        dataset_dir / "A" / "sample.png"
    )

    Image.fromarray(image_b).save(
        dataset_dir / "B" / "sample.png"
    )

    Image.fromarray(label).save(
        dataset_dir / "label" / "sample.png"
    )

    # Initialize the dataset.
    dataset = LEVIRDataset(
        root_dir=str(dataset_dir),
        image_size=256,
        augment=False,
    )

    assert len(dataset) == 1

    # Load the sample pair.
    tensor_a, tensor_b, tensor_label = dataset[0]

    # Validate tensor shapes.
    assert tensor_a.shape == (3, 256, 256)
    assert tensor_b.shape == (3, 256, 256)
    assert tensor_label.shape == (1, 256, 256)

    # Validate binary mask values.
    unique_values = set(tensor_label.unique().tolist())

    assert unique_values.issubset({0.0, 1.0})
    assert unique_values == {0.0, 1.0}

    # Validate the synthetic changed-building region.
    assert tensor_label[0, 100, 100].item() == 1.0
    assert tensor_label[0, 20, 20].item() == 0.0

    print("LEVIR-CD dataset test passed!")
