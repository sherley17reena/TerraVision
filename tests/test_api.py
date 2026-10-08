
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)

TEST_DIR = Path.home() / "Downloads/test"
IMAGE_T1 = TEST_DIR / "A" / "test_10.png"
IMAGE_T2 = TEST_DIR / "B" / "test_10.png"


def upload_pair(endpoint):
    with IMAGE_T1.open("rb") as before, IMAGE_T2.open("rb") as after:
        return client.post(
            endpoint,
            files={
                "image_t1": ("before.png", before, "image/png"),
                "image_t2": ("after.png", after, "image/png"),
            },
        )


def test_building_change_detection():
    response = upload_pair("/analyze/buildings")

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["model"] == "Temporal U-Net"
    assert data["image_size"] == [256, 256]
    assert data["total_pixels"] == 65536
    assert 0 <= data["change_percentage"] <= 100
    assert isinstance(data["change_mask"], str)
    assert len(data["change_mask"]) > 0


def test_land_cover_analysis():
    response = upload_pair("/analyze")

    assert response.status_code == 200, response.text

    data = response.json()

    assert 0 <= data["change_percentage"] <= 100
    assert data["total_pixels"] == 65536
    assert "land_cover" in data
    assert "transitions" in data
    assert "visualizations" in data


def test_missing_images():
    response = client.post("/analyze/buildings")

    assert response.status_code == 422


def test_invalid_image():
    response = client.post(
        "/analyze/buildings",
        files={
            "image_t1": (
                "invalid.txt",
                b"not an image",
                "text/plain",
            ),
            "image_t2": (
                "invalid.txt",
                b"not an image",
                "text/plain",
            ),
        },
    )

    assert response.status_code == 400


def test_mismatched_dimensions():
    from io import BytesIO
    from PIL import Image

    before = Image.new("RGB", (256, 256))
    after = Image.new("RGB", (128, 128))

    before_buffer = BytesIO()
    after_buffer = BytesIO()

    before.save(before_buffer, format="PNG")
    after.save(after_buffer, format="PNG")

    response = client.post(
        "/analyze/buildings",
        files={
            "image_t1": (
                "before.png",
                before_buffer.getvalue(),
                "image/png",
            ),
            "image_t2": (
                "after.png",
                after_buffer.getvalue(),
                "image/png",
            ),
        },
    )

    assert response.status_code == 400
