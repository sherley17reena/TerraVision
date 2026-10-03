from pathlib import Path
import shutil
import tempfile

import base64
import io

from src.analysis.visualization import (
    colorize_segmentation_mask,
    create_change_map,
)

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from src.analysis.pipeline import analyze_land_change
from src.models.inference import load_unet_model


app = FastAPI(
    title="TerraVision API",
    description=(
        "AI-powered satellite imagery analysis "
        "and land-change detection API."
    ),
    version="1.0.0",
)


# Load the trained model once when the API starts.
model, device = load_unet_model()


@app.get("/")
def root():
    return {
        "name": "TerraVision",
        "status": "running",
        "message": (
            "AI-powered land-change detection API"
        ),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_device": str(device),
    }


@app.post("/analyze")
async def analyze(
    image_t1: UploadFile = File(...),
    image_t2: UploadFile = File(...),
):
    """
    Analyze land-cover change between two images.
    """

    allowed_types = {
        "image/png",
        "image/jpeg",
    }

    if image_t1.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="T1 must be a PNG or JPEG image.",
        )

    if image_t2.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="T2 must be a PNG or JPEG image.",
        )

    temp_dir = Path(
        tempfile.mkdtemp()
    )

    try:
        t1_path = temp_dir / "image_t1.png"
        t2_path = temp_dir / "image_t2.png"

        with t1_path.open("wb") as buffer:
            shutil.copyfileobj(
                image_t1.file,
                buffer,
            )

        with t2_path.open("wb") as buffer:
            shutil.copyfileobj(
                image_t2.file,
                buffer,
            )

        results = analyze_land_change(
            t1_path,
            t2_path,
            model=model,
            device=device,
        )

        segmentation_t1 = colorize_segmentation_mask(
            results["mask_t1"]
        )

        segmentation_t2 = colorize_segmentation_mask(
            results["mask_t2"]
        )

        change_map = create_change_map(
            results["change_mask"]
        )


        def image_to_base64(image):
            buffer = io.BytesIO()

            image.save(
                buffer,
                format="PNG",
            )

            encoded = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

            return (
                "data:image/png;base64,"
                + encoded
            )


        segmentation_t1_base64 = image_to_base64(
            segmentation_t1
        )

        segmentation_t2_base64 = image_to_base64(
            segmentation_t2
        )

        change_map_base64 = image_to_base64(
            change_map
        )

        # Return only JSON-friendly statistics and
        # Base64-encoded visualization images.
        return {
            "status": "success",
            "change_percentage": (
                results["change_percentage"]
            ),
            "changed_pixels": (
                results["changed_pixels"]
            ),
            "total_pixels": (
                results["total_pixels"]
            ),
            "transitions": (
                results["transitions"]
            ),
            "land_cover": (
                results["land_cover"]
            ),

            "visualizations": {
                "segmentation_t1": segmentation_t1_base64,
                "segmentation_t2": segmentation_t2_base64,
                "change_map": change_map_base64,
            },
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    finally:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )