from pathlib import Path
import base64
import io
import shutil
import tempfile

import numpy as np
from PIL import Image

from fastapi.middleware.cors import CORSMiddleware
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from src.analysis.pipeline import analyze_land_change
from src.analysis.visualization import (
    colorize_segmentation_mask,
    create_change_map,
)
from src.models.inference import load_unet_model


from src.models.temporal_inference import TemporalChangeDetector


# ---------------------------------------------------------
# FASTAPI APPLICATION
# ---------------------------------------------------------

app = FastAPI(
    title="TerraVision API",
    description=(
        "AI-powered satellite imagery analysis "
        "and land-change detection API."
    ),
    version="1.0.0",
)
# Allow the React frontend to communicate with FastAPI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

# Load the trained U-Net once when the API starts.
model, device = load_unet_model()


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def image_to_base64(image):
    """
    Convert a PIL image into a Base64 PNG data URL.
    """

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


# ---------------------------------------------------------
# ROOT
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "TerraVision",
        "status": "running",
        "message": (
            "AI-powered land-change detection API"
        ),
    }


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_device": str(device),
    }


# ---------------------------------------------------------
# LAND-CHANGE ANALYSIS
# ---------------------------------------------------------

@app.post("/analyze")
async def analyze(
    image_t1: UploadFile = File(...),
    image_t2: UploadFile = File(...),
):
    """
    Analyze land-cover change between two satellite images.
    """

    allowed_types = {
        "image/png",
        "image/jpeg",
    }

    # Validate T1 image.
    if image_t1.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="T1 must be a PNG or JPEG image.",
        )

    # Validate T2 image.
    if image_t2.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="T2 must be a PNG or JPEG image.",
        )

    # Create temporary storage for uploaded images.
    temp_dir = Path(
        tempfile.mkdtemp()
    )

    try:

        t1_path = (
            temp_dir / "image_t1.png"
        )

        t2_path = (
            temp_dir / "image_t2.png"
        )

        # Save T1.
        with t1_path.open("wb") as buffer:
            shutil.copyfileobj(
                image_t1.file,
                buffer,
            )

        # Save T2.
        with t2_path.open("wb") as buffer:
            shutil.copyfileobj(
                image_t2.file,
                buffer,
            )

        # -------------------------------------------------
        # RUN TERRAVISION PIPELINE
        # -------------------------------------------------

        results = analyze_land_change(
            t1_path,
            t2_path,
            model=model,
            device=device,
        )

        # -------------------------------------------------
        # CREATE VISUALIZATIONS
        # -------------------------------------------------

        segmentation_t1 = (
            colorize_segmentation_mask(
                results["mask_t1"]
            )
        )

        segmentation_t2 = (
            colorize_segmentation_mask(
                results["mask_t2"]
            )
        )

        change_map = create_change_map(
            results["change_mask"]
        )

        # Convert visualization images to Base64.
        segmentation_t1_base64 = (
            image_to_base64(
                segmentation_t1
            )
        )

        segmentation_t2_base64 = (
            image_to_base64(
                segmentation_t2
            )
        )

        change_map_base64 = (
            image_to_base64(
                change_map
            )
        )

        # -------------------------------------------------
        # API RESPONSE
        # -------------------------------------------------

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
                "segmentation_t1": (
                    segmentation_t1_base64
                ),
                "segmentation_t2": (
                    segmentation_t2_base64
                ),
                "change_map": (
                    change_map_base64
                ),
            },
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    finally:

        # Always delete temporary uploaded files.
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )

    
from functools import lru_cache


@lru_cache(maxsize=1)
def get_temporal_detector():
    return TemporalChangeDetector()


def encode_change_mask(mask):
    mask_image = Image.fromarray(
        (mask * 255).astype(np.uint8),
        mode="L",
    )

    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.post("/analyze/buildings")
async def analyze_buildings(
    image_t1: UploadFile = File(...),
    image_t2: UploadFile = File(...),
):
    try:
        before = Image.open(
            io.BytesIO(await image_t1.read())
        ).convert("RGB")

        after = Image.open(
            io.BytesIO(await image_t2.read())
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Please upload two valid satellite images.",
        )

    if before.size != after.size:
        raise HTTPException(
            status_code=400,
            detail="Before and after images must have matching dimensions.",
        )

    detector = get_temporal_detector()
    result = detector.predict(before, after)

    return {
        "analysis_type": "building_change_detection",
        "model": "Temporal U-Net",
        "image_size": [256, 256],
        "changed_pixels": result["changed_pixels"],
        "total_pixels": result["total_pixels"],
        "change_percentage": result["change_percentage"],
        "change_mask": encode_change_mask(
            result["change_mask"]
        ),
    }
