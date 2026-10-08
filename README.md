# TerraVision

### AI-Powered Satellite Imagery Analysis and Building Change Detection

TerraVision is a full-stack geospatial AI application that analyzes satellite imagery using machine learning and deep learning. It supports land-cover segmentation and building change detection through a React dashboard and FastAPI backend.

The project combines semantic segmentation, temporal image analysis, model evaluation, and interactive visualization to help users explore predicted changes between satellite images captured at different times.

## Key Features

- **Land-Cover Segmentation:** Classifies satellite imagery into seven land-cover categories using a U-Net model trained on LoveDA.
- **Building Change Detection:** Uses a Temporal U-Net trained on LEVIR-CD to identify changed building regions between two satellite images.
- **Interactive Dashboard:** React interface for uploading before-and-after images and exploring results.
- **Before/After Comparison:** Interactive slider for visually comparing satellite images.
- **Change Visualization:** Binary building-change masks and red-highlighted overlays.
- **Downloadable Results:** Export analysis reports in JSON format and building-change masks and overlays as PNG files.
- **REST API:** FastAPI endpoints for both analysis workflows.
- **Automated Testing:** Pytest-based API and model-component tests.

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript, CSS |
| Backend | Python, FastAPI, Uvicorn |
| Deep Learning | PyTorch, Torchvision, U-Net |
| Machine Learning | Scikit-learn, Random Forest |
| Image Processing | OpenCV, Pillow, NumPy |
| Data Analysis | Pandas, Matplotlib |
| Testing | Pytest, FastAPI TestClient |
| Version Control | Git, GitHub |

## System Architecture

```text
                  TerraVision Dashboard
                        React + Vite
                             |
                   Satellite Image Upload
                        T1 and T2
                             |
                       FastAPI Backend
                             |
                  Select Analysis Mode
                       /           \
                      /             \
             Land-Cover          Building Change
              Analysis              Detection
                 |                      |
             U-Net Model          Temporal U-Net
                 |                      |
          Segmentation Maps       Binary Change Mask
                 |                      |
          Class Statistics        Change Percentage
                 |                      |
                 +----------+-----------+
                            |
                    React Visualization
                            |
                 Downloadable Results
```

## Machine Learning Models

### 1. Land-Cover Segmentation — U-Net

The land-cover model performs semantic segmentation of satellite images into seven categories:

1. Background
2. Building
3. Road
4. Water
5. Barren
6. Forest
7. Agriculture

The model was trained using the LoveDA satellite imagery dataset.

Land-cover change analysis processes the before-and-after images independently and compares their predicted segmentation maps.

**Validation performance (300 sampled LoveDA images):**

| Metric | Result |
|---|---|
| Mean IoU | 31.32% |
| Mean Dice | 47.07% |

### 2. Building Change Detection — Temporal U-Net

The Temporal U-Net processes two satellite images together by combining their RGB channels into a six-channel input.

The model predicts a binary mask identifying building-change regions.

**Held-out LEVIR-CD test performance (127 image pairs):**

| Metric | Result |
|---|---|
| Precision | 86.48% |
| Recall | 80.09% |
| F1 Score | 83.16% |
| IoU | 71.18% |
| Pixel Accuracy | 98.34% |

The model was trained for up to 30 epochs, with the best validation checkpoint selected for inference.

### 3. Random Forest Baseline

A Random Forest classifier was implemented as a traditional machine-learning baseline for land-cover classification.

| Metric | Result |
|---|---|
| Mean IoU | 10.40% |
| Mean Dice | 18.73% |

The baseline uses pixel-level RGB features. Its validation sample differs from the U-Net evaluation sample, so the results are preliminary rather than a strictly controlled comparison.

## Datasets

**LoveDA:** Used for seven-class land-cover segmentation. Contains urban and rural satellite imagery with semantic segmentation labels.

**LEVIR-CD:** Used for binary building change detection with paired satellite images and ground-truth change masks.

The project excludes raw datasets and trained model checkpoints from Git because of their size. Users must obtain the datasets and model checkpoints separately to reproduce the relevant workflows.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sherley17reena/TerraVision.git
cd TerraVision
```

### 2. Set up the Python environment

Python 3.12 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment using `.venv\Scripts\activate`.

### 3. Set up model checkpoints

Place the trained model checkpoints in `outputs/models/` and ensure the checkpoint paths match those expected by the inference code.

The application requires the appropriate checkpoints for the selected analysis mode.

### 4. Start the backend

From the project root:

```bash
uvicorn src.api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### 5. Start the frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, typically `http://localhost:5173`.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/analyze` | Land-cover segmentation and predicted land-cover change analysis |
| POST | `/analyze/buildings` | Temporal U-Net building-change detection |
| GET | `/docs` | Interactive FastAPI documentation |

Both analysis endpoints accept two uploaded images using the form fields `image_t1` and `image_t2`.

## Testing

Run the automated API tests from the project root:

```bash
python -m pytest tests/test_api.py -v
```

The current API test suite contains five passing tests covering valid analysis requests, missing files, invalid images, and mismatched image dimensions.

Additional tests cover the LEVIR-CD dataset loader