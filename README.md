  # PEC-Hackathon

  # Document Processing API -- README

This repository contains a FastAPI-based backend for document processing
using YOLO-based layout detection, OCR-based table extraction, and
OneChart-based chart analysis.

------------------------------------------------------------------------

## 🚀 Features

### 1. **Document Layout Detection (YOLOv8)**

-   Detects tables, charts, images, and figures from PDF or image files.
-   Crops and classifies regions using a custom YOLO model.
-   Preloads the model at startup for fast inference.

### 2. **Table Extraction with OCR (EasyOCR)**

-   Applies OCR on detected table regions.
-   Auto-detects table rows/columns using bounding box clustering.
-   Returns structured JSON output ready for UI display.

### 3. **Chart Analysis Using OneChart (VLM)**

-   Extracts key information from charts using the OneChart transformer
    model.
-   Gracefully handles CUDA errors and switches to CPU fallback.
-   Uses timeouts to prevent long-running GPU operations.

⚠️ **Note:**\
**ChartOCR / OneChart currently fails or times out due to insufficient
GPU VRAM.**\
On systems with low compute resources, chart analysis will return
`"status": "failed"` or `"timeout"`.

------------------------------------------------------------------------

## 📦 Installation

### 1. Clone the repository

``` bash
git clone <repo-url>
cd <project-folder>
```

### 2. Create and activate environment

You must install dependencies manually or via `requirements.txt`.

Example:

``` bash
conda create -n doc-process python=3.10
conda activate doc-process
pip install -r requirements.txt
```

### 3. Install Poppler (for PDF processing)

Required for `pdf2image`.

#### Ubuntu:

``` bash
sudo apt install poppler-utils
```

#### Windows:

Download from:\
https://github.com/oschwartz10612/poppler-windows

------------------------------------------------------------------------

## ▶️ Running the API

``` bash
uvicorn main:app --reload
```

Server runs at:

    http://127.0.0.1:8000

------------------------------------------------------------------------

## 📤 API Endpoints

### **1. Health Check**

    GET /health

### **2. Upload & Process a File**

    POST /upload_and_process/

Payload: - PDF, PNG, JPG files - Optional flag `analyze_charts=true`

Returns: - Layout detections\
- Extracted table data\
- Chart images & analysis (if enabled)\
- Summary statistics

------------------------------------------------------------------------

## 📁 Project Structure

    main.py                 ← Main FastAPI application  
    models/                 ← YOLO model files (custom)  
    utils/                  ← OCR + processing utilities  

------------------------------------------------------------------------

## ⚠️ Known Issues

### 1. ❗ OneChart fails on GPU with limited VRAM

-   Causes CUDA OOM or inference timeout
-   CPU fallback works but is very slow
-   Expected output: `"status": "failed"`, `"timeout"`, or `"skipped"`

### 2. ❗ EasyOCR is slow on large tables

-   Consider switching to PaddleOCR or Tesseract if speed is required.

------------------------------------------------------------------------

## 🔧 Customization

### Change YOLO model path:

Set environment variable:

``` bash
export YOLO_MODEL_PATH="path/to/model.pt"
```

### Disable chart analysis:

``` bash
POST /upload_and_process/?analyze_charts=false
```

------------------------------------------------------------------------

## 📄 License

MIT License (modify as per your project needs)

------------------------------------------------------------------------

If you need a documentation site, Postman collection, or frontend
integration, I can generate those as well.
