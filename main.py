import io
import os
import tempfile
from typing import List, Dict, Any, Optional
import json
from contextlib import asynccontextmanager
import base64

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import torch

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Add this near the top with other globals
executor = ThreadPoolExecutor(max_workers=2)

# CRITICAL FIX: Patch torch.load to allow loading custom YOLO models
# This must happen BEFORE importing ultralytics
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that sets weights_only=False for custom models."""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

import dill
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential

# Add safe globals for PyTorch 2.6+ compatibility
from torch.nn import (
    Module, Conv2d, BatchNorm2d, ReLU, MaxPool2d, 
    Upsample, Linear, Dropout, Identity
)
from collections import OrderedDict

safe_globals_list = [
    DetectionModel,
    dill._dill._load_type,
    Sequential,
    Module,
    Conv2d,
    BatchNorm2d,
    ReLU,
    MaxPool2d,
    Upsample,
    Linear,
    Dropout,
    Identity,
    OrderedDict,
]

torch.serialization.add_safe_globals(safe_globals_list)

from ultralytics import YOLO

# Try importing OpenCV and EasyOCR with graceful fallbacks
try:
    import cv2
    import easyocr
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    HAS_OCR_DEPS = True
except ImportError:
    print("Warning: 'easyocr', 'opencv-python', or 'numpy' missing. Table parsing will be skipped.")
    HAS_OCR_DEPS = False
    ocr_reader = None

try:
    from pdf2image import convert_from_bytes
except ImportError:
    convert_from_bytes = None
    print("Warning: 'pdf2image' is not installed or Poppler is missing. PDF processing will fail.")

# Try importing OneChart model
try:
    from transformers import AutoModel, AutoTokenizer
    HAS_ONECHART = True
except ImportError:
    print("Warning: 'transformers' not installed. Chart analysis will be skipped.")
    HAS_ONECHART = False

# --- Configuration ---
MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    r"yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt"
)
MAX_IMAGE_DIMENSION = 4096

# --- Global state for models (loaded once) ---
yolo_model = None
onechart_model = None
onechart_tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global yolo_model, onechart_model, onechart_tokenizer
    
    # Load YOLO model
    print(f"Loading YOLO model from {MODEL_PATH}...")
    try:
        yolo_model = YOLO(MODEL_PATH)
        print("✓ YOLO model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        import traceback
        traceback.print_exc()
        yolo_model = None
    
    # Load OneChart model
    if HAS_ONECHART:
        print("Loading OneChart model...")
        try:
            onechart_tokenizer = AutoTokenizer.from_pretrained(
                'kppkkp/OneChart', 
                trust_remote_code=True, 
                use_fast=False, 
                padding_side="right"
            )
            onechart_model = AutoModel.from_pretrained(
                "kppkkp/OneChart", 
                trust_remote_code=True
            )
            onechart_model = onechart_model.eval()
            
            # Move to CUDA if available
            if torch.cuda.is_available():
                onechart_model = onechart_model.cuda()
                print("✓ OneChart model loaded successfully (GPU)")
            else:
                print("✓ OneChart model loaded successfully (CPU)")
        except Exception as e:
            print(f"✗ Failed to load OneChart model: {e}")
            import traceback
            traceback.print_exc()
            onechart_model = None
            onechart_tokenizer = None
    
    yield
    
    # Cleanup
    print("Shutting down...")
    if yolo_model:
        del yolo_model
    if onechart_model:
        del onechart_model
    if onechart_tokenizer:
        del onechart_tokenizer


app = FastAPI(
    title="Document Processing API",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Table Extraction Logic ---
class TableExtractor:
    """Encapsulates logic to extract tabular data from an image using EasyOCR."""
    
    def __init__(self, reader=ocr_reader, scale_factor=2):
        self.reader = reader
        self.scale_factor = scale_factor

    def process_image(self, pil_image: Image.Image) -> List[List[str]]:
        if not self.reader:
            return []
        
        try:
            # Convert PIL to OpenCV format
            img_np = np.array(pil_image)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Upscale for better OCR
            height, width = img_cv.shape[:2]
            new_size = (width * self.scale_factor, height * self.scale_factor)
            img_cv = cv2.resize(img_cv, new_size, interpolation=cv2.INTER_LINEAR)
            
            # Run OCR
            results = self.reader.readtext(img_cv)
            if not results:
                return []
            
            # Process results
            ocr_boxes = self._normalize_boxes(results)
            if not ocr_boxes:
                return []
                
            rows = self._cluster_into_rows(ocr_boxes)
            merged_rows = self._merge_multiline_rows(rows)
            table_data = self._align_to_columns(merged_rows)
            
            return table_data
        except Exception as e:
            print(f"Error in table extraction: {e}")
            return []

    def _normalize_boxes(self, raw_results):
        boxes = []
        for bbox, text, conf in raw_results:
            x_center = sum([pt[0] for pt in bbox]) / 4
            y_center = sum([pt[1] for pt in bbox]) / 4
            boxes.append({
                'x_center': x_center,
                'y_center': y_center,
                'text': text,
                'bbox': bbox
            })
        boxes.sort(key=lambda b: b['y_center'])
        return boxes

    def _cluster_into_rows(self, boxes):
        y_centers = [b['y_center'] for b in boxes]
        vertical_gaps = [y_centers[i+1] - y_centers[i] for i in range(len(y_centers)-1)]
        row_threshold = np.mean(vertical_gaps) * 0.25 if vertical_gaps else 15 * self.scale_factor
        
        rows = []
        for box in boxes:
            added = False
            for r in rows:
                if abs(r['y_center'] - box['y_center']) < row_threshold:
                    r['boxes'].append(box)
                    r['y_center'] = np.mean([b['y_center'] for b in r['boxes']])
                    added = True
                    break
            if not added:
                rows.append({'y_center': box['y_center'], 'boxes': [box]})
        return rows

    def _merge_multiline_rows(self, rows):
        row_centers = [r['y_center'] for r in rows]
        row_gaps = [row_centers[i+1] - row_centers[i] for i in range(len(row_centers)-1)]
        avg_row_gap = np.mean(row_gaps) if row_gaps else (15 * self.scale_factor)
        
        merged_rows = []
        i = 0
        while i < len(rows):
            current_row = rows[i]
            j = i + 1
            while j < len(rows) and (rows[j]['y_center'] - current_row['y_center']) < avg_row_gap:
                current_row['boxes'].extend(rows[j]['boxes'])
                current_row['y_center'] = np.mean([b['y_center'] for b in current_row['boxes']])
                j += 1
            merged_rows.append(current_row)
            i = j
        return merged_rows

    def _align_to_columns(self, rows):
        if not rows:
            return []
        
        first_row_boxes = sorted(rows[0]['boxes'], key=lambda b: b['x_center'])
        col_centers = [b['x_center'] for b in first_row_boxes]
        num_columns = len(col_centers)
        
        table = []
        for r in rows:
            row_cells = [""] * num_columns
            for b in r['boxes']:
                distances = [abs(b['x_center'] - cc) for cc in col_centers]
                col_idx = distances.index(min(distances))
                if row_cells[col_idx]:
                    row_cells[col_idx] += " " + b['text']
                else:
                    row_cells[col_idx] = b['text']
            table.append(row_cells)
        return table


# Instantiate extractor
table_extractor = TableExtractor() if HAS_OCR_DEPS else None


# --- Chart Analysis with OneChart ---
def analyze_chart(pil_image: Image.Image, timeout: int = 30) -> Dict[str, Any]:
    """Analyze chart using OneChart model with timeout."""
    if not onechart_model or not onechart_tokenizer:
        return {"error": "OneChart model not available", "status": "skipped"}
    
    try:
        # Save image temporarily for OneChart
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            pil_image.save(tmp_file.name, format='PNG')
            tmp_path = tmp_file.name
        
        try:
            print(f"    ... Running OneChart analysis (timeout: {timeout}s) ...")
            
            # Clear CUDA cache before running
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Wrap the model call to catch CUDA errors
            def run_inference():
                try:
                    with torch.no_grad():  # Disable gradients for inference
                        return onechart_model.chat(onechart_tokenizer, tmp_path)
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        print(f"    ... CUDA error detected, retrying on CPU ...")
                        # Move model to CPU temporarily
                        device = next(onechart_model.parameters()).device
                        onechart_model.to('cpu')
                        try:
                            result = onechart_model.chat(onechart_tokenizer, tmp_path)
                            onechart_model.to(device)  # Move back to original device
                            return result
                        except:
                            onechart_model.to(device)
                            raise
                    raise
            
            # Run with timeout
            future = executor.submit(run_inference)
            result = future.result(timeout=timeout)
            
            return {
                "analysis": result,
                "status": "success"
            }
        except FuturesTimeoutError:
            print(f"    ... OneChart analysis timed out after {timeout}s ...")
            return {
                "error": f"Analysis timed out after {timeout} seconds",
                "status": "timeout"
            }
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        print(f"Error in chart analysis: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "status": "failed"
        }


# --- Helper Functions ---
def image_to_base64(image: Image.Image, format: str = 'PNG') -> str:
    """Convert PIL Image to base64 string."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')


def resize_if_needed(image: Image.Image, max_dim: int = MAX_IMAGE_DIMENSION) -> Image.Image:
    """Resize image if it exceeds max dimension while maintaining aspect ratio."""
    width, height = image.size
    if width <= max_dim and height <= max_dim:
        return image
    
    scale = min(max_dim / width, max_dim / height)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def run_yolo_detection(image: Image.Image) -> List[Dict[str, Any]]:
    """Runs YOLO detection on an image using the pre-loaded model."""
    if yolo_model is None:
        raise RuntimeError("YOLO model not loaded")
    
    try:
        results = yolo_model(image)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0])
                cls_id = int(b.cls[0])
                class_name = yolo_model.names[cls_id]
                
                detections.append({
                    "class_name": class_name,
                    "box": [x1, y1, x2, y2],
                    "confidence": conf
                })
        
        return detections
    except Exception as e:
        print(f"Error during YOLO detection: {e}")
        return []


# --- Core Processing Logic ---
async def process_image_and_route(image: Image.Image, page_num: int, analyze_charts: bool = True) -> Dict[str, Any]:
    """Runs detection, extracts data if table, analyzes charts, and returns all results."""
    
    # Resize if needed to prevent memory issues
    image = resize_if_needed(image)
    
    # Run YOLO Detection
    detection_results = run_yolo_detection(image)
    
    tables = []
    charts = []
    
    # Process each detection
    for idx, det in enumerate(detection_results):
        class_name = det["class_name"].lower()
        box = det["box"]  # [x1, y1, x2, y2]
        confidence = det["confidence"]
        
        print(f"  Detected: {class_name} (conf: {confidence:.2f})")
        
        try:
            # Crop the detected region
            cropped = image.crop((box[0], box[1], box[2], box[3]))
            
            if "table" in class_name:
                # Extract table data
                extracted_data = []
                if table_extractor:
                    print(f"    ... Running OCR on Table in Page {page_num} ...")
                    extracted_data = table_extractor.process_image(cropped)
                else:
                    print("    ... Skipping OCR (Dependencies missing) ...")
                
                # Convert cropped image to base64 for frontend
                img_base64 = image_to_base64(cropped)
                
                tables.append({
                    "page": page_num,
                    "bounding_box": {
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3]
                    },
                    "confidence": confidence,
                    "extracted_data": extracted_data,
                    "image": img_base64,
                    "rows": len(extracted_data),
                    "columns": len(extracted_data[0]) if extracted_data else 0
                })
                    
            elif "image" in class_name or "chart" in class_name or "figure" in class_name or "picture" in class_name:
                # Convert cropped image to base64 for frontend
                img_base64 = image_to_base64(cropped)
                
                chart_data = {
                    "page": page_num,
                    "bounding_box": {
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3]
                    },
                    "confidence": confidence,
                    "image": img_base64,
                    "type": class_name
                }
                
                # Only analyze if requested
                if analyze_charts and onechart_model:
                    print(f"    ... Analyzing Chart #{len(charts) + 1} in Page {page_num} ...")
                    chart_analysis = analyze_chart(cropped, timeout=30)
                    chart_data["analysis"] = chart_analysis
                else:
                    chart_data["analysis"] = {"status": "skipped", "reason": "Chart analysis disabled"}
                
                charts.append(chart_data)
                    
        except Exception as e:
            print(f"Error processing detection on page {page_num}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return {
        "page": page_num,
        "tables": tables,
        "charts": charts,
        "detections": detection_results
    }


# --- FastAPI Endpoints ---
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "yolo_model_loaded": yolo_model is not None,
        "ocr_available": HAS_OCR_DEPS,
        "onechart_available": onechart_model is not None,
        "pdf_support": convert_from_bytes is not None,
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/upload_and_process/")
async def upload_and_process(
    file: UploadFile = File(...),
    analyze_charts: bool = True  # Add this parameter
):
    """Process uploaded PDF or image file and return all extracted data."""
    
    # Validate file type
    if file.content_type not in ["application/pdf", "image/jpeg", "image/png"]:
        raise HTTPException(400, "Invalid file type. Upload PDF, JPG, or PNG.")
    
    # Check if YOLO model is loaded
    if yolo_model is None:
        raise HTTPException(500, "YOLO model not loaded. Check server logs.")
    
    file_bytes = await file.read()
    all_results = []
    
    # Handle PDF
    if file.content_type == "application/pdf":
        if not convert_from_bytes:
            raise HTTPException(500, "PDF processing unavailable (Poppler missing).")
        
        with tempfile.TemporaryDirectory() as path:
            try:
                images = convert_from_bytes(file_bytes, dpi=200, output_folder=path)
            except Exception as e:
                raise HTTPException(500, f"PDF conversion failed: {e}")
            
            print(f"PDF converted to {len(images)} pages.")
            
            for i, image in enumerate(images):
                page_num = i + 1
                print(f"\nProcessing page {page_num}/{len(images)}...")
                
                results = await process_image_and_route(image, page_num, analyze_charts)
                all_results.append(results)
    
    # Handle Single Image
    else:
        try:
            image = Image.open(io.BytesIO(file_bytes))
            results = await process_image_and_route(image, 1, analyze_charts)
            all_results.append(results)
        except IOError as e:
            raise HTTPException(400, f"Invalid image file: {e}")
    
    # Calculate summary statistics
    total_tables = sum(len(page["tables"]) for page in all_results)
    total_charts = sum(len(page["charts"]) for page in all_results)
    
    return {
        "filename": file.filename,
        "pages_processed": len(all_results),
        "summary": {
            "total_tables": total_tables,
            "total_charts": total_charts,
            "total_detections": sum(len(page["detections"]) for page in all_results)
        },
        "results": all_results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)