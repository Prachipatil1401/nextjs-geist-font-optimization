#!/usr/bin/env python3
"""
AgroDrone Plant Disease Detection API
FastAPI-based backend for plant disease detection using ML models
"""

import os
import io
import base64
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from PIL import Image
import numpy as np
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AgroDrone Plant Disease Detection API",
    description="API for plant disease detection using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
HUGGINGFACE_API_URL = os.getenv("HUGGINGFACE_API_URL", "https://api-inference.huggingface.co/models")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "plant-disease-detection")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str

class DetectionResult(BaseModel):
    disease_name: str
    confidence: float
    description: str
    treatment: Optional[str] = None
    severity: Optional[str] = None

class DetectionResponse(BaseModel):
    success: bool
    image_id: str
    results: List[DetectionResult]
    processing_time: float
    model_version: str

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    timestamp: datetime

class ModelInfo(BaseModel):
    name: str
    version: str
    status: str
    last_updated: Optional[datetime] = None

# Utility functions
def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
        )

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model inference"""
    # Resize to model input size (adjust based on your model)
    image = image.convert('RGB')
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    img_array = np.array(image)
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

async def call_huggingface_api(image_data: bytes) -> Dict[str, Any]:
    """Call Hugging Face inference API"""
    if not HUGGINGFACE_API_TOKEN:
        raise HTTPException(
            status_code=500, 
            detail="Hugging Face API token not configured"
        )
    
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Convert image to base64
    img_base64 = base64.b64encode(image_data).decode('utf-8')
    
    payload = {
        "inputs": img_base64
    }
    
    try:
        response = requests.post(
            f"{HUGGINGFACE_API_URL}/{MODEL_NAME}",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Hugging Face API error: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Model inference failed: {str(e)}"
        )

def format_detection_results(raw_results: List[Dict[str, Any]]) -> List[DetectionResult]:
    """Format raw model results into structured response"""
    formatted_results = []
    
    for result in raw_results:
        # Adjust based on your model's output format
        disease_name = result.get('label', 'Unknown Disease')
        confidence = float(result.get('score', 0.0))
        
        # Map disease names to descriptions and treatments
        disease_info = get_disease_info(disease_name)
        
        formatted_results.append(
            DetectionResult(
                disease_name=disease_name,
                confidence=confidence,
                description=disease_info.get('description', 'No description available'),
                treatment=disease_info.get('treatment'),
                severity=disease_info.get('severity')
            )
        )
    
    return formatted_results

def get_disease_info(disease_name: str) -> Dict[str, str]:
    """Get additional information about detected disease"""
    # This could be expanded with a comprehensive disease database
    disease_db = {
        "healthy": {
            "description": "Plant appears to be healthy with no visible disease symptoms",
            "treatment": "Continue regular care and monitoring",
            "severity": "None"
        },
        "leaf_rust": {
            "description": "Fungal disease causing orange-brown pustules on leaves",
            "treatment": "Apply fungicide, improve air circulation, remove infected leaves",
            "severity": "Moderate"
        },
        "powdery_mildew": {
            "description": "White powdery fungal growth on leaf surfaces",
            "treatment": "Apply fungicide, reduce humidity, increase air circulation",
            "severity": "Mild to Moderate"
        },
        "bacterial_blight": {
            "description": "Dark water-soaked lesions on leaves and stems",
            "treatment": "Remove infected parts, apply copper-based bactericide, avoid overhead watering",
            "severity": "Severe"
        }
    }
    
    return disease_db.get(disease_name.lower(), {
        "description": f"Disease detected: {disease_name}",
        "treatment": "Consult agricultural expert for specific treatment",
        "severity": "Unknown"
    })

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AgroDrone Plant Disease Detection API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0"
    )

@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get available models information"""
    return [
        ModelInfo(
            name=MODEL_NAME,
            version="1.0.0",
            status="active",
            last_updated=datetime.utcnow()
        )
    ]

@app.post("/upload", response_model=Dict[str, str])
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
):
    """Upload image for processing"""
    validate_image_file(file)
    
    # Generate unique image ID
    image_id = f"img_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(file.filename) % 10000}"
    
    # Read image data
    image_data = await file.read()
    
    # Store image temporarily (in production, use proper storage)
    temp_path = f"/tmp/{image_id}.jpg"
    with open(temp_path, "wb") as f:
        f.write(image_data)
    
    # Schedule cleanup
    background_tasks.add_task(os.remove, temp_path)
    
    return {
        "image_id": image_id,
        "filename": file.filename,
        "status": "uploaded",
        "size": len(image_data)
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_disease(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """Detect plant diseases in uploaded image"""
    validate_image_file(file)
    
    start_time = datetime.utcnow()
    
    try:
        # Read and validate image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Call ML model
        raw_results = await call_huggingface_api(image_data)
        
        # Format results
        all_results = format_detection_results(raw_results)
        
        # Filter by confidence threshold
        filtered_results = [
            result for result in all_results 
            if result.confidence >= confidence_threshold
        ]
        
        # If no results above threshold, return top result
        if not filtered_results and all_results:
            filtered_results = [all_results[0]]
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return DetectionResponse(
            success=True,
            image_id=f"detection_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            results=filtered_results,
            processing_time=processing_time,
            model_version=MODEL_NAME
        )
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )

@app.post("/detect/base64", response_model=DetectionResponse)
async def detect_disease_base64(
    image_base64: str = Form(...),
    confidence_threshold: float = Form(0.5)
):
    """Detect diseases from base64 encoded image"""
    start_time = datetime.utcnow()
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Call ML model
        raw_results = await call_huggingface_api(image_data)
        
        # Format results
        all_results = format_detection_results(raw_results)
        
        # Filter by confidence threshold
        filtered_results = [
            result for result in all_results 
            if result.confidence >= confidence_threshold
        ]
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return DetectionResponse(
            success=True,
            image_id=f"base64_detection_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            results=filtered_results,
            processing_time=processing_time,
            model_version=MODEL_NAME
        )
        
    except Exception as e:
        logger.error(f"Base64 detection error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Base64 detection failed: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.utcnow()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc),
            timestamp=datetime.utcnow()
        ).dict()
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting AgroDrone API...")
    
    # Check Hugging Face API token
    if not HUGGINGFACE_API_TOKEN:
        logger.warning("Hugging Face API token not configured")
    
    logger.info("API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AgroDrone API...")

# Main entry point
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level="info"
    )
