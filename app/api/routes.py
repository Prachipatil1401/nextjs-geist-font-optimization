import base64
import logging
from typing import List
from datetime import datetime
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse

from app.models.schemas import (
    DetectionRequest, 
    DetectionResponse, 
    HealthCheckResponse,
    BatchDetectionRequest,
    BatchDetectionResponse,
    ModelInfo,
    ErrorResponse
)
from app.services.huggingface_service import HuggingFaceService
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["plant-disease-detection"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        model_status="ready"
    )


@router.post("/detect", response_model=DetectionResponse)
async def detect_disease(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    drone_id: str = Form(None),
    latitude: float = Form(None),
    longitude: float = Form(None)
):
    """
    Detect plant disease from uploaded image.
    
    Args:
        image: Image file to analyze
        drone_id: Optional drone identifier
        latitude: Optional GPS latitude
        longitude: Optional GPS longitude
        
    Returns:
        DetectionResponse with disease information
    """
    try:
        # Read and encode image
        image_data = await image.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Prepare location data
        location = None
        if latitude is not None and longitude is not None:
            location = {"latitude": latitude, "longitude": longitude}
        
        # Create detection request
        detection_request = DetectionRequest(
            image_data=image_base64,
            filename=image.filename,
            location=location,
            drone_id=drone_id
        )
        
        # Process with Hugging Face
        async with HuggingFaceService() as service:
            result = await service.detect_disease(
                detection_request.image_data, 
                detection_request.filename
            )
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@router.post("/detect/base64", response_model=DetectionResponse)
async def detect_disease_base64(request: DetectionRequest):
    """
    Detect plant disease from base64 encoded image.
    
    Args:
        request: DetectionRequest with base64 image data
        
    Returns:
        DetectionResponse with disease information
    """
    try:
        async with HuggingFaceService() as service:
            result = await service.detect_disease(
                request.image_data, 
                request.filename
            )
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@router.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_disease_batch(request: BatchDetectionRequest):
    """
    Batch detect plant diseases from multiple images.
    
    Args:
        request: BatchDetectionRequest with multiple images
        
    Returns:
        BatchDetectionResponse with results for all images
    """
    try:
        results = []
        successful_count = 0
        failed_count = 0
        start_time = datetime.now()
        
        async with HuggingFaceService() as service:
            for detection_request in request.images:
                try:
                    result = await service.detect_disease(
                        detection_request.image_data,
                        detection_request.filename
                    )
                    if result.success:
                        successful_count += 1
                    else:
                        failed_count += 1
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing image {detection_request.filename}: {str(e)}")
                    failed_count += 1
                    # Create error response
                    error_result = DetectionResponse(
                        id=f"det_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
                        success=False,
                        disease_name="Unknown",
                        disease_type=DiseaseType.ENVIRONMENTAL,
                        confidence=0.0,
                        severity=SeverityLevel.LOW,
                        recommendations=["Image processing failed"],
                        timestamp=datetime.now(),
                        image_url=None,
                        metadata={"error": str(e)}
                    )
                    results.append(error_result)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        batch_id = request.batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return BatchDetectionResponse(
            batch_id=batch_id,
            results=results,
            total_processed=len(request.images),
            successful_detections=successful_count,
            failed_detections=failed_count,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in batch processing: {str(e)}"
        )


@router.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the current ML model."""
    return ModelInfo(
        name=settings.MODEL_NAME,
        version="1.0.0",
        description="Plant disease detection model using Hugging Face Transformers",
        supported_diseases=[
            "Wheat Rust",
            "Potato Blight", 
            "Powdery Mildew",
            "Leaf Spot",
            "Root Rot",
            "Fusarium Wilt",
            "Tobacco Mosaic Virus",
            "Healthy Plant"
        ],
        last_updated=datetime.now(),
        accuracy=0.92
    )


@router.get("/diseases", response_model=List[str])
async def get_supported_diseases():
    """Get list of supported diseases."""
    return [
        "Wheat Rust",
        "Potato Blight",
        "Powdery Mildew", 
        "Leaf Spot",
        "Root Rot",
        "Fusarium Wilt",
        "Tobacco Mosaic Virus",
        "Healthy Plant"
    ]


@router.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc),
            timestamp=datetime.now()
        ).dict()
    )
