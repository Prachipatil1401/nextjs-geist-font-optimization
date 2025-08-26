from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DiseaseType(str, Enum):
    HEALTHY = "healthy"
    BACTERIAL = "bacterial"
    FUNGAL = "fungal"
    VIRAL = "viral"
    NUTRITIONAL = "nutritional"
    ENVIRONMENTAL = "environmental"


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    filename: str = Field(..., description="Original filename")
    location: Optional[Dict[str, float]] = Field(None, description="GPS coordinates")
    drone_id: Optional[str] = Field(None, description="Drone identifier")


class DetectionResponse(BaseModel):
    id: str = Field(..., description="Unique detection ID")
    success: bool = Field(..., description="Whether detection was successful")
    disease_name: str = Field(..., description="Detected disease name")
    disease_type: DiseaseType = Field(..., description="Type of disease")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    severity: SeverityLevel = Field(..., description="Disease severity")
    recommendations: List[str] = Field(..., description="Treatment recommendations")
    timestamp: datetime = Field(..., description="Detection timestamp")
    image_url: Optional[str] = Field(None, description="URL to processed image")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    model_status: str = Field(..., description="ML model status")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(..., description="Error timestamp")


class BatchDetectionRequest(BaseModel):
    images: List[DetectionRequest] = Field(..., min_items=1, max_items=10)
    batch_id: Optional[str] = Field(None, description="Batch identifier")


class BatchDetectionResponse(BaseModel):
    batch_id: str = Field(..., description="Batch identifier")
    results: List[DetectionResponse] = Field(..., description="Detection results")
    total_processed: int = Field(..., description="Total images processed")
    successful_detections: int = Field(..., description="Successful detections")
    failed_detections: int = Field(..., description="Failed detections")
    processing_time: float = Field(..., description="Total processing time in seconds")


class ModelInfo(BaseModel):
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    description: str = Field(..., description="Model description")
    supported_diseases: List[str] = Field(..., description="List of supported diseases")
    last_updated: datetime = Field(..., description="Last model update")
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model accuracy")
