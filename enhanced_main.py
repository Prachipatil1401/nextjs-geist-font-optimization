#!/usr/bin/env python3
"""
Enhanced AgroDrone Plant Disease Detection API
Production-ready FastAPI backend with comprehensive features
"""

import os
import io
import base64
import json
import logging
import asyncio
import uuid
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import time

from fastapi import (
    FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks,
    Depends, Header, Query, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn
from PIL import Image
import numpy as np
import aiohttp
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Float, Text, Integer
from sqlalchemy.ext.declaratory import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import jwt
from cryptography.fernet import Fernet
import boto3
from botocore.exceptions import ClientError
from prometheus_client import Counter, Histogram, generate_latest
import psutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AgroDrone Plant Disease Detection API",
    description="Production-ready API for plant disease detection with ML models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Configuration
class Config:
    """Application configuration"""
    # API Configuration
    API_VERSION = "v2"
    API_PREFIX = f"/api/{API_VERSION}"
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Model Configuration
    HUGGINGFACE_API_URL = os.getenv("HUGGINGFACE_API_URL", "https://api-inference.huggingface.co/models")
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
    MODEL_NAME = os.getenv("MODEL_NAME", "plant-disease-detection")
    BACKUP_MODEL_NAME = os.getenv("BACKUP_MODEL_NAME", "microsoft/resnet-50")
    
    # File Configuration
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Storage Configuration
    USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
    S3_BUCKET = os.getenv("S3_BUCKET", "agrodrone-images")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./agrodrone.db")
    
    # Redis Configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour

config = Config()

# Database setup
Base = declarative_base()

class DetectionRecord(Base):
    __tablename__ = "detection_records"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=True)
    image_hash = Column(String, unique=True)
    filename = Column(String)
    file_size = Column(Integer)
    detection_results = Column(Text)
    processing_time = Column(Float)
    confidence_threshold = Column(Float)
    model_version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database connection
engine = create_engine(config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# Redis connection
redis_client = None
if config.REDIS_URL:
    try:
        redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
        redis_client.ping()
    except redis.ConnectionError:
        logger.warning("Redis connection failed, caching disabled")

# S3 client
s3_client = None
if config.USE_S3:
    try:
        s3_client = boto3.client(
            's3',
            region_name=config.AWS_REGION,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
    except Exception as e:
        logger.warning(f"S3 client initialization failed: {e}")

# Encryption
encryption_key = os.getenv("ENCRYPTION_KEY", Fernet.generate_key())
fernet = Fernet(encryption_key)

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
DETECTION_COUNT = Counter('detections_total', 'Total disease detections', ['disease_name'])
ERROR_COUNT = Counter('errors_total', 'Total errors', ['error_type'])

# Security
security = HTTPBearer()

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(..., min_length=8)

class UserLogin(BaseModel):
    username: str
    password: str

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    uptime: float
    memory_usage: Dict[str, float]
    model_status: str

class DetectionResult(BaseModel):
    disease_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    description: str
    treatment: Optional[str] = None
    severity: Optional[str] = None
    common_symptoms: Optional[List[str]] = None
    prevention_tips: Optional[List[str]] = None

class DetectionResponse(BaseModel):
    success: bool
    image_id: str
    results: List[DetectionResult]
    processing_time: float
    model_version: str
    cached: bool = False
    confidence_threshold: float

class BatchDetectionResponse(BaseModel):
    success: bool
    batch_id: str
    total_images: int
    processed_images: int
    failed_images: int
    results: List[Dict[str, Any]]
    processing_time: float

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    timestamp: datetime
    request_id: str

class ModelInfo(BaseModel):
    name: str
    version: str
    status: str
    last_updated: Optional[datetime] = None
    accuracy: Optional[float] = None
    description: Optional[str] = None

class ImageMetadata(BaseModel):
    width: int
    height: int
    format: str
    file_size: int
    color_mode: str

# Utility functions
def get_db():
    """Database dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)
    return encoded_jwt

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return TokenData(username=username)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )
    
    if file.size and file.size > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Maximum size: {config.MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
        )

def calculate_image_hash(image_data: bytes) -> str:
    """Calculate MD5 hash of image data"""
    return hashlib.md5(image_data).hexdigest()

async def check_rate_limit(client_ip: str) -> bool:
    """Check rate limiting using Redis"""
    if not redis_client:
        return True
    
    key = f"rate_limit:{client_ip}"
    try:
        current = redis_client.incr(key)
        if current == 1:
            redis_client.expire(key, config.RATE_LIMIT_WINDOW)
        
        return current <= config.RATE_LIMIT_REQUESTS
    except redis.RedisError:
        return True  # Allow if Redis is down

async def cache_detection_result(image_hash: str, result: Dict[str, Any]) -> None:
    """Cache detection result in Redis"""
    if redis_client:
        try:
            redis_client.setex(
                f"detection:{image_hash}",
                3600,  # 1 hour TTL
                json.dumps(result)
            )
        except redis.RedisError:
            pass

async def get_cached_detection(image_hash: str) -> Optional[Dict[str, Any]]:
    """Get cached detection result"""
    if redis_client:
        try:
            cached = redis_client.get(f"detection:{image_hash}")
            return json.loads(cached) if cached else None
        except redis.RedisError:
            return None
    return None

async def upload_to_s3(file_data: bytes, filename: str) -> str:
    """Upload file to S3"""
    if not s3_client:
        raise HTTPException(status_code=500, detail="S3 not configured")
    
    try:
        key = f"uploads/{datetime.utcnow().strftime('%Y/%m/%d')}/{uuid.uuid4()}_{filename}"
        s3_client.put_object(
            Bucket=config.S3_BUCKET,
            Key=key,
            Body=file_data,
            ContentType='image/jpeg'
        )
        return f"s3://{config.S3_BUCKET}/{key}"
    except ClientError as e:
        logger.error(f"S3 upload error: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload to S3")

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model inference"""
    image = image.convert('RGB')
    image = image.resize((224, 224))
    
    img_array = np.array(image)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

async def call_huggingface_api(image_data: bytes, model_name: str = None) -> Dict[str, Any]:
    """Call Hugging Face inference API with retry logic"""
    if not config.HUGGINGFACE_API_TOKEN:
        raise HTTPException(
            status_code=500, 
            detail="Hugging Face API token not configured"
        )
    
    model = model_name or config.MODEL_NAME
    
    headers = {
        "Authorization": f"Bearer {config.HUGGINGFACE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    img_base64 = base64.b64encode(image_data).decode('utf-8')
    payload = {"inputs": img_base64}
    
    # Retry logic
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{config.HUGGINGFACE_API_URL}/{model}",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 503 and attempt < max_retries - 1:
                        # Model loading, wait and retry
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    
                    response.raise_for_status()
                    return await response.json()
                    
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
            raise HTTPException(status_code=504, detail="Model inference timeout")
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
            logger.error(f"Hugging Face API error: {str(e)}")
            raise HTTPException(status_code=503, detail=f"Model inference failed: {str(e)}")

def format_detection_results(raw_results: List[Dict[str, Any]]) -> List[DetectionResult]:
    """Format raw model results into structured response"""
    formatted_results = []
    
    # Comprehensive disease database
    disease_db = {
        "healthy": {
            "description": "Plant appears to be healthy with no visible disease symptoms",
            "treatment": "Continue regular care and monitoring",
            "severity": "None",
            "common_symptoms": ["Normal leaf color", "No spots or lesions", "Proper growth"],
            "prevention_tips": ["Regular watering", "Proper nutrition", "Adequate sunlight"]
        },
        "leaf_rust": {
            "description": "Fungal disease causing orange-brown pustules on leaves",
            "treatment": "Apply copper-based fungicide, improve air circulation, remove infected leaves",
            "severity": "Moderate",
            "common_symptoms": ["Orange-brown pustules", "Leaf yellowing", "Premature leaf drop"],
            "prevention_tips": ["Avoid overhead watering", "Ensure proper spacing", "Remove plant debris"]
        },
        "powdery_mildew": {
            "description": "White powdery fungal growth on leaf surfaces",
            "treatment": "Apply sulfur-based fungicide, reduce humidity, increase air circulation",
            "severity": "Mild to Moderate",
            "common_symptoms": ["White powdery coating", "Leaf distortion", "Stunted growth"],
            "prevention_tips": ["Improve air circulation", "Reduce humidity", "Avoid excessive nitrogen"]
        },
        "bacterial_blight": {
            "description": "Dark water-soaked lesions on leaves and stems",
            "treatment": "Remove infected parts, apply copper-based bactericide, avoid overhead watering",
            "severity": "Severe",
            "common_symptoms": ["Dark water-soaked spots", "Leaf wilting", "Stem cankers"],
            "prevention_tips": ["Use disease-free seeds", "Avoid overhead watering", "Practice crop rotation"]
        },
        "late_blight": {
            "description": "Destructive fungal disease causing dark lesions on leaves and stems",
            "treatment": "Apply fungicide immediately, remove infected plants, improve drainage",
            "severity": "Severe",
            "common_symptoms": ["Dark brown lesions", "White fungal growth", "Rapid plant death"],
            "prevention_tips": ["Improve air circulation", "Avoid overhead watering", "Use resistant varieties"]
        },
        "downy_mildew": {
            "description": "Fungal disease causing yellow patches on upper leaf surfaces",
            "treatment": "Apply fungicide, improve air circulation, reduce leaf wetness",
            "severity": "Moderate",
            "common_symptoms": ["Yellow patches", "Gray fuzzy growth", "Leaf curling"],
            "prevention_tips": ["Avoid overhead watering", "Improve air circulation", "Remove infected debris"]
        }
    }
    
    for result in raw_results:
        disease_name = result.get('label', 'Unknown Disease')
        confidence = float(result.get('score', 0.0))
        
        disease_info = disease_db.get(
            disease_name.lower().replace(" ", "_"),
            {
                "description": f"Disease detected: {disease_name}",
                "treatment": "Consult agricultural expert for specific treatment",
                "severity": "Unknown",
                "common_symptoms": ["Visible symptoms"],
                "prevention_tips": ["Monitor plant health", "Maintain proper care"]
            }
        )
        
        formatted_results.append(
            DetectionResult(
                disease_name=disease_name,
                confidence=confidence,
                description=disease_info["description"],
                treatment=disease_info["treatment"],
                severity=disease_info["severity"],
                common_symptoms=disease_info["common_symptoms"],
                prevention_tips=disease_info["prevention_tips"]
            )
        )
    
    return formatted_results

# Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header and metrics"""
    start_time = time.time()
    
    # Rate limiting
    client_ip = request.client.host
    if not await check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "retry_after": 3600}
        )
    
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Process-Time"] = str(process_time)
    
    # Update metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_DURATION.observe(process_time)
    
    return response

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AgroDrone Plant Disease Detection API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "api": f"{config.API_PREFIX}"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    memory = psutil.virtual_memory()
    
    # Check model status
    model_status = "healthy" if config.HUGGINGFACE_API_TOKEN else "degraded"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="2.0.0",
        uptime=time.time() - psutil.boot_time(),
        memory_usage={
            "total": memory.total / (1024**3),  # GB
            "used": memory.used / (1024**3),    # GB
            "percent": memory.percent
        },
        model_status=model_status
    )

@app.get(f"{config.API_PREFIX}/models", response_model=List[ModelInfo])
async def get_models():
    """Get available models information"""
    return [
        ModelInfo(
            name=config.MODEL_NAME,
            version="2.0.0",
            status="active",
            last_updated=datetime.utcnow(),
            accuracy=0.95,
            description="Advanced plant disease detection model"
        ),
        ModelInfo(
            name=config.BACKUP_MODEL_NAME,
            version="1.0.0",
            status="standby",
            last_updated=datetime.utcnow(),
            accuracy=0.92,
            description="Backup ResNet-50 model"
        )
    ]

@app.post(f"{config.API_PREFIX}/detect", response_model=DetectionResponse)
async def detect_disease(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5, ge=0.0, le=1.0),
    save_result: bool = Form(True),
    db: Session = Depends(get_db),
    request: Request = None
):
    """Enhanced disease detection with caching and storage"""
    validate_image_file(file)
    
    start_time = datetime.utcnow()
    
    try:
        # Read image data
        image_data = await file.read()
        image_hash = calculate_image_hash(image_data)
        
        # Check cache
        cached_result = await get_cached_detection(image_hash)
        if cached_result:
            cached_result["cached"] = True
            return DetectionResponse(**cached_result)
        
        # Get image metadata
        image = Image.open(io.BytesIO(image_data))
        metadata = ImageMetadata(
            width=image.width,
            height=image.height,
            format=image.format,
            file_size=len(image_data),
            color_mode=image.mode
        )
        
        # Call ML model
        raw_results = await call_huggingface_api(image_data)
        
        # Format results
        all_results = format_detection_results(raw_results)
        
        # Filter by confidence threshold
        filtered_results = [
            result for result in all_results 
            if result.confidence >= confidence_threshold
        ]
        
        if not filtered_results and all_results:
            filtered_results = [all_results[0]]
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Generate response
        response = DetectionResponse(
            success=True,
            image_id=f"detection_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{image_hash[:8]}",
            results=filtered_results,
            processing_time=processing_time,
            model_version=config.MODEL_NAME,
            cached=False,
            confidence_threshold=confidence_threshold
        )
        
        # Save to database
        if save_result:
            detection_record = DetectionRecord(
                image_hash=image_hash,
                filename=file.filename,
                file_size=len(image_data),
                detection_results=json.dumps([r.dict() for r in filtered_results]),
                processing_time=processing_time,
                confidence_threshold=confidence_threshold,
                model_version=config.MODEL_NAME
            )
            db.add(detection_record)
            db.commit()
        
        # Cache result
        await cache_detection_result(image_hash, response.dict())
        
        # Update metrics
        for result in filtered_results:
            DETECTION_COUNT.labels(disease_name=result.disease_name).inc()
        
        return response
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        ERROR_COUNT.labels(error_type="detection_error").inc()
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )

@app.post(f"{config.API_PREFIX}/detect/batch")
async def batch_detect_disease(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = Form(0.5, ge=0.0, le=1.0),
    max_concurrent: int = Form(3, ge=1, le=10),
    db: Session = Depends(get_db)
):
    """Process multiple images in batch"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    start_time = datetime.utcnow()
    
    results = []
    processed = 0
    failed = 0
    
    # Process files concurrently
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_file(file: UploadFile):
        nonlocal processed, failed
        
        async with semaphore:
            try:
                validate_image_file(file)
                image_data = await file.read()
                
                raw_results = await call_huggingface_api(image_data)
                formatted_results = format_detection_results(raw_results)
                
                filtered_results = [
                    r for r in formatted_results 
                    if r.confidence >= confidence_threshold
                ]
                
                return {
                    "filename": file.filename,
                    "success": True,
                    "results": [r.dict() for r in filtered_results]
                }
                
            except Exception as e:
                failed += 1
                return {
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                }
    
    # Process all files
    tasks = [process_single_file(file) for file in files]
    results = await asyncio.gather(*tasks)
    
    processing_time = (datetime.utcnow() - start_time).total_seconds()
    
    return BatchDetectionResponse(
        success=True,
        batch_id=batch_id,
        total_images=len(files),
        processed_images=len([r for r in results if r["success"]]),
        failed_images=failed,
        results=results,
        processing_time=processing_time
    )

@app.post(f"{config.API_PREFIX}/detect/base64", response_model=DetectionResponse)
async def detect_disease_base64(
    image_base64: str = Form(...),
    confidence_threshold: float = Form(0.5, ge=0.0, le=1.0),
    filename: str = Form("base64_image.jpg")
):
    """Detect diseases from base64 encoded image"""
    start_time = datetime.utcnow()
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image_hash = calculate_image_hash(image_data)
        
        # Check cache
        cached_result = await get_cached_detection(image_hash)
        if cached_result:
            cached_result["cached"] = True
            return DetectionResponse(**cached_result)
        
        # Call ML model
        raw_results = await call_huggingface_api(image_data)
        formatted_results = format_detection_results(raw_results)
        
        # Filter by confidence threshold
        filtered_results = [
            result for result in formatted_results 
            if result.confidence >= confidence_threshold
        ]
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        response = DetectionResponse(
            success=True,
            image_id=f"base64_detection_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{image_hash[:8]}",
            results=filtered_results,
            processing_time=processing_time,
            model_version=config.MODEL_NAME,
            cached=False,
            confidence_threshold=confidence_threshold
        )
        
        # Cache result
        await cache_detection_result(image_hash, response.dict())
        
        return response
        
    except Exception as e:
        logger.error(f"Base64 detection error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Base64 detection failed: {str(e)}"
        )

@app.get(f"{config.API_PREFIX}/history")
async def get_detection_history(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """Get detection history"""
    records = db.query(DetectionRecord).order_by(
        DetectionRecord.created_at.desc()
    ).offset(offset).limit(limit).all()
    
    return {
        "total": db.query(DetectionRecord).count(),
        "records": [
            {
                "id": record.id,
                "filename": record.filename,
                "created_at": record.created_at,
                "processing_time": record.processing_time,
                "results": json.loads(record.detection_results) if record.detection_results else []
            }
            for record in records
        ]
    }

@app.get(f"{config.API_PREFIX}/metrics")
async def get_metrics():
    """Get Prometheus metrics"""
    return StreamingResponse(
        io.BytesIO(generate_latest()),
        media_type="text/plain"
    )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.utcnow(),
            request_id=str(uuid.uuid4())
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    ERROR_COUNT.labels(error_type="unhandled_exception").inc()
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc),
            timestamp=datetime.utcnow(),
            request_id=str(uuid.uuid4())
        ).dict()
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Enhanced AgroDrone API...")
    
    # Check dependencies
    if not config.HUGGINGFACE_API_TOKEN:
        logger.warning("Hugging Face API token not configured")
    
    if redis_client:
        logger.info("Redis connected successfully")
    
    if s3_client:
        logger.info("S3 client initialized")
    
    logger.info("API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Enhanced AgroDrone API...")
    
    if redis_client:
        redis_client.close()
    
    logger.info("API shutdown complete")

# Main entry point
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "enhanced_main:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level="info",
        workers=1 if os.getenv("RELOAD", "false").lower() == "true" else 4
    )
