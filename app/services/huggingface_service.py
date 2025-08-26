import base64
import json
import logging
from typing import Dict, List, Optional, Tuple
import aiohttp
from datetime import datetime
import asyncio

from app.core.config import settings
from app.models.schemas import DiseaseType, SeverityLevel, DetectionResponse

logger = logging.getLogger(__name__)


class HuggingFaceService:
    """Service for interacting with Hugging Face Inference API for plant disease detection."""
    
    def __init__(self):
        self.api_token = settings.HUGGINGFACE_API_TOKEN
        self.api_url = settings.HUGGINGFACE_API_URL
        self.model_name = settings.MODEL_NAME
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    async def detect_disease(self, image_data: str, filename: str) -> DetectionResponse:
        """
        Detect plant disease from image using Hugging Face API.
        
        Args:
            image_data: Base64 encoded image data
            filename: Original filename
            
        Returns:
            DetectionResponse with disease information
        """
        try:
            # Prepare the request payload
            payload = {
                "inputs": image_data
            }
            
            # Make API call to Hugging Face
            model_url = f"{self.api_url}/{self.model_name}"
            
            async with self.session.post(model_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._parse_detection_result(result, filename)
                else:
                    error_text = await response.text()
                    logger.error(f"Hugging Face API error: {response.status} - {error_text}")
                    return self._create_fallback_response(filename)
                    
        except Exception as e:
            logger.error(f"Error calling Hugging Face API: {str(e)}")
            return self._create_fallback_response(filename)
    
    def _parse_detection_result(self, result: Dict, filename: str) -> DetectionResponse:
        """Parse the Hugging Face API response into our schema."""
        # This is a mock implementation - adjust based on actual API response format
        # Real implementation would parse the actual model output
        
        # Mock disease detection based on filename patterns for demo
        disease_name = self._get_disease_from_filename(filename)
        confidence = 0.85  # Mock confidence
        
        return DetectionResponse(
            id=f"det_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(filename) % 10000}",
            success=True,
            disease_name=disease_name,
            disease_type=self._get_disease_type(disease_name),
            confidence=confidence,
            severity=self._get_severity_level(confidence),
            recommendations=self._get_recommendations(disease_name),
            timestamp=datetime.now(),
            image_url=None,
            metadata={
                "model": self.model_name,
                "api_source": "huggingface"
            }
        )
    
    def _create_fallback_response(self, filename: str) -> DetectionResponse:
        """Create a fallback response when API fails."""
        disease_name = "Healthy Plant"
        
        return DetectionResponse(
            id=f"det_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(filename) % 10000}",
            success=False,
            disease_name=disease_name,
            disease_type=DiseaseType.HEALTHY,
            confidence=0.7,
            severity=SeverityLevel.LOW,
            recommendations=["Monitor plant health regularly", "Ensure proper watering and nutrition"],
            timestamp=datetime.now(),
            image_url=None,
            metadata={
                "note": "Fallback response due to API failure",
                "model": self.model_name
            }
        )
    
    def _get_disease_from_filename(self, filename: str) -> str:
        """Extract disease name from filename for demo purposes."""
        filename_lower = filename.lower()
        
        disease_mapping = {
            "rust": "Wheat Rust",
            "blight": "Potato Blight",
            "mildew": "Powdery Mildew",
            "spot": "Leaf Spot",
            "rot": "Root Rot",
            "wilt": "Fusarium Wilt",
            "mosaic": "Tobacco Mosaic Virus",
            "healthy": "Healthy Plant"
        }
        
        for key, disease in disease_mapping.items():
            if key in filename_lower:
                return disease
        
        return "Healthy Plant"
    
    def _get_disease_type(self, disease_name: str) -> DiseaseType:
        """Determine disease type from disease name."""
        disease_name_lower = disease_name.lower()
        
        if "healthy" in disease_name_lower:
            return DiseaseType.HEALTHY
        elif "bacteria" in disease_name_lower or "bacterial" in disease_name_lower:
            return DiseaseType.BACTERIAL
        elif "fungus" in disease_name_lower or "fungal" in disease_name_lower or "mildew" in disease_name_lower or "rust" in disease_name_lower:
            return DiseaseType.FUNGAL
        elif "virus" in disease_name_lower or "mosaic" in disease_name_lower:
            return DiseaseType.VIRAL
        elif "nutrient" in disease_name_lower or "deficiency" in disease_name_lower:
            return DiseaseType.NUTRITIONAL
        else:
            return DiseaseType.ENVIRONMENTAL
    
    def _get_severity_level(self, confidence: float) -> SeverityLevel:
        """Determine severity level based on confidence."""
        if confidence >= 0.9:
            return SeverityLevel.CRITICAL
        elif confidence >= 0.8:
            return SeverityLevel.HIGH
        elif confidence >= 0.7:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def _get_recommendations(self, disease_name: str) -> List[str]:
        """Get treatment recommendations based on disease."""
        recommendations = {
            "Wheat Rust": [
                "Apply appropriate fungicide (e.g., propiconazole)",
                "Improve air circulation around plants",
                "Remove infected plant debris",
                "Consider resistant varieties for next planting"
            ],
            "Potato Blight": [
                "Apply copper-based fungicide",
                "Ensure proper drainage",
                "Avoid overhead watering",
                "Remove and destroy infected plants"
            ],
            "Powdery Mildew": [
                "Apply sulfur-based fungicide",
                "Increase air circulation",
                "Reduce humidity around plants",
                "Water at soil level, not on leaves"
            ],
            "Leaf Spot": [
                "Apply appropriate fungicide",
                "Remove infected leaves",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "Root Rot": [
                "Improve drainage",
                "Reduce watering frequency",
                "Apply fungicide to soil",
                "Consider raised beds for better drainage"
            ],
            "Fusarium Wilt": [
                "Remove and destroy infected plants",
                "Rotate crops",
                "Use disease-resistant varieties",
                "Improve soil drainage"
            ],
            "Tobacco Mosaic Virus": [
                "Remove infected plants immediately",
                "Disinfect tools between plants",
                "Control insect vectors",
                "Use virus-free seeds"
            ],
            "Healthy Plant": [
                "Continue regular care routine",
                "Monitor for early signs of disease",
                "Maintain proper nutrition",
                "Ensure adequate watering"
            ]
        }
        
        return recommendations.get(disease_name, ["Consult local agricultural extension office"])
