from pydantic import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # API Configuration
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    RELOAD: bool = os.getenv("RELOAD", "true").lower() == "true"
    
    # Hugging Face Configuration
    HUGGINGFACE_API_TOKEN: str = os.getenv("HUGGINGFACE_API_TOKEN", "")
    HUGGINGFACE_API_URL: str = os.getenv("HUGGINGFACE_API_URL", "https://api-inference.huggingface.co/models")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "plant-disease-detection")
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    
    # Security Configuration
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./agrodrone.db")
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/agrodrone.log")
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:19006").split(",")
    
    # Storage Configuration
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    MAX_UPLOADS_PER_DAY: int = int(os.getenv("MAX_UPLOADS_PER_DAY", "100"))
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
