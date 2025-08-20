from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Redis
    REDIS_URL: str = "redis://redis:6379/0"
    
    # S3/MinIO
    S3_ENDPOINT: str = "http://minio:9000"
    S3_ACCESS_KEY: str = "medimetrics"
    S3_SECRET_KEY: str = "medimetricssecret"
    S3_BUCKET_RAW: str = "medimetrics-raw"
    S3_BUCKET_DERIV: str = "medimetrics-derivatives"
    S3_REGION: str = "us-east-1"
    
    # Webhook
    WEBHOOK_URL: Optional[str] = "http://api:8000/inference/webhook"
    WEBHOOK_HMAC_SECRET: str = "change-me-webhook-hmac-secret"
    
    # Model settings
    MODEL_REGISTRY_PATH: str = "/app/models"
    GPU_ENABLED: bool = False
    DEVICE: str = "cpu"  # Will be set to "cuda" if GPU available
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 9200
    METRICS_PORT: int = 9201
    
    # Processing settings
    MAX_IMAGE_SIZE: int = 4096
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 2
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()