from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import redis
from rq import Queue
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import PlainTextResponse
import logging
import time
from typing import Optional
from pydantic import BaseModel
import hashlib
import hmac
import httpx
import json
import os

from ..pipeline.registry import ModelRegistry
from ..worker.jobs import process_inference_job
from .settings import Settings

# Initialize settings
settings = Settings()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
inference_requests = Counter('inference_requests_total', 'Total inference requests')
inference_duration = Histogram('inference_duration_seconds', 'Inference duration')
active_jobs = Gauge('active_jobs', 'Number of active inference jobs')
model_loads = Counter('model_loads_total', 'Total model loads', ['model_name'])

# Initialize services
redis_client = redis.from_url(settings.REDIS_URL)
job_queue = Queue('inference', connection=redis_client)
model_registry = ModelRegistry()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting inference service...")
    # Load models on startup
    model_registry.load_default_models()
    yield
    logger.info("Shutting down inference service...")

app = FastAPI(
    title="MediMetrics Inference Service",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    study_id: str
    model: str = "classifier"
    options: dict = {}

class InferenceResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    result: Optional[dict] = None
    error: Optional[str] = None

class WebhookPayload(BaseModel):
    job_id: str
    study_id: str
    status: str
    results: Optional[dict] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    return {
        "service": "MediMetrics Inference",
        "version": "1.0.0",
        "status": "healthy",
        "models": model_registry.list_models()
    }

@app.get("/health")
async def health_check():
    try:
        # Check Redis
        redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/v1/jobs", response_model=InferenceResponse)
async def create_inference_job(
    request: InferenceRequest,
    background_tasks: BackgroundTasks
):
    """Create a new inference job"""
    inference_requests.inc()
    
    try:
        # Validate model exists
        if request.model not in model_registry.list_models():
            raise HTTPException(status_code=400, detail=f"Model '{request.model}' not found")
        
        # Enqueue job
        job = job_queue.enqueue(
            process_inference_job,
            study_id=request.study_id,
            model_name=request.model,
            options=request.options,
            job_timeout='10m',
            result_ttl='24h'
        )
        
        active_jobs.inc()
        
        return InferenceResponse(
            job_id=job.id,
            status="queued",
            message=f"Inference job queued for study {request.study_id}"
        )
    
    except Exception as e:
        logger.error(f"Failed to create job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of an inference job"""
    try:
        from rq.job import Job
        job = Job.fetch(job_id, connection=redis_client)
        
        if job.is_finished:
            active_jobs.dec()
            return JobStatus(
                job_id=job_id,
                status="completed",
                progress=1.0,
                result=job.result
            )
        elif job.is_failed:
            active_jobs.dec()
            return JobStatus(
                job_id=job_id,
                status="failed",
                progress=0.0,
                error=str(job.exc_info)
            )
        elif job.is_started:
            return JobStatus(
                job_id=job_id,
                status="running",
                progress=0.5
            )
        else:
            return JobStatus(
                job_id=job_id,
                status="queued",
                progress=0.0
            )
    
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=404, detail="Job not found")

@app.post("/v1/webhook")
async def send_webhook(payload: WebhookPayload):
    """Send webhook to NestJS API on job completion"""
    webhook_url = settings.WEBHOOK_URL
    webhook_secret = settings.WEBHOOK_HMAC_SECRET
    
    if not webhook_url:
        logger.warning("Webhook URL not configured")
        return {"status": "skipped"}
    
    try:
        # Create HMAC signature
        payload_json = payload.json()
        signature = hmac.new(
            webhook_secret.encode(),
            payload_json.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Send webhook
        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook_url,
                json=payload.dict(),
                headers={
                    "X-Signature": signature,
                    "Content-Type": "application/json"
                },
                timeout=10.0
            )
            response.raise_for_status()
            
        return {"status": "sent", "response": response.status_code}
    
    except Exception as e:
        logger.error(f"Failed to send webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/v1/models")
async def list_models():
    """List available models"""
    models = model_registry.list_models()
    return {
        "models": [
            {
                "name": name,
                "type": info.get("type"),
                "version": info.get("version"),
                "description": info.get("description")
            }
            for name, info in models.items()
        ]
    }

@app.post("/v1/models/{model_name}/load")
async def load_model(model_name: str):
    """Explicitly load a model into memory"""
    try:
        model_registry.get_model(model_name)
        model_loads.labels(model_name=model_name).inc()
        return {"status": "loaded", "model": model_name}
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9200)