from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
import numpy as np
import torch
import cv2
import io
import json
import asyncio
import base64

from database import get_db, engine, Base
from models import DefectModel, YieldModel, ModelTrainingModel, DatasetModel
from schemas import (
    DefectDetectionRequest, DefectDetectionResponse,
    YieldPredictionRequest, YieldPredictionResponse,
    ModelTrainingRequest, ModelTrainingResponse,
    DatasetUploadRequest, DatasetResponse
)
from auth import get_current_user, create_access_token
from ml_engine import (
    DefectClassifier,
    YieldPredictor,
    DataAugmentation,
    ModelTrainer,
    ActiveLearning
)
from celery_worker import celery_app, train_model_task, batch_inference_task
import redis
import mlflow
import wandb

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize Redis
redis_client = redis.Redis(host='redis', port=6379, decode_responses=False)

# Initialize MLflow
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("defect_detection")

app = FastAPI(
    title="Defect Detection & Yield Prediction ML API",
    description="Advanced ML platform for semiconductor defect detection and yield optimization",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML models
defect_classifier = DefectClassifier()
yield_predictor = YieldPredictor()
data_augmentation = DataAugmentation()
model_trainer = ModelTrainer()
active_learning = ActiveLearning()

# WebSocket manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.get("/")
async def root():
    return {
        "message": "Defect Detection & Yield Prediction ML API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "detect": "/api/detect",
            "predict_yield": "/api/predict_yield",
            "train": "/api/train",
            "models": "/api/models",
            "datasets": "/api/datasets",
            "analytics": "/api/analytics"
        }
    }

@app.post("/api/detect", response_model=DefectDetectionResponse)
async def detect_defects(
    image: UploadFile = File(...),
    model_version: Optional[str] = None,
    confidence_threshold: float = 0.5,
    use_ensemble: bool = True,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Detect and classify defects in wafer/SEM images"""
    
    # Read image
    image_data = await image.read()
    img_array = cv2.imdecode(
        np.frombuffer(image_data, np.uint8),
        cv2.IMREAD_COLOR
    )
    
    # Run detection
    if use_ensemble:
        detections = defect_classifier.detect_ensemble(
            img_array,
            confidence_threshold=confidence_threshold,
            model_version=model_version
        )
    else:
        detections = defect_classifier.detect(
            img_array,
            confidence_threshold=confidence_threshold,
            model_version=model_version
        )
    
    # Calculate uncertainty
    uncertainty_scores = defect_classifier.calculate_uncertainty(detections)
    
    # Classify defect severity
    for detection, uncertainty in zip(detections, uncertainty_scores):
        detection['uncertainty'] = uncertainty
        detection['severity'] = defect_classifier.classify_severity(detection)
        detection['killer_defect'] = detection['severity'] > 0.8
    
    # Store in database
    defect_record = DefectModel(
        user_id=current_user.id,
        image_name=image.filename,
        detections=detections,
        total_defects=len(detections),
        confidence_threshold=confidence_threshold,
        model_version=model_version or "latest"
    )
    db.add(defect_record)
    db.commit()
    db.refresh(defect_record)
    
    # Generate visualization
    annotated_image = defect_classifier.visualize_detections(img_array, detections)
    
    # Encode image to base64
    _, buffer = cv2.imencode('.png', annotated_image)
    img_base64 = base64.b64encode(buffer).decode()
    
    # Calculate statistics
    stats = {
        'total_defects': len(detections),
        'by_class': {},
        'killer_defects': sum(1 for d in detections if d.get('killer_defect', False)),
        'mean_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0,
        'mean_uncertainty': np.mean(uncertainty_scores) if uncertainty_scores else 0
    }
    
    for detection in detections:
        class_name = detection['class']
        if class_name not in stats['by_class']:
            stats['by_class'][class_name] = 0
        stats['by_class'][class_name] += 1
    
    return DefectDetectionResponse(
        id=defect_record.id,
        detections=detections,
        statistics=stats,
        annotated_image=f"data:image/png;base64,{img_base64}",
        processing_time=defect_classifier.last_inference_time,
        model_version=model_version or "latest"
    )

@app.post("/api/detect/batch")
async def batch_detect(
    images: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Process multiple images in batch"""
    
    batch_id = f"batch_{datetime.now().timestamp()}"
    
    # Queue batch processing
    task = batch_inference_task.delay(
        batch_id=batch_id,
        image_count=len(images),
        user_id=current_user.id
    )
    
    # Store images temporarily
    for idx, image in enumerate(images):
        image_data = await image.read()
        redis_client.setex(
            f"batch:{batch_id}:image:{idx}",
            3600,  # 1 hour TTL
            image_data
        )
    
    return {
        "batch_id": batch_id,
        "task_id": task.id,
        "image_count": len(images),
        "status": "queued",
        "message": "Batch processing started"
    }

@app.post("/api/predict_yield", response_model=YieldPredictionResponse)
async def predict_yield(
    request: YieldPredictionRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Predict wafer/lot yield based on defect data and process parameters"""
    
    # Prepare features
    features = yield_predictor.prepare_features(
        defect_counts=request.defect_counts,
        process_params=request.process_parameters,
        historical_data=request.historical_data
    )
    
    # Make prediction
    if request.use_bayesian:
        prediction, uncertainty = yield_predictor.predict_bayesian(features)
        confidence_interval = yield_predictor.calculate_confidence_interval(
            prediction, uncertainty
        )
    else:
        prediction = yield_predictor.predict(features)
        uncertainty = None
        confidence_interval = None
    
    # Calculate risk score
    risk_score = yield_predictor.calculate_risk_score(
        prediction,
        request.defect_counts,
        uncertainty
    )
    
    # Generate recommendations
    recommendations = yield_predictor.generate_recommendations(
        prediction,
        risk_score,
        request.defect_counts,
        request.process_parameters
    )
    
    # Store prediction
    yield_record = YieldModel(
        user_id=current_user.id,
        lot_id=request.lot_id,
        predicted_yield=prediction,
        uncertainty=uncertainty,
        risk_score=risk_score,
        defect_counts=request.defect_counts,
        process_parameters=request.process_parameters,
        recommendations=recommendations
    )
    db.add(yield_record)
    db.commit()
    db.refresh(yield_record)
    
    return YieldPredictionResponse(
        id=yield_record.id,
        lot_id=request.lot_id,
        predicted_yield=prediction,
        uncertainty=uncertainty,
        confidence_interval=confidence_interval,
        risk_score=risk_score,
        recommendations=recommendations,
        feature_importance=yield_predictor.get_feature_importance()
    )

@app.post("/api/train", response_model=ModelTrainingResponse)
async def train_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Train a new defect detection or yield prediction model"""
    
    # Create training record
    training = ModelTrainingModel(
        user_id=current_user.id,
        model_type=request.model_type,
        dataset_id=request.dataset_id,
        hyperparameters=request.hyperparameters,
        status="pending"
    )
    db.add(training)
    db.commit()
    db.refresh(training)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{request.model_type}_{training.id}"):
        mlflow.log_params(request.hyperparameters)
        
        # Queue training task
        task = train_model_task.delay(
            training_id=training.id,
            model_type=request.model_type,
            dataset_id=request.dataset_id,
            hyperparameters=request.hyperparameters
        )
    
    # Store task ID
    redis_client.set(f"training:{training.id}:task", task.id)
    
    return ModelTrainingResponse(
        training_id=training.id,
        task_id=task.id,
        status="queued",
        estimated_time=model_trainer.estimate_training_time(
            request.model_type,
            request.dataset_id
        ),
        message="Model training started"
    )

@app.get("/api/models")
async def list_models(
    model_type: Optional[str] = None,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List available trained models"""
    
    query = db.query(ModelTrainingModel).filter(
        ModelTrainingModel.user_id == current_user.id,
        ModelTrainingModel.status == "completed"
    )
    
    if model_type:
        query = query.filter(ModelTrainingModel.model_type == model_type)
    
    models = query.all()
    
    # Get model metrics from MLflow
    model_info = []
    for model in models:
        mlflow_run = mlflow.get_run(model.mlflow_run_id) if model.mlflow_run_id else None
        
        info = {
            'id': model.id,
            'type': model.model_type,
            'created_at': model.created_at,
            'metrics': mlflow_run.data.metrics if mlflow_run else {},
            'parameters': model.hyperparameters,
            'dataset_id': model.dataset_id
        }
        model_info.append(info)
    
    return model_info

@app.post("/api/datasets/upload")
async def upload_dataset(
    dataset_name: str,
    images: List[UploadFile] = File(...),
    labels: Optional[UploadFile] = None,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a new dataset for training"""
    
    # Create dataset record
    dataset = DatasetModel(
        user_id=current_user.id,
        name=dataset_name,
        image_count=len(images),
        status="uploading"
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    
    # Process and store images
    dataset_path = f"/app/data/datasets/{dataset.id}"
    
    for idx, image in enumerate(images):
        image_data = await image.read()
        # Store in MinIO or local storage
        redis_client.setex(
            f"dataset:{dataset.id}:image:{idx}",
            7200,  # 2 hours TTL
            image_data
        )
    
    # Process labels if provided
    if labels:
        labels_data = await labels.read()
        redis_client.set(f"dataset:{dataset.id}:labels", labels_data)
    
    # Update dataset status
    dataset.status = "ready"
    db.commit()
    
    return {
        "dataset_id": dataset.id,
        "name": dataset_name,
        "image_count": len(images),
        "status": "ready",
        "message": "Dataset uploaded successfully"
    }

@app.get("/api/analytics/defects")
async def defect_analytics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get defect detection analytics"""
    
    query = db.query(DefectModel).filter(DefectModel.user_id == current_user.id)
    
    if start_date:
        query = query.filter(DefectModel.created_at >= start_date)
    if end_date:
        query = query.filter(DefectModel.created_at <= end_date)
    
    defects = query.all()
    
    # Calculate analytics
    total_images = len(defects)
    total_defects = sum(d.total_defects for d in defects)
    
    defect_distribution = {}
    confidence_distribution = []
    
    for record in defects:
        for detection in record.detections:
            class_name = detection['class']
            if class_name not in defect_distribution:
                defect_distribution[class_name] = 0
            defect_distribution[class_name] += 1
            confidence_distribution.append(detection['confidence'])
    
    return {
        'total_images_processed': total_images,
        'total_defects_detected': total_defects,
        'average_defects_per_image': total_defects / total_images if total_images > 0 else 0,
        'defect_distribution': defect_distribution,
        'confidence_stats': {
            'mean': np.mean(confidence_distribution) if confidence_distribution else 0,
            'std': np.std(confidence_distribution) if confidence_distribution else 0,
            'min': np.min(confidence_distribution) if confidence_distribution else 0,
            'max': np.max(confidence_distribution) if confidence_distribution else 0
        },
        'time_series': defect_classifier.generate_time_series(defects)
    }

@app.get("/api/analytics/yield")
async def yield_analytics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get yield prediction analytics"""
    
    query = db.query(YieldModel).filter(YieldModel.user_id == current_user.id)
    
    if start_date:
        query = query.filter(YieldModel.created_at >= start_date)
    if end_date:
        query = query.filter(YieldModel.created_at <= end_date)
    
    yields = query.all()
    
    # Calculate analytics
    predictions = [y.predicted_yield for y in yields]
    risk_scores = [y.risk_score for y in yields]
    
    return {
        'total_predictions': len(yields),
        'yield_stats': {
            'mean': np.mean(predictions) if predictions else 0,
            'std': np.std(predictions) if predictions else 0,
            'min': np.min(predictions) if predictions else 0,
            'max': np.max(predictions) if predictions else 0
        },
        'risk_distribution': {
            'low': sum(1 for r in risk_scores if r < 0.3),
            'medium': sum(1 for r in risk_scores if 0.3 <= r < 0.7),
            'high': sum(1 for r in risk_scores if r >= 0.7)
        },
        'trend_analysis': yield_predictor.analyze_trends(yields),
        'correlation_matrix': yield_predictor.calculate_correlations(yields)
    }

@app.post("/api/active_learning/suggest")
async def suggest_for_labeling(
    model_id: int,
    sample_size: int = 10,
    strategy: str = "uncertainty",
    current_user = Depends(get_current_user)
):
    """Suggest samples for active learning"""
    
    suggestions = active_learning.suggest_samples(
        model_id=model_id,
        sample_size=sample_size,
        strategy=strategy
    )
    
    return {
        'suggestions': suggestions,
        'strategy': strategy,
        'expected_improvement': active_learning.estimate_improvement(suggestions)
    }

@app.websocket("/ws/training/{training_id}")
async def websocket_training_updates(websocket: WebSocket, training_id: int):
    """WebSocket for real-time training updates"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Get training status from Redis
            status = redis_client.get(f"training:{training_id}:status")
            progress = redis_client.get(f"training:{training_id}:progress")
            metrics = redis_client.get(f"training:{training_id}:metrics")
            
            if status:
                update = {
                    "training_id": training_id,
                    "status": status.decode() if status else "unknown",
                    "progress": float(progress.decode()) if progress else 0,
                    "metrics": json.loads(metrics.decode()) if metrics else {}
                }
                
                await manager.send_personal_message(
                    json.dumps(update),
                    websocket
                )
            
            if status and status.decode() in ["completed", "failed"]:
                break
                
            await asyncio.sleep(1)
            
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        manager.disconnect(websocket)

@app.get("/api/model/explain/{model_id}")
async def explain_model(
    model_id: int,
    image: UploadFile = File(...),
    current_user = Depends(get_current_user)
):
    """Generate model explanations using SHAP/GradCAM"""
    
    # Read image
    image_data = await image.read()
    img_array = cv2.imdecode(
        np.frombuffer(image_data, np.uint8),
        cv2.IMREAD_COLOR
    )
    
    # Generate explanations
    explanations = defect_classifier.explain_prediction(
        img_array,
        model_id=model_id,
        methods=['gradcam', 'shap', 'lime']
    )
    
    return {
        'explanations': explanations,
        'feature_importance': defect_classifier.get_feature_importance(model_id),
        'attention_maps': explanations.get('gradcam', {})
    }

@app.post("/api/optimize/hyperparameters")
async def optimize_hyperparameters(
    model_type: str,
    dataset_id: int,
    n_trials: int = 20,
    current_user = Depends(get_current_user)
):
    """Optimize model hyperparameters using Optuna"""
    
    study = model_trainer.optimize_hyperparameters(
        model_type=model_type,
        dataset_id=dataset_id,
        n_trials=n_trials
    )
    
    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials),
        'optimization_history': [
            {'trial': i, 'value': trial.value}
            for i, trial in enumerate(study.trials)
        ]
    }

@app.get("/api/export/model/{model_id}")
async def export_model(
    model_id: int,
    format: str = "onnx",
    current_user = Depends(get_current_user)
):
    """Export trained model in various formats"""
    
    export_path = model_trainer.export_model(
        model_id=model_id,
        format=format
    )
    
    # Return file
    with open(export_path, 'rb') as f:
        content = f.read()
    
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/octet-stream",
        headers={
            f"Content-Disposition": f"attachment; filename=model_{model_id}.{format}"
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)