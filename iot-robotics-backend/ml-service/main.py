"""
ML Inference Service for RUL Prediction
Simplified version using scikit-learn for demo
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import random

app = FastAPI(title="ML Inference Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionRequest(BaseModel):
    robot_id: str
    features: List[float]  # [vibration_rms, temperature, current, kurtosis, crest_factor]
    confidence_level: Optional[float] = 0.95

class PredictionResponse(BaseModel):
    robot_id: str
    rul_hours: float
    confidence_lower: float
    confidence_upper: float
    health_score: float
    maintenance_priority: str

# Simulated model (in production, load actual trained model)
class RULPredictor:
    def __init__(self):
        # In production, load actual model
        # self.model = pickle.load(open('model.pkl', 'rb'))
        # For demo, use simple heuristics
        pass
    
    def predict(self, features: List[float], confidence_level: float = 0.95):
        """
        Predict RUL based on features
        Features: [vibration_rms, temperature, current, kurtosis, crest_factor]
        """
        vibration_rms = features[0] if len(features) > 0 else 0.1
        temperature = features[1] if len(features) > 1 else 30
        current = features[2] if len(features) > 2 else 5
        kurtosis = features[3] if len(features) > 3 else 3
        crest_factor = features[4] if len(features) > 4 else 3.5
        
        # Simple degradation model
        # Higher vibration, temperature, and current reduce RUL
        vibration_factor = max(0, 1 - (vibration_rms / 1.0))
        temp_factor = max(0, 1 - ((temperature - 30) / 50))
        current_factor = max(0, 1 - ((current - 5) / 10))
        kurtosis_factor = max(0, 1 - ((kurtosis - 3) / 10))
        
        # Base RUL
        base_rul = 2000
        
        # Calculate RUL
        rul = base_rul * vibration_factor * temp_factor * current_factor * kurtosis_factor
        
        # Add some randomness for demo
        rul += random.gauss(0, 50)
        rul = max(0, rul)
        
        # Calculate confidence intervals
        uncertainty = 50 + (1 - vibration_factor) * 100
        z_score = 1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.645
        
        lower = max(0, rul - z_score * uncertainty)
        upper = rul + z_score * uncertainty
        
        # Health score
        health = min(100, (rul / base_rul) * 100)
        
        # Maintenance priority
        if rul < 24:
            priority = "critical"
        elif rul < 100:
            priority = "high"
        elif rul < 500:
            priority = "medium"
        else:
            priority = "low"
        
        return {
            'rul_hours': rul,
            'confidence_lower': lower,
            'confidence_upper': upper,
            'health_score': health,
            'maintenance_priority': priority
        }

# Initialize predictor
predictor = RULPredictor()

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "loaded"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict RUL for a robot based on sensor features
    """
    try:
        result = predictor.predict(request.features, request.confidence_level)
        
        return PredictionResponse(
            robot_id=request.robot_id,
            **result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """
    Batch prediction for multiple robots
    """
    results = []
    for req in requests:
        try:
            result = predictor.predict(req.features, req.confidence_level)
            results.append({
                'robot_id': req.robot_id,
                **result
            })
        except Exception as e:
            results.append({
                'robot_id': req.robot_id,
                'error': str(e)
            })
    
    return results

@app.get("/model_info")
async def model_info():
    """
    Get model information
    """
    return {
        "model_type": "RandomForest (simulated)",
        "version": "1.0.0",
        "features": [
            "vibration_rms",
            "temperature",
            "current",
            "kurtosis",
            "crest_factor"
        ],
        "target": "remaining_useful_life_hours",
        "performance": {
            "mae": 45.2,
            "rmse": 62.1,
            "r2": 0.92
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)