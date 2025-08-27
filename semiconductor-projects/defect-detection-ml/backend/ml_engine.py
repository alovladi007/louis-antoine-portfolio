import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
import time

class DefectClassifier:
    """CNN-based defect classification system"""
    
    def __init__(self):
        self.model = self._load_model()
        self.classes = ['particle', 'scratch', 'pattern_defect', 'contamination', 'void', 'bridge', 'missing_feature']
        self.last_inference_time = 0
        
    def _load_model(self):
        """Load pre-trained ResNet model"""
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(self.classes))
        model.eval()
        return model
    
    def detect(self, image: np.ndarray, confidence_threshold: float = 0.5, model_version: str = None) -> List[Dict]:
        """Detect defects in image"""
        start_time = time.time()
        
        # Preprocess image
        processed = self._preprocess_image(image)
        
        # Run detection
        with torch.no_grad():
            outputs = self.model(processed)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Extract detections
        detections = []
        for i, prob in enumerate(probabilities[0]):
            if prob > confidence_threshold:
                detections.append({
                    'class': self.classes[i],
                    'confidence': float(prob),
                    'bbox': self._generate_bbox(image.shape),
                    'area': np.random.randint(10, 100)
                })
        
        self.last_inference_time = time.time() - start_time
        return detections
    
    def detect_ensemble(self, image: np.ndarray, confidence_threshold: float = 0.5, model_version: str = None) -> List[Dict]:
        """Ensemble detection for better accuracy"""
        # Simplified ensemble - in production would use multiple models
        detections1 = self.detect(image, confidence_threshold)
        detections2 = self.detect(cv2.flip(image, 1), confidence_threshold)  # Horizontal flip
        
        # Merge detections
        all_detections = detections1 + detections2
        return self._nms(all_detections, 0.5)
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        resized = cv2.resize(image, (224, 224))
        normalized = resized / 255.0
        tensor = torch.from_numpy(normalized).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor
    
    def _generate_bbox(self, shape: Tuple) -> Dict:
        """Generate random bounding box for demo"""
        h, w = shape[:2]
        x = np.random.randint(0, w-50)
        y = np.random.randint(0, h-50)
        return {
            'x': int(x),
            'y': int(y),
            'width': np.random.randint(20, 50),
            'height': np.random.randint(20, 50)
        }
    
    def _nms(self, detections: List[Dict], threshold: float) -> List[Dict]:
        """Non-maximum suppression"""
        # Simplified NMS
        return detections[:min(len(detections), 10)]
    
    def calculate_uncertainty(self, detections: List[Dict]) -> List[float]:
        """Calculate uncertainty scores"""
        return [1 - d['confidence'] for d in detections]
    
    def classify_severity(self, detection: Dict) -> float:
        """Classify defect severity"""
        severity_map = {
            'particle': 0.6,
            'scratch': 0.7,
            'pattern_defect': 0.9,
            'contamination': 0.5,
            'void': 0.8,
            'bridge': 0.85,
            'missing_feature': 0.95
        }
        return severity_map.get(detection['class'], 0.5) * detection['confidence']
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Visualize detections on image"""
        result = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            color = (0, 255, 0) if det.get('killer_defect', False) else (255, 0, 0)
            cv2.rectangle(result, 
                         (bbox['x'], bbox['y']), 
                         (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                         color, 2)
            cv2.putText(result, f"{det['class']}: {det['confidence']:.2f}",
                       (bbox['x'], bbox['y'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    def explain_prediction(self, image: np.ndarray, model_id: int, methods: List[str]) -> Dict:
        """Generate model explanations"""
        explanations = {}
        
        if 'gradcam' in methods:
            explanations['gradcam'] = self._generate_gradcam(image)
        if 'shap' in methods:
            explanations['shap'] = self._generate_shap(image)
        if 'lime' in methods:
            explanations['lime'] = self._generate_lime(image)
        
        return explanations
    
    def _generate_gradcam(self, image: np.ndarray) -> np.ndarray:
        """Generate GradCAM heatmap"""
        # Simplified - would implement actual GradCAM
        return np.random.rand(*image.shape[:2])
    
    def _generate_shap(self, image: np.ndarray) -> np.ndarray:
        """Generate SHAP values"""
        return np.random.rand(*image.shape[:2])
    
    def _generate_lime(self, image: np.ndarray) -> np.ndarray:
        """Generate LIME explanation"""
        return np.random.rand(*image.shape[:2])
    
    def get_feature_importance(self, model_id: int = None) -> Dict:
        """Get feature importance"""
        return {
            'spatial_features': 0.3,
            'texture_features': 0.25,
            'edge_features': 0.2,
            'color_features': 0.15,
            'shape_features': 0.1
        }
    
    def generate_time_series(self, defects: List) -> List[Dict]:
        """Generate time series data"""
        return [
            {
                'timestamp': defect.created_at.isoformat(),
                'count': defect.total_defects
            }
            for defect in defects
        ]


class YieldPredictor:
    """Yield prediction using ML"""
    
    def __init__(self):
        self.model = self._load_model()
        self.feature_names = ['defect_density', 'killer_defects', 'process_temp', 'process_pressure', 'exposure_dose']
        
    def _load_model(self):
        """Load yield prediction model"""
        # Simplified - would load actual trained model
        return None
    
    def prepare_features(self, defect_counts: Dict, process_params: Dict, historical_data: Optional[List] = None) -> np.ndarray:
        """Prepare features for prediction"""
        features = []
        
        # Defect features
        total_defects = sum(defect_counts.values())
        features.append(total_defects / 1000)  # Normalized defect density
        features.append(defect_counts.get('killer', 0))
        
        # Process parameters
        features.append(process_params.get('temperature', 25))
        features.append(process_params.get('pressure', 1.0))
        features.append(process_params.get('exposure', 1.0))
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, features: np.ndarray) -> float:
        """Predict yield"""
        # Simplified prediction
        base_yield = 95.0
        defect_impact = features[0, 0] * 10  # Defect density impact
        process_impact = (features[0, 2] - 25) * 0.1  # Temperature impact
        
        predicted_yield = base_yield - defect_impact - process_impact
        return max(0, min(100, predicted_yield))
    
    def predict_bayesian(self, features: np.ndarray) -> Tuple[float, float]:
        """Bayesian prediction with uncertainty"""
        prediction = self.predict(features)
        uncertainty = np.random.uniform(1, 5)  # Simplified uncertainty
        return prediction, uncertainty
    
    def calculate_confidence_interval(self, prediction: float, uncertainty: float) -> Tuple[float, float]:
        """Calculate confidence interval"""
        lower = max(0, prediction - 2 * uncertainty)
        upper = min(100, prediction + 2 * uncertainty)
        return lower, upper
    
    def calculate_risk_score(self, prediction: float, defect_counts: Dict, uncertainty: Optional[float]) -> float:
        """Calculate yield risk score"""
        base_risk = (100 - prediction) / 100
        defect_risk = sum(defect_counts.values()) / 1000
        uncertainty_factor = (uncertainty / 10) if uncertainty else 0
        
        return min(1.0, base_risk + defect_risk * 0.3 + uncertainty_factor)
    
    def generate_recommendations(self, prediction: float, risk_score: float, defect_counts: Dict, process_params: Dict) -> List[str]:
        """Generate process recommendations"""
        recommendations = []
        
        if prediction < 90:
            recommendations.append("Consider additional inspection steps")
        
        if risk_score > 0.7:
            recommendations.append("High risk detected - recommend process review")
        
        if defect_counts.get('particle', 0) > 10:
            recommendations.append("Increase cleanroom filtration")
        
        if process_params.get('temperature', 25) > 30:
            recommendations.append("Reduce process temperature for better yield")
        
        return recommendations
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance"""
        return {
            'defect_density': 0.35,
            'killer_defects': 0.25,
            'process_temp': 0.15,
            'process_pressure': 0.15,
            'exposure_dose': 0.1
        }
    
    def analyze_trends(self, yields: List) -> Dict:
        """Analyze yield trends"""
        predictions = [y.predicted_yield for y in yields]
        
        return {
            'trend': 'improving' if len(predictions) > 1 and predictions[-1] > predictions[0] else 'declining',
            'volatility': np.std(predictions) if predictions else 0,
            'average': np.mean(predictions) if predictions else 0
        }
    
    def calculate_correlations(self, yields: List) -> Dict:
        """Calculate correlations between parameters"""
        # Simplified correlation matrix
        return {
            'defects_vs_yield': -0.85,
            'temperature_vs_yield': -0.3,
            'pressure_vs_yield': 0.1
        }


class DataAugmentation:
    """Data augmentation for training"""
    
    def augment(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply augmentations"""
        augmented = [image]
        
        # Flip
        augmented.append(cv2.flip(image, 1))
        augmented.append(cv2.flip(image, 0))
        
        # Rotate
        for angle in [90, 180, 270]:
            M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1)
            rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            augmented.append(rotated)
        
        # Brightness
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        augmented.append(bright)
        
        return augmented


class ModelTrainer:
    """Model training utilities"""
    
    def estimate_training_time(self, model_type: str, dataset_id: int) -> int:
        """Estimate training time in minutes"""
        base_time = {'defect_detection': 60, 'yield_prediction': 30}
        return base_time.get(model_type, 45)
    
    def optimize_hyperparameters(self, model_type: str, dataset_id: int, n_trials: int) -> Any:
        """Optimize hyperparameters using Optuna"""
        # Simplified - would use actual Optuna optimization
        class Study:
            best_params = {'learning_rate': 0.001, 'batch_size': 32}
            best_value = 0.95
            trials = [None] * n_trials
        
        return Study()
    
    def export_model(self, model_id: int, format: str) -> str:
        """Export model to specified format"""
        # Create dummy model file
        export_path = f"/tmp/model_{model_id}.{format}"
        with open(export_path, 'wb') as f:
            f.write(b"Model data")
        return export_path


class ActiveLearning:
    """Active learning for data efficiency"""
    
    def suggest_samples(self, model_id: int, sample_size: int, strategy: str) -> List[Dict]:
        """Suggest samples for labeling"""
        suggestions = []
        
        for i in range(sample_size):
            suggestions.append({
                'sample_id': f"sample_{i}",
                'uncertainty': np.random.random(),
                'predicted_class': np.random.choice(['particle', 'scratch', 'void']),
                'confidence': np.random.uniform(0.4, 0.6)
            })
        
        # Sort by uncertainty
        suggestions.sort(key=lambda x: x['uncertainty'], reverse=True)
        return suggestions
    
    def estimate_improvement(self, suggestions: List[Dict]) -> float:
        """Estimate expected model improvement"""
        avg_uncertainty = np.mean([s['uncertainty'] for s in suggestions])
        return avg_uncertainty * 0.1  # Simplified estimation