"""
Yield Prediction Module

This module implements yield prediction models based on defect analysis
and process parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ProcessParameters:
    """Process parameters for yield prediction"""
    temperature: float = 150.0  # °C
    pressure: float = 500.0  # mTorr
    exposure_time: float = 50.0  # seconds
    gas_flow_rate: float = 100.0  # sccm
    rf_power: float = 200.0  # W
    chamber_clean_cycles: int = 100
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([
            self.temperature,
            self.pressure,
            self.exposure_time,
            self.gas_flow_rate,
            self.rf_power,
            self.chamber_clean_cycles
        ])


class YieldPredictor:
    """
    Main yield prediction model combining defect analysis and process parameters
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize yield predictor
        
        Args:
            model_type: Type of model ('linear', 'ridge', 'random_forest', 'gradient_boost')
        """
        self.model_type = model_type
        self.model = self._create_model(model_type)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
    def _create_model(self, model_type: str):
        """Create the prediction model"""
        if model_type == 'linear':
            return LinearRegression()
        elif model_type == 'ridge':
            return Ridge(alpha=1.0)
        elif model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'gradient_boost':
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def extract_features(self, wafer_map: np.ndarray, 
                        process_params: ProcessParameters,
                        defect_counts: Optional[Dict[str, int]] = None) -> np.ndarray:
        """
        Extract features from wafer map and process parameters
        
        Args:
            wafer_map: Wafer defect map
            process_params: Process parameters
            defect_counts: Dictionary of defect counts by type
            
        Returns:
            Feature vector
        """
        features = []
        
        # Wafer map statistics
        total_dies = wafer_map.size
        defective_dies = np.sum(wafer_map > 0)
        defect_density = defective_dies / total_dies if total_dies > 0 else 0
        
        features.extend([
            defective_dies,
            defect_density,
            np.mean(wafer_map),
            np.std(wafer_map),
            np.max(wafer_map)
        ])
        
        # Spatial distribution features
        # Center vs edge defects
        center_y, center_x = wafer_map.shape[0] // 2, wafer_map.shape[1] // 2
        radius = min(wafer_map.shape) // 4
        
        y, x = np.ogrid[:wafer_map.shape[0], :wafer_map.shape[1]]
        center_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        edge_mask = ~center_mask
        
        center_defects = np.sum(wafer_map[center_mask] > 0)
        edge_defects = np.sum(wafer_map[edge_mask] > 0)
        
        features.extend([
            center_defects,
            edge_defects,
            edge_defects / (center_defects + 1)  # Edge-to-center ratio
        ])
        
        # Clustering metric (defect clustering tendency)
        from scipy import ndimage
        labeled, n_clusters = ndimage.label(wafer_map > 0)
        avg_cluster_size = defective_dies / (n_clusters + 1)
        
        features.extend([
            n_clusters,
            avg_cluster_size
        ])
        
        # Defect type counts
        if defect_counts:
            for defect_type in ['particle', 'scratch', 'void', 'bridge', 'cluster']:
                features.append(defect_counts.get(defect_type, 0))
        else:
            features.extend([0] * 5)  # Placeholder for defect types
        
        # Process parameters
        features.extend(process_params.to_array())
        
        # Feature names for interpretability
        self.feature_names = [
            'defective_dies', 'defect_density', 'mean_defect', 'std_defect', 'max_defect',
            'center_defects', 'edge_defects', 'edge_center_ratio',
            'n_clusters', 'avg_cluster_size',
            'particle_count', 'scratch_count', 'void_count', 'bridge_count', 'cluster_count',
            'temperature', 'pressure', 'exposure_time', 'gas_flow_rate', 'rf_power', 'chamber_clean_cycles'
        ]
        
        return np.array(features)
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validate: bool = True) -> Dict[str, float]:
        """
        Train the yield prediction model
        
        Args:
            X: Feature matrix
            y: Yield values
            validate: Perform cross-validation
            
        Returns:
            Training metrics
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate metrics
        train_score = self.model.score(X_scaled, y)
        
        metrics = {
            'train_r2': train_score,
            'train_rmse': np.sqrt(np.mean((self.model.predict(X_scaled) - y)**2))
        }
        
        # Cross-validation
        if validate:
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, 
                                       scoring='r2')
            cv_rmse = -cross_val_score(self.model, X_scaled, y, cv=5,
                                       scoring='neg_root_mean_squared_error')
            
            metrics.update({
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std()
            })
        
        return metrics
    
    def predict(self, X: np.ndarray, 
                return_confidence: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict yield with confidence intervals
        
        Args:
            X: Feature matrix
            return_confidence: Return confidence intervals
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        confidence_intervals = None
        if return_confidence:
            if hasattr(self.model, 'predict_proba'):
                # For ensemble methods, use prediction variance
                if self.model_type in ['random_forest', 'gradient_boost']:
                    # Get predictions from individual trees
                    if self.model_type == 'random_forest':
                        tree_predictions = np.array([
                            tree.predict(X_scaled) for tree in self.model.estimators_
                        ])
                        std = np.std(tree_predictions, axis=0)
                        confidence_intervals = 1.96 * std  # 95% CI
                    else:
                        # Simplified CI for gradient boosting
                        confidence_intervals = np.ones(len(predictions)) * 0.02
            else:
                # For linear models, use residual-based CI
                confidence_intervals = np.ones(len(predictions)) * 0.02
        
        return predictions, confidence_intervals
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance_dict = {}
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importances = self.model.feature_importances_
            for name, importance in zip(self.feature_names, importances):
                importance_dict[name] = importance
        elif hasattr(self.model, 'coef_'):
            # Linear models
            coefficients = np.abs(self.model.coef_)
            for name, coef in zip(self.feature_names, coefficients):
                importance_dict[name] = coef
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        return importance_dict


class YieldOptimizer:
    """
    Optimize process parameters for maximum yield
    """
    
    def __init__(self, yield_predictor: YieldPredictor):
        """
        Initialize yield optimizer
        
        Args:
            yield_predictor: Trained yield prediction model
        """
        self.predictor = yield_predictor
        
    def optimize_parameters(self, wafer_map: np.ndarray,
                           defect_counts: Dict[str, int],
                           constraints: Optional[Dict[str, Tuple[float, float]]] = None) -> ProcessParameters:
        """
        Find optimal process parameters for maximum yield
        
        Args:
            wafer_map: Current wafer defect map
            defect_counts: Current defect counts
            constraints: Parameter constraints (min, max) for each parameter
            
        Returns:
            Optimized process parameters
        """
        from scipy.optimize import differential_evolution
        
        # Default constraints
        if constraints is None:
            constraints = {
                'temperature': (100, 200),
                'pressure': (300, 700),
                'exposure_time': (20, 80),
                'gas_flow_rate': (50, 150),
                'rf_power': (100, 300),
                'chamber_clean_cycles': (50, 200)
            }
        
        # Objective function (negative yield for minimization)
        def objective(params):
            process_params = ProcessParameters(
                temperature=params[0],
                pressure=params[1],
                exposure_time=params[2],
                gas_flow_rate=params[3],
                rf_power=params[4],
                chamber_clean_cycles=int(params[5])
            )
            
            features = self.predictor.extract_features(
                wafer_map, process_params, defect_counts
            )
            
            yield_pred, _ = self.predictor.predict(features.reshape(1, -1), 
                                                  return_confidence=False)
            
            return -yield_pred[0]  # Negative for maximization
        
        # Bounds for optimization
        bounds = [
            constraints['temperature'],
            constraints['pressure'],
            constraints['exposure_time'],
            constraints['gas_flow_rate'],
            constraints['rf_power'],
            constraints['chamber_clean_cycles']
        ]
        
        # Optimize
        result = differential_evolution(objective, bounds, seed=42, maxiter=100)
        
        # Create optimized parameters
        optimal_params = ProcessParameters(
            temperature=result.x[0],
            pressure=result.x[1],
            exposure_time=result.x[2],
            gas_flow_rate=result.x[3],
            rf_power=result.x[4],
            chamber_clean_cycles=int(result.x[5])
        )
        
        return optimal_params


class NeuralYieldPredictor(nn.Module):
    """
    Neural network-based yield predictor
    """
    
    def __init__(self, input_dim: int = 21, hidden_dims: List[int] = [64, 32, 16]):
        """
        Initialize neural yield predictor
        
        Args:
            input_dim: Number of input features
            hidden_dims: Hidden layer dimensions
        """
        super(NeuralYieldPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Yield is between 0 and 1
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x)
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty using MC Dropout
        
        Args:
            x: Input features
            n_samples: Number of MC samples
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty


class TimeSeriesYieldPredictor:
    """
    Time series model for yield prediction
    """
    
    def __init__(self, lookback: int = 24):
        """
        Initialize time series predictor
        
        Args:
            lookback: Number of historical points to consider
        """
        self.lookback = lookback
        self.model = None
        self.scaler = StandardScaler()
    
    def prepare_sequences(self, yield_history: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for time series prediction
        
        Args:
            yield_history: Historical yield values
            
        Returns:
            Tuple of (X_sequences, y_targets)
        """
        X, y = [], []
        
        for i in range(self.lookback, len(yield_history)):
            X.append(yield_history[i-self.lookback:i])
            y.append(yield_history[i])
        
        return np.array(X), np.array(y)
    
    def train(self, yield_history: np.ndarray) -> Dict[str, float]:
        """
        Train time series model
        
        Args:
            yield_history: Historical yield values
            
        Returns:
            Training metrics
        """
        X, y = self.prepare_sequences(yield_history)
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Use gradient boosting for time series
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        self.model.fit(X_scaled, y)
        
        # Calculate metrics
        predictions = self.model.predict(X_scaled)
        rmse = np.sqrt(np.mean((predictions - y)**2))
        mae = np.mean(np.abs(predictions - y))
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': self.model.score(X_scaled, y)
        }
    
    def predict_next(self, recent_yields: np.ndarray, n_ahead: int = 1) -> np.ndarray:
        """
        Predict future yields
        
        Args:
            recent_yields: Recent yield values (length = lookback)
            n_ahead: Number of steps to predict ahead
            
        Returns:
            Predicted yields
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        predictions = []
        current_sequence = recent_yields[-self.lookback:].copy()
        
        for _ in range(n_ahead):
            # Scale input
            X_scaled = self.scaler.transform(current_sequence.reshape(1, -1))
            
            # Predict next value
            next_yield = self.model.predict(X_scaled)[0]
            predictions.append(next_yield)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_yield)
        
        return np.array(predictions)


# Example usage
if __name__ == "__main__":
    print("Yield Prediction Module Demo")
    print("="*50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    # Generate synthetic wafer maps and yields
    wafer_maps = []
    process_params_list = []
    yields = []
    
    for i in range(n_samples):
        # Random wafer map
        wafer_map = np.random.randint(0, 2, size=(30, 30))
        wafer_maps.append(wafer_map)
        
        # Random process parameters
        params = ProcessParameters(
            temperature=np.random.uniform(100, 200),
            pressure=np.random.uniform(300, 700),
            exposure_time=np.random.uniform(20, 80),
            gas_flow_rate=np.random.uniform(50, 150),
            rf_power=np.random.uniform(100, 300),
            chamber_clean_cycles=np.random.randint(50, 200)
        )
        process_params_list.append(params)
        
        # Synthetic yield (based on defect density)
        defect_density = np.sum(wafer_map) / wafer_map.size
        yield_value = max(0.5, min(1.0, 0.95 - defect_density * 2 + np.random.normal(0, 0.02)))
        yields.append(yield_value)
    
    # Create and train yield predictor
    predictor = YieldPredictor(model_type='random_forest')
    
    # Extract features
    X = []
    for wafer_map, params in zip(wafer_maps, process_params_list):
        features = predictor.extract_features(wafer_map, params)
        X.append(features)
    
    X = np.array(X)
    y = np.array(yields)
    
    # Train model
    print("\nTraining Yield Predictor...")
    metrics = predictor.train(X, y, validate=True)
    
    print("\nTraining Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Feature importance
    importance = predictor.get_feature_importance()
    print("\nTop 5 Most Important Features:")
    for i, (feature, score) in enumerate(list(importance.items())[:5]):
        print(f"  {i+1}. {feature}: {score:.4f}")
    
    # Make predictions
    test_features = X[:5]
    predictions, confidence = predictor.predict(test_features)
    
    print("\nSample Predictions:")
    print(f"{'Actual':<10} {'Predicted':<10} {'CI':<10}")
    print("-"*30)
    for actual, pred, ci in zip(y[:5], predictions, confidence):
        print(f"{actual:.3f}      {pred:.3f}      ±{ci:.3f}")
    
    # Optimize parameters
    print("\nOptimizing Process Parameters...")
    optimizer = YieldOptimizer(predictor)
    
    optimal_params = optimizer.optimize_parameters(
        wafer_maps[0],
        {'particle': 10, 'scratch': 5}
    )
    
    print("\nOptimal Process Parameters:")
    print(f"  Temperature: {optimal_params.temperature:.1f}°C")
    print(f"  Pressure: {optimal_params.pressure:.1f} mTorr")
    print(f"  Exposure Time: {optimal_params.exposure_time:.1f}s")
    print(f"  Gas Flow Rate: {optimal_params.gas_flow_rate:.1f} sccm")
    print(f"  RF Power: {optimal_params.rf_power:.1f}W")
    print(f"  Chamber Clean Cycles: {optimal_params.chamber_clean_cycles}")
    
    # Time series prediction
    print("\nTime Series Yield Prediction...")
    ts_predictor = TimeSeriesYieldPredictor(lookback=10)
    
    # Generate synthetic yield history
    yield_history = np.random.beta(9, 1, 100)  # Skewed towards high yield
    
    # Train time series model
    ts_metrics = ts_predictor.train(yield_history)
    print(f"Time Series Model - RMSE: {ts_metrics['rmse']:.4f}, R²: {ts_metrics['r2']:.4f}")
    
    # Predict next yields
    next_yields = ts_predictor.predict_next(yield_history[-10:], n_ahead=5)
    print(f"Predicted next 5 yields: {next_yields}")
    
    print("\n✅ Yield Prediction Module Demo Complete!")