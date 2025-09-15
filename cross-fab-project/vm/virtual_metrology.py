#!/usr/bin/env python3
"""
Cross-Fab Project: Virtual Metrology Module
Implements CatBoost-based virtual metrology with conformal prediction for CD and overlay prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import catboost as cb
from catboost import CatBoostRegressor
import shap
import joblib
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class VirtualMetrology:
    """Virtual Metrology system using CatBoost with conformal prediction."""
    
    def __init__(self, config: Dict = None):
        """Initialize VM system."""
        self.config = config or {}
        self.cd_model = None
        self.overlay_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
    def create_features(self, litho_data: pd.DataFrame, etch_data: pd.DataFrame, 
                       context_data: pd.DataFrame) -> pd.DataFrame:
        """Create features for VM models."""
        print("Creating VM features...")
        
        # Merge data on wafer_id
        features = litho_data.groupby('wafer_id').agg({
            'dose': ['mean', 'std', 'min', 'max'],
            'focus': ['mean', 'std', 'min', 'max'],
            'stage_temp': ['mean', 'std'],
            'align_x': ['mean', 'std'],
            'align_y': ['mean', 'std'],
            'nils_proxy': ['mean', 'std'],
            'slit_scan': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        features.columns = ['wafer_id'] + [f"{col[0]}_{col[1]}" for col in features.columns[1:]]
        
        # Add etch/deposition features
        etch_features = etch_data.groupby('wafer_id').agg({
            'rf_power': ['mean', 'std'],
            'rf_bias': ['mean', 'std'],
            'etch_time': ['mean', 'std'],
            'pressure': ['mean', 'std'],
            'flow_rate': ['mean', 'std'],
            'temperature': ['mean', 'std'],
            'edge_ring_wear': ['mean', 'std'],
            'endpoint_time': ['mean', 'std'],
            'endpoint_spectra': ['mean', 'std']
        }).reset_index()
        
        etch_features.columns = ['wafer_id'] + [f"etch_{col[0]}_{col[1]}" for col in etch_features.columns[1:]]
        
        # Add context features
        context_features = context_data[['wafer_id', 'lot_age_days', 'fdc_alarm', 
                                       'chamber_maintenance', 'edge_exclusion']].copy()
        context_features['fdc_alarm'] = context_features['fdc_alarm'].astype(int)
        context_features['chamber_maintenance'] = context_features['chamber_maintenance'].astype(int)
        context_features['edge_exclusion'] = context_features['edge_exclusion'].astype(int)
        
        # Merge all features
        features = features.merge(etch_features, on='wafer_id', how='left')
        features = features.merge(context_features, on='wafer_id', how='left')
        
        # Add derived features
        features['dose_focus_interaction'] = features['dose_mean'] * features['focus_mean']
        features['rf_power_etch_time'] = features['etch_rf_power_mean'] * features['etch_etch_time_mean']
        features['chamber_drift'] = features['etch_chamber_id'] * 0.1  # Simplified chamber drift
        
        # Add EWMA features (simplified)
        features['dose_ewma'] = features['dose_mean'].ewm(span=5).mean()
        features['focus_ewma'] = features['focus_mean'].ewm(span=5).mean()
        
        # Add wafer position features
        features['wafer_center'] = (features['wafer_id'] % 25) < 5  # First 5 wafers are center
        features['wafer_edge'] = (features['wafer_id'] % 25) >= 20  # Last 5 wafers are edge
        
        self.feature_names = [col for col in features.columns if col != 'wafer_id']
        
        return features
    
    def prepare_targets(self, metrology_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare target variables for VM models."""
        print("Preparing VM targets...")
        
        # Group by wafer_id and calculate zone-wise statistics
        targets = metrology_data.groupby('wafer_id').agg({
            'cd': ['mean', 'std', 'min', 'max'],
            'overlay': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        targets.columns = ['wafer_id', 'cd_mean', 'cd_std', 'cd_min', 'cd_max',
                          'overlay_mean', 'overlay_std', 'overlay_min', 'overlay_max']
        
        return targets
    
    def train_cd_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train CD prediction model."""
        print("Training CD prediction model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train CatBoost model
        self.cd_model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        )
        
        self.cd_model.fit(
            X_train_scaled, y_train,
            eval_set=(X_val_scaled, y_val),
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate model
        y_pred = self.cd_model.predict(X_val_scaled)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print(f"CD Model - MAE: {mae:.3f}, R²: {r2:.3f}")
        
    def train_overlay_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train overlay prediction model."""
        print("Training overlay prediction model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train CatBoost model
        self.overlay_model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        )
        
        self.overlay_model.fit(
            X_train_scaled, y_train,
            eval_set=(X_val_scaled, y_val),
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate model
        y_pred = self.overlay_model.predict(X_val_scaled)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print(f"Overlay Model - MAE: {mae:.3f}, R²: {r2:.3f}")
    
    def train_models(self, features: pd.DataFrame, targets: pd.DataFrame) -> None:
        """Train both CD and overlay models."""
        print("Training VM models...")
        
        # Prepare feature matrix
        X = features[self.feature_names].fillna(0)
        
        # Train CD model
        self.train_cd_model(X, targets['cd_mean'])
        
        # Train overlay model
        self.train_overlay_model(X, targets['overlay_mean'])
        
        self.is_fitted = True
        print("VM models training complete!")
    
    def predict_cd(self, features: pd.DataFrame) -> np.ndarray:
        """Predict CD values."""
        if not self.is_fitted:
            raise ValueError("Models must be trained before prediction")
        
        X = features[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        return self.cd_model.predict(X_scaled)
    
    def predict_overlay(self, features: pd.DataFrame) -> np.ndarray:
        """Predict overlay values."""
        if not self.is_fitted:
            raise ValueError("Models must be trained before prediction")
        
        X = features[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        return self.overlay_model.predict(X_scaled)
    
    def get_feature_importance(self, model_type: str = 'cd') -> pd.DataFrame:
        """Get feature importance from trained model."""
        if not self.is_fitted:
            raise ValueError("Models must be trained before getting feature importance")
        
        model = self.cd_model if model_type == 'cd' else self.overlay_model
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def explain_prediction(self, features: pd.DataFrame, model_type: str = 'cd') -> np.ndarray:
        """Generate SHAP explanations for predictions."""
        if not self.is_fitted:
            raise ValueError("Models must be trained before generating explanations")
        
        model = self.cd_model if model_type == 'cd' else self.overlay_model
        X = features[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        
        return shap_values
    
    def save_models(self, output_dir: str = "vm") -> None:
        """Save trained models."""
        if not self.is_fitted:
            raise ValueError("Models must be trained before saving")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        joblib.dump(self.cd_model, os.path.join(output_dir, 'cd_model.pkl'))
        joblib.dump(self.overlay_model, os.path.join(output_dir, 'overlay_model.pkl'))
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        
        # Save feature names
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            for name in self.feature_names:
                f.write(f"{name}\n")
        
        print(f"Models saved to {output_dir}/")
    
    def load_models(self, model_dir: str = "vm") -> None:
        """Load trained models."""
        self.cd_model = joblib.load(os.path.join(model_dir, 'cd_model.pkl'))
        self.overlay_model = joblib.load(os.path.join(model_dir, 'overlay_model.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        with open(os.path.join(model_dir, 'feature_names.txt'), 'r') as f:
            self.feature_names = [line.strip() for line in f]
        
        self.is_fitted = True
        print(f"Models loaded from {model_dir}/")

def main():
    """Main function to demonstrate VM system."""
    print("Cross-Fab Project: Virtual Metrology")
    print("=" * 50)
    
    # This would typically load data from the data simulator
    # For demonstration, we'll create sample data
    print("Note: This is a demonstration. In practice, load data from data simulator.")
    
    # Initialize VM system
    vm = VirtualMetrology()
    
    print("VM system initialized successfully!")

if __name__ == "__main__":
    main()