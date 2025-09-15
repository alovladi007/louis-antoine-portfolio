#!/usr/bin/env python3
"""
Cross-Fab Project: Run-to-Run (R2R) Control Module
Implements double-EWMA and Kalman filter-based R2R control for process optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize
import joblib
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class R2RController:
    """Run-to-Run controller with double-EWMA and Kalman filter options."""
    
    def __init__(self, config: Dict = None):
        """Initialize R2R controller."""
        self.config = config or {
            'lambda1': 0.1,  # EWMA smoothing parameter 1
            'lambda2': 0.05,  # EWMA smoothing parameter 2
            'target_cd': 45.0,  # Target CD (nm)
            'target_overlay': 5.0,  # Target overlay (nm)
            'cd_tolerance': 2.0,  # CD tolerance (nm)
            'overlay_tolerance': 3.0,  # Overlay tolerance (nm)
            'max_dose_adjustment': 2.0,  # Max dose adjustment (mJ/cm²)
            'max_focus_adjustment': 0.5,  # Max focus adjustment (μm)
            'max_etch_time_adjustment': 5.0,  # Max etch time adjustment (s)
            'max_rf_bias_adjustment': 5.0,  # Max RF bias adjustment (%)
            'chamber_matching_enabled': True,
            'feedforward_enabled': True
        }
        
        self.reset()
        
    def reset(self):
        """Reset controller state."""
        self.cd_ewma1 = self.config['target_cd']
        self.cd_ewma2 = self.config['target_cd']
        self.overlay_ewma1 = self.config['target_overlay']
        self.overlay_ewma2 = self.config['target_overlay']
        
        self.dose_offset = 0.0
        self.focus_offset = 0.0
        self.etch_time_offset = 0.0
        self.rf_bias_offset = 0.0
        
        self.control_history = []
        self.prediction_history = []
        
    def double_ewma_update(self, measurement: float, target: float, 
                          lambda1: float, lambda2: float) -> Tuple[float, float]:
        """Update double-EWMA filter."""
        # First EWMA
        ewma1 = lambda1 * measurement + (1 - lambda1) * self.cd_ewma1
        # Second EWMA
        ewma2 = lambda2 * ewma1 + (1 - lambda2) * self.cd_ewma2
        
        # Control action
        control_action = 2 * ewma1 - ewma2 - target
        
        return ewma1, ewma2, control_action
    
    def kalman_filter_update(self, measurement: float, prediction: float, 
                           process_noise: float = 0.01, measurement_noise: float = 0.1) -> Tuple[float, float]:
        """Update Kalman filter for process state estimation."""
        # Simplified Kalman filter implementation
        # In practice, this would be more sophisticated
        
        # Prediction step
        predicted_state = prediction
        predicted_uncertainty = process_noise
        
        # Update step
        kalman_gain = predicted_uncertainty / (predicted_uncertainty + measurement_noise)
        updated_state = predicted_state + kalman_gain * (measurement - predicted_state)
        updated_uncertainty = (1 - kalman_gain) * predicted_uncertainty
        
        return updated_state, updated_uncertainty
    
    def calculate_control_action(self, cd_prediction: float, overlay_prediction: float,
                               cd_measurement: float = None, overlay_measurement: float = None) -> Dict:
        """Calculate control action based on predictions and measurements."""
        
        # Use measurements if available, otherwise use predictions
        cd_value = cd_measurement if cd_measurement is not None else cd_prediction
        overlay_value = overlay_measurement if overlay_measurement is not None else overlay_prediction
        
        # Double-EWMA update for CD
        self.cd_ewma1, self.cd_ewma2, cd_control = self.double_ewma_update(
            cd_value, self.config['target_cd'], 
            self.config['lambda1'], self.config['lambda2']
        )
        
        # Double-EWMA update for overlay
        self.overlay_ewma1, self.overlay_ewma2, overlay_control = self.double_ewma_update(
            overlay_value, self.config['target_overlay'],
            self.config['lambda1'], self.config['lambda2']
        )
        
        # Calculate process adjustments
        dose_adjustment = self._calculate_dose_adjustment(cd_control, overlay_control)
        focus_adjustment = self._calculate_focus_adjustment(cd_control, overlay_control)
        etch_time_adjustment = self._calculate_etch_time_adjustment(cd_control)
        rf_bias_adjustment = self._calculate_rf_bias_adjustment(overlay_control)
        
        # Apply constraints
        dose_adjustment = np.clip(dose_adjustment, -self.config['max_dose_adjustment'], 
                                self.config['max_dose_adjustment'])
        focus_adjustment = np.clip(focus_adjustment, -self.config['max_focus_adjustment'], 
                                 self.config['max_focus_adjustment'])
        etch_time_adjustment = np.clip(etch_time_adjustment, -self.config['max_etch_time_adjustment'], 
                                     self.config['max_etch_time_adjustment'])
        rf_bias_adjustment = np.clip(rf_bias_adjustment, -self.config['max_rf_bias_adjustment'], 
                                   self.config['max_rf_bias_adjustment'])
        
        # Update offsets
        self.dose_offset += dose_adjustment
        self.focus_offset += focus_adjustment
        self.etch_time_offset += etch_time_adjustment
        self.rf_bias_offset += rf_bias_adjustment
        
        control_action = {
            'dose_adjustment': dose_adjustment,
            'focus_adjustment': focus_adjustment,
            'etch_time_adjustment': etch_time_adjustment,
            'rf_bias_adjustment': rf_bias_adjustment,
            'dose_offset': self.dose_offset,
            'focus_offset': self.focus_offset,
            'etch_time_offset': self.etch_time_offset,
            'rf_bias_offset': self.rf_bias_offset,
            'cd_control': cd_control,
            'overlay_control': overlay_control,
            'cd_ewma1': self.cd_ewma1,
            'cd_ewma2': self.cd_ewma2,
            'overlay_ewma1': self.overlay_ewma1,
            'overlay_ewma2': self.overlay_ewma2
        }
        
        # Store history
        self.control_history.append(control_action)
        self.prediction_history.append({
            'cd_prediction': cd_prediction,
            'overlay_prediction': overlay_prediction,
            'cd_measurement': cd_measurement,
            'overlay_measurement': overlay_measurement
        })
        
        return control_action
    
    def _calculate_dose_adjustment(self, cd_control: float, overlay_control: float) -> float:
        """Calculate dose adjustment based on CD and overlay control signals."""
        # Simplified control law - in practice, this would be more sophisticated
        dose_gain = 0.5  # mJ/cm² per nm CD error
        return -dose_gain * cd_control
    
    def _calculate_focus_adjustment(self, cd_control: float, overlay_control: float) -> float:
        """Calculate focus adjustment based on CD and overlay control signals."""
        focus_gain = 0.1  # μm per nm CD error
        return -focus_gain * cd_control
    
    def _calculate_etch_time_adjustment(self, cd_control: float) -> float:
        """Calculate etch time adjustment based on CD control signal."""
        etch_gain = 0.2  # seconds per nm CD error
        return -etch_gain * cd_control
    
    def _calculate_rf_bias_adjustment(self, overlay_control: float) -> float:
        """Calculate RF bias adjustment based on overlay control signal."""
        rf_gain = 0.3  # % per nm overlay error
        return -rf_gain * overlay_control
    
    def apply_chamber_matching(self, chamber_id: int) -> Dict:
        """Apply chamber matching offsets."""
        if not self.config['chamber_matching_enabled']:
            return {}
        
        # Chamber-specific offsets (simplified)
        chamber_offsets = {
            0: {'dose': 0.0, 'focus': 0.0, 'etch_time': 0.0, 'rf_bias': 0.0},
            1: {'dose': 0.5, 'focus': 0.1, 'etch_time': 1.0, 'rf_bias': 2.0},
            2: {'dose': -0.3, 'focus': -0.05, 'etch_time': -0.5, 'rf_bias': -1.0},
            3: {'dose': 0.8, 'focus': 0.15, 'etch_time': 2.0, 'rf_bias': 3.0}
        }
        
        return chamber_offsets.get(chamber_id, {})
    
    def get_recipe_adjustments(self, base_recipe: Dict, chamber_id: int = 0) -> Dict:
        """Get adjusted recipe with control actions and chamber matching."""
        
        # Get control action
        if self.control_history:
            latest_action = self.control_history[-1]
        else:
            latest_action = {
                'dose_offset': 0.0, 'focus_offset': 0.0,
                'etch_time_offset': 0.0, 'rf_bias_offset': 0.0
            }
        
        # Get chamber matching offsets
        chamber_offsets = self.apply_chamber_matching(chamber_id)
        
        # Calculate final recipe
        adjusted_recipe = {
            'dose': base_recipe.get('dose', 20.0) + latest_action['dose_offset'] + chamber_offsets.get('dose', 0.0),
            'focus': base_recipe.get('focus', 0.0) + latest_action['focus_offset'] + chamber_offsets.get('focus', 0.0),
            'etch_time': base_recipe.get('etch_time', 60.0) + latest_action['etch_time_offset'] + chamber_offsets.get('etch_time', 0.0),
            'rf_bias': base_recipe.get('rf_bias', 50.0) + latest_action['rf_bias_offset'] + chamber_offsets.get('rf_bias', 0.0)
        }
        
        return adjusted_recipe
    
    def plot_control_history(self, output_dir: str = "apc") -> None:
        """Plot control history and performance."""
        if not self.control_history:
            print("No control history to plot")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data for plotting
        steps = range(len(self.control_history))
        dose_adjustments = [action['dose_adjustment'] for action in self.control_history]
        focus_adjustments = [action['focus_adjustment'] for action in self.control_history]
        cd_controls = [action['cd_control'] for action in self.control_history]
        overlay_controls = [action['overlay_control'] for action in self.control_history]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Dose adjustments
        axes[0, 0].plot(steps, dose_adjustments, 'b-', linewidth=2)
        axes[0, 0].set_title('Dose Adjustments')
        axes[0, 0].set_xlabel('Control Step')
        axes[0, 0].set_ylabel('Dose Adjustment (mJ/cm²)')
        axes[0, 0].grid(True)
        
        # Focus adjustments
        axes[0, 1].plot(steps, focus_adjustments, 'g-', linewidth=2)
        axes[0, 1].set_title('Focus Adjustments')
        axes[0, 1].set_xlabel('Control Step')
        axes[0, 1].set_ylabel('Focus Adjustment (μm)')
        axes[0, 1].grid(True)
        
        # CD control signal
        axes[1, 0].plot(steps, cd_controls, 'r-', linewidth=2)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('CD Control Signal')
        axes[1, 0].set_xlabel('Control Step')
        axes[1, 0].set_ylabel('CD Control (nm)')
        axes[1, 0].grid(True)
        
        # Overlay control signal
        axes[1, 1].plot(steps, overlay_controls, 'orange', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Overlay Control Signal')
        axes[1, 1].set_xlabel('Control Step')
        axes[1, 1].set_ylabel('Overlay Control (nm)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'r2r_control_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Control history plot saved to {output_dir}/r2r_control_history.png")
    
    def save_controller(self, output_dir: str = "apc") -> None:
        """Save controller state and configuration."""
        os.makedirs(output_dir, exist_ok=True)
        
        controller_data = {
            'config': self.config,
            'cd_ewma1': self.cd_ewma1,
            'cd_ewma2': self.cd_ewma2,
            'overlay_ewma1': self.overlay_ewma1,
            'overlay_ewma2': self.overlay_ewma2,
            'dose_offset': self.dose_offset,
            'focus_offset': self.focus_offset,
            'etch_time_offset': self.etch_time_offset,
            'rf_bias_offset': self.rf_bias_offset,
            'control_history': self.control_history,
            'prediction_history': self.prediction_history
        }
        
        joblib.dump(controller_data, os.path.join(output_dir, 'r2r_controller.pkl'))
        print(f"Controller saved to {output_dir}/r2r_controller.pkl")
    
    def load_controller(self, controller_file: str = "apc/r2r_controller.pkl") -> None:
        """Load controller state and configuration."""
        controller_data = joblib.load(controller_file)
        
        self.config = controller_data['config']
        self.cd_ewma1 = controller_data['cd_ewma1']
        self.cd_ewma2 = controller_data['cd_ewma2']
        self.overlay_ewma1 = controller_data['overlay_ewma1']
        self.overlay_ewma2 = controller_data['overlay_ewma2']
        self.dose_offset = controller_data['dose_offset']
        self.focus_offset = controller_data['focus_offset']
        self.etch_time_offset = controller_data['etch_time_offset']
        self.rf_bias_offset = controller_data['rf_bias_offset']
        self.control_history = controller_data['control_history']
        self.prediction_history = controller_data['prediction_history']
        
        print(f"Controller loaded from {controller_file}")

def main():
    """Main function to demonstrate R2R controller."""
    print("Cross-Fab Project: Run-to-Run Control")
    print("=" * 50)
    
    # Initialize R2R controller
    controller = R2RController()
    
    # Simulate some control steps
    print("Simulating R2R control...")
    
    for step in range(10):
        # Simulate predictions and measurements
        cd_pred = 45.0 + np.random.normal(0, 1.0)
        overlay_pred = 5.0 + np.random.normal(0, 0.5)
        cd_meas = cd_pred + np.random.normal(0, 0.2)
        overlay_meas = overlay_pred + np.random.normal(0, 0.1)
        
        # Calculate control action
        action = controller.calculate_control_action(cd_pred, overlay_pred, cd_meas, overlay_meas)
        
        print(f"Step {step+1}: CD={cd_meas:.2f}, Overlay={overlay_meas:.2f}, "
              f"Dose Adj={action['dose_adjustment']:.3f}, Focus Adj={action['focus_adjustment']:.3f}")
    
    # Plot control history
    controller.plot_control_history()
    
    # Save controller
    controller.save_controller()
    
    print("R2R control demonstration complete!")

if __name__ == "__main__":
    main()