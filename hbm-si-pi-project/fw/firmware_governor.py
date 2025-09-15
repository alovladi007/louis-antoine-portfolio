#!/usr/bin/env python3
"""
HBM3E/4 SI-PI Co-Design: Firmware Governor Module
Thermal-aware throughput control with bandwidth/latency/temperature triage.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class HBMFirmwareGovernor:
    """HBM firmware governor for thermal-aware throughput control."""
    
    def __init__(self, config: Dict = None):
        """Initialize HBM firmware governor."""
        self.config = config or {
            'max_bandwidth': 1000,           # Maximum bandwidth in GB/s
            'max_latency': 100,              # Maximum latency in ns
            'max_temperature': 85,           # Maximum temperature in °C
            'throttling_threshold': 80,      # Throttling threshold in °C
            'shutdown_threshold': 90,        # Shutdown threshold in °C
            'control_period': 1e-3,          # Control period in s
            'pid_kp': 1.0,                   # PID proportional gain
            'pid_ki': 0.1,                   # PID integral gain
            'pid_kd': 0.01,                  # PID derivative gain
            'bandwidth_weights': {
                'temperature': 0.4,          # Weight for temperature
                'latency': 0.3,              # Weight for latency
                'power': 0.3                 # Weight for power
            },
            'throttling_levels': [           # Throttling levels
                {'level': 0, 'bandwidth_factor': 1.0, 'latency_factor': 1.0, 'power_factor': 1.0},
                {'level': 1, 'bandwidth_factor': 0.8, 'latency_factor': 1.2, 'power_factor': 0.8},
                {'level': 2, 'bandwidth_factor': 0.6, 'latency_factor': 1.5, 'power_factor': 0.6},
                {'level': 3, 'bandwidth_factor': 0.4, 'latency_factor': 2.0, 'power_factor': 0.4},
                {'level': 4, 'bandwidth_factor': 0.2, 'latency_factor': 3.0, 'power_factor': 0.2},
                {'level': 5, 'bandwidth_factor': 0.0, 'latency_factor': 10.0, 'power_factor': 0.0}
            ],
            'adaptive_control': True,        # Enable adaptive control
            'learning_rate': 0.01,           # Learning rate for adaptive control
            'history_length': 100,           # History length for learning
            'performance_targets': {
                'min_bandwidth': 500,        # Minimum acceptable bandwidth in GB/s
                'max_latency': 200,          # Maximum acceptable latency in ns
                'min_efficiency': 0.7        # Minimum power efficiency
            }
        }
        
        self.control_state = {
            'current_level': 0,
            'bandwidth': self.config['max_bandwidth'],
            'latency': 50,
            'temperature': 25,
            'power': 10,
            'efficiency': 1.0,
            'pid_integral': 0.0,
            'pid_previous_error': 0.0,
            'history': [],
            'performance_score': 1.0
        }
        
    def update_sensors(self, temperature: float, power: float, latency: float) -> None:
        """Update sensor readings."""
        self.control_state['temperature'] = temperature
        self.control_state['power'] = power
        self.control_state['latency'] = latency
        
        # Calculate efficiency
        self.control_state['efficiency'] = self._calculate_efficiency()
        
        # Update history
        self._update_history()
    
    def _calculate_efficiency(self) -> float:
        """Calculate power efficiency."""
        if self.control_state['power'] == 0:
            return 1.0
        
        # Efficiency based on bandwidth per watt
        efficiency = self.control_state['bandwidth'] / self.control_state['power']
        max_efficiency = self.config['max_bandwidth'] / self.config['max_bandwidth']  # Normalized
        return min(efficiency / max_efficiency, 1.0)
    
    def _update_history(self) -> None:
        """Update control history."""
        history_entry = {
            'timestamp': len(self.control_state['history']),
            'temperature': self.control_state['temperature'],
            'power': self.control_state['power'],
            'latency': self.control_state['latency'],
            'bandwidth': self.control_state['bandwidth'],
            'efficiency': self.control_state['efficiency'],
            'throttling_level': self.control_state['current_level']
        }
        
        self.control_state['history'].append(history_entry)
        
        # Keep only recent history
        if len(self.control_state['history']) > self.config['history_length']:
            self.control_state['history'] = self.control_state['history'][-self.config['history_length']:]
    
    def calculate_throttling_level(self) -> int:
        """Calculate appropriate throttling level based on current state."""
        temp = self.control_state['temperature']
        power = self.control_state['power']
        latency = self.control_state['latency']
        
        # Temperature-based throttling
        if temp >= self.config['shutdown_threshold']:
            return 5  # Shutdown
        elif temp >= self.config['throttling_threshold']:
            # Gradual throttling based on temperature
            temp_factor = (temp - self.config['throttling_threshold']) / (self.config['shutdown_threshold'] - self.config['throttling_threshold'])
            return min(int(temp_factor * 4) + 1, 4)
        
        # Power-based throttling
        if power > self.config['max_bandwidth'] * 0.8:  # 80% of max power
            return 2
        
        # Latency-based throttling
        if latency > self.config['max_latency'] * 0.8:  # 80% of max latency
            return 1
        
        return 0  # No throttling
    
    def calculate_adaptive_control(self) -> Dict[str, float]:
        """Calculate adaptive control parameters."""
        if not self.config['adaptive_control'] or len(self.control_state['history']) < 10:
            return self._get_default_control()
        
        # Analyze recent performance
        recent_history = self.control_state['history'][-10:]
        
        # Calculate performance trends
        temp_trend = self._calculate_trend([h['temperature'] for h in recent_history])
        power_trend = self._calculate_trend([h['power'] for h in recent_history])
        latency_trend = self._calculate_trend([h['latency'] for h in recent_history])
        
        # Adaptive control based on trends
        adaptive_control = self._get_default_control()
        
        # Adjust control based on trends
        if temp_trend > 0.1:  # Temperature increasing
            adaptive_control['temperature_weight'] *= 1.1
        if power_trend > 0.1:  # Power increasing
            adaptive_control['power_weight'] *= 1.1
        if latency_trend > 0.1:  # Latency increasing
            adaptive_control['latency_weight'] *= 1.1
        
        return adaptive_control
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        return slope
    
    def _get_default_control(self) -> Dict[str, float]:
        """Get default control parameters."""
        return {
            'temperature_weight': self.config['bandwidth_weights']['temperature'],
            'power_weight': self.config['bandwidth_weights']['power'],
            'latency_weight': self.config['bandwidth_weights']['latency']
        }
    
    def calculate_bandwidth_allocation(self) -> Dict[str, float]:
        """Calculate bandwidth allocation based on current state."""
        # Get throttling level
        throttling_level = self.calculate_throttling_level()
        
        # Get throttling parameters
        throttle_params = self.config['throttling_levels'][throttling_level]
        
        # Calculate base bandwidth
        base_bandwidth = self.config['max_bandwidth'] * throttle_params['bandwidth_factor']
        
        # Apply adaptive control
        if self.config['adaptive_control']:
            adaptive_control = self.calculate_adaptive_control()
            
            # Weighted adjustment based on current state
            temp_factor = 1.0 - (self.control_state['temperature'] - 25) / 60  # 0.5 at 85°C
            power_factor = 1.0 - (self.control_state['power'] - 5) / 15  # 0.5 at 20W
            latency_factor = 1.0 - (self.control_state['latency'] - 50) / 150  # 0.5 at 200ns
            
            # Apply weights
            weighted_factor = (adaptive_control['temperature_weight'] * temp_factor +
                             adaptive_control['power_weight'] * power_factor +
                             adaptive_control['latency_weight'] * latency_factor)
            
            base_bandwidth *= weighted_factor
        
        # Ensure minimum bandwidth
        min_bandwidth = self.config['performance_targets']['min_bandwidth']
        final_bandwidth = max(base_bandwidth, min_bandwidth)
        
        # Update control state
        self.control_state['bandwidth'] = final_bandwidth
        self.control_state['current_level'] = throttling_level
        
        return {
            'bandwidth': final_bandwidth,
            'throttling_level': throttling_level,
            'bandwidth_factor': throttle_params['bandwidth_factor'],
            'latency_factor': throttle_params['latency_factor'],
            'power_factor': throttle_params['power_factor']
        }
    
    def calculate_pid_control(self, target_temperature: float = None) -> Dict[str, float]:
        """Calculate PID control for temperature regulation."""
        if target_temperature is None:
            target_temperature = self.config['throttling_threshold'] - 5  # 5°C below threshold
        
        # Current error
        error = target_temperature - self.control_state['temperature']
        
        # PID terms
        proportional = self.config['pid_kp'] * error
        
        # Integral term
        self.control_state['pid_integral'] += error * self.config['control_period']
        integral = self.config['pid_ki'] * self.control_state['pid_integral']
        
        # Derivative term
        derivative = self.config['pid_kd'] * (error - self.control_state['pid_previous_error']) / self.config['control_period']
        
        # PID output
        pid_output = proportional + integral + derivative
        
        # Update previous error
        self.control_state['pid_previous_error'] = error
        
        # Convert PID output to bandwidth adjustment
        bandwidth_adjustment = max(0.1, min(1.0, 1.0 + pid_output / 10))  # Scale PID output
        
        return {
            'pid_output': pid_output,
            'bandwidth_adjustment': bandwidth_adjustment,
            'error': error,
            'proportional': proportional,
            'integral': integral,
            'derivative': derivative
        }
    
    def calculate_performance_score(self) -> float:
        """Calculate overall performance score."""
        # Normalize metrics
        bandwidth_score = self.control_state['bandwidth'] / self.config['max_bandwidth']
        latency_score = 1.0 - (self.control_state['latency'] - 50) / 150  # 0.5 at 200ns
        efficiency_score = self.control_state['efficiency']
        temperature_score = 1.0 - (self.control_state['temperature'] - 25) / 60  # 0.5 at 85°C
        
        # Weighted average
        weights = self.config['bandwidth_weights']
        performance_score = (weights['temperature'] * temperature_score +
                           weights['latency'] * latency_score +
                           weights['power'] * efficiency_score)
        
        self.control_state['performance_score'] = performance_score
        
        return performance_score
    
    def run_control_loop(self, duration: float = 1.0, dt: float = 1e-3) -> Dict[str, np.ndarray]:
        """Run control loop simulation."""
        print(f"Running control loop simulation for {duration} seconds...")
        
        # Time vector
        t = np.arange(0, duration, dt)
        n_steps = len(t)
        
        # Initialize arrays
        bandwidth_history = np.zeros(n_steps)
        latency_history = np.zeros(n_steps)
        temperature_history = np.zeros(n_steps)
        power_history = np.zeros(n_steps)
        throttling_level_history = np.zeros(n_steps)
        performance_score_history = np.zeros(n_steps)
        
        # Simulate system dynamics
        for i in range(n_steps):
            # Simulate sensor readings (simplified)
            temperature = 25 + 30 * np.sin(2 * np.pi * 0.1 * t[i]) + np.random.normal(0, 2)
            power = 10 + 5 * np.sin(2 * np.pi * 0.2 * t[i]) + np.random.normal(0, 1)
            latency = 50 + 20 * np.sin(2 * np.pi * 0.15 * t[i]) + np.random.normal(0, 5)
            
            # Update sensors
            self.update_sensors(temperature, power, latency)
            
            # Calculate bandwidth allocation
            bandwidth_allocation = self.calculate_bandwidth_allocation()
            
            # Calculate PID control
            pid_control = self.calculate_pid_control()
            
            # Calculate performance score
            performance_score = self.calculate_performance_score()
            
            # Store history
            bandwidth_history[i] = self.control_state['bandwidth']
            latency_history[i] = self.control_state['latency']
            temperature_history[i] = self.control_state['temperature']
            power_history[i] = self.control_state['power']
            throttling_level_history[i] = self.control_state['current_level']
            performance_score_history[i] = performance_score
        
        return {
            'time': t,
            'bandwidth': bandwidth_history,
            'latency': latency_history,
            'temperature': temperature_history,
            'power': power_history,
            'throttling_level': throttling_level_history,
            'performance_score': performance_score_history
        }
    
    def plot_control_analysis(self, control_data: Dict[str, np.ndarray], output_dir: str = "fw") -> None:
        """Plot control analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Bandwidth vs time
        axes[0, 0].plot(control_data['time'], control_data['bandwidth'], 'b-', linewidth=2)
        axes[0, 0].set_title('Bandwidth vs Time')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Bandwidth (GB/s)')
        axes[0, 0].grid(True)
        
        # Temperature vs time
        axes[0, 1].plot(control_data['time'], control_data['temperature'], 'r-', linewidth=2)
        axes[0, 1].axhline(y=self.config['throttling_threshold'], color='orange', linestyle='--', 
                          label='Throttling Threshold')
        axes[0, 1].axhline(y=self.config['shutdown_threshold'], color='red', linestyle='--', 
                          label='Shutdown Threshold')
        axes[0, 1].set_title('Temperature vs Time')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Temperature (°C)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Latency vs time
        axes[1, 0].plot(control_data['time'], control_data['latency'], 'g-', linewidth=2)
        axes[1, 0].set_title('Latency vs Time')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Latency (ns)')
        axes[1, 0].grid(True)
        
        # Power vs time
        axes[1, 1].plot(control_data['time'], control_data['power'], 'purple', linewidth=2)
        axes[1, 1].set_title('Power vs Time')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Power (W)')
        axes[1, 1].grid(True)
        
        # Throttling level vs time
        axes[2, 0].plot(control_data['time'], control_data['throttling_level'], 'orange', linewidth=2)
        axes[2, 0].set_title('Throttling Level vs Time')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Throttling Level')
        axes[2, 0].grid(True)
        
        # Performance score vs time
        axes[2, 1].plot(control_data['time'], control_data['performance_score'], 'brown', linewidth=2)
        axes[2, 1].set_title('Performance Score vs Time')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Performance Score')
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/control_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Control analysis plot saved to {output_dir}/control_analysis.png")

def main():
    """Main function to demonstrate HBM firmware governor."""
    print("HBM3E/4 SI-PI Co-Design: Firmware Governor")
    print("=" * 60)
    
    # Initialize firmware governor
    governor = HBMFirmwareGovernor()
    
    # Run control loop simulation
    control_data = governor.run_control_loop(duration=5.0, dt=1e-3)
    
    # Plot results
    governor.plot_control_analysis(control_data)
    
    print("Firmware governor simulation complete!")

if __name__ == "__main__":
    main()