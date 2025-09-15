#!/usr/bin/env python3
"""
FMCW LiDAR-on-a-Tabletop: Firmware Module
Laser current/TEC control, PLL/chirp synthesis, and real-time control.
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

class LaserController:
    """Laser current and TEC controller for FMCW LiDAR."""
    
    def __init__(self, config: Dict = None):
        """Initialize laser controller."""
        self.config = config or {
            'laser_wavelength': 1550e-9,     # Laser wavelength in m
            'laser_power': 10e-3,            # Laser power in W
            'laser_current': 100e-3,         # Laser current in A
            'laser_threshold': 50e-3,        # Laser threshold current in A
            'laser_efficiency': 0.3,         # Laser efficiency
            'laser_ripple': 0.01,            # Laser current ripple
            'tec_setpoint': 25.0,            # TEC setpoint in °C
            'tec_tolerance': 0.1,            # TEC temperature tolerance in °C
            'tec_power': 2.0,                # TEC power in W
            'tec_efficiency': 0.8,           # TEC efficiency
            'control_frequency': 1e6,        # Control frequency in Hz
            'pid_kp': 1.0,                   # PID proportional gain
            'pid_ki': 0.1,                   # PID integral gain
            'pid_kd': 0.01,                  # PID derivative gain
            'max_current': 200e-3,           # Maximum laser current in A
            'min_current': 0,                # Minimum laser current in A
            'max_tec_power': 5.0,            # Maximum TEC power in W
            'min_tec_power': 0,              # Minimum TEC power in W
            'temperature_coefficient': 0.1,  # Temperature coefficient in nm/°C
            'current_coefficient': 0.01,     # Current coefficient in nm/mA
            'feedback_delay': 1e-6,          # Feedback delay in s
            'control_loop_delay': 1e-6,      # Control loop delay in s
            'noise_level': 1e-6,             # Noise level
            'calibration_points': 100,       # Number of calibration points
            'stability_threshold': 0.001,    # Stability threshold
            'settling_time': 1e-3,           # Settling time in s
            'overshoot_limit': 0.05,         # Overshoot limit
            'steady_state_error': 0.001      # Steady state error limit
        }
        
        self.current_state = {
            'laser_current': self.config['laser_current'],
            'tec_power': self.config['tec_power'],
            'temperature': self.config['tec_setpoint'],
            'wavelength': self.config['laser_wavelength'],
            'power': self.config['laser_power'],
            'stability': 0.0,
            'settling_time': 0.0,
            'overshoot': 0.0,
            'steady_state_error': 0.0
        }
        
        self.control_history = []
        self.calibration_data = None
        
    def calculate_laser_current_control(self, target_power: float) -> Dict[str, float]:
        """Calculate laser current control."""
        print("Calculating laser current control...")
        
        # Laser current calculation
        threshold_current = self.config['laser_threshold']
        efficiency = self.config['laser_efficiency']
        
        # Power-current relationship
        required_current = threshold_current + (target_power / efficiency)
        
        # Apply limits
        required_current = np.clip(required_current, 
                                 self.config['min_current'], 
                                 self.config['max_current'])
        
        # Current ripple
        ripple = self.config['laser_ripple'] * required_current
        current_with_ripple = required_current + ripple * np.sin(2 * np.pi * self.config['control_frequency'] * 0.001)
        
        # Update state
        self.current_state['laser_current'] = current_with_ripple
        self.current_state['power'] = target_power
        
        return {
            'required_current': required_current,
            'current_with_ripple': current_with_ripple,
            'ripple_amplitude': ripple,
            'efficiency': efficiency,
            'threshold_current': threshold_current
        }
    
    def calculate_tec_control(self, target_temperature: float) -> Dict[str, float]:
        """Calculate TEC control."""
        print("Calculating TEC control...")
        
        # Temperature difference
        current_temp = self.current_state['temperature']
        temp_diff = target_temperature - current_temp
        
        # TEC power calculation
        tec_efficiency = self.config['tec_efficiency']
        required_power = temp_diff / tec_efficiency
        
        # Apply limits
        required_power = np.clip(required_power, 
                               self.config['min_tec_power'], 
                               self.config['max_tec_power'])
        
        # PID control
        pid_output = self._calculate_pid_control(temp_diff)
        final_power = required_power + pid_output
        
        # Apply limits again
        final_power = np.clip(final_power, 
                            self.config['min_tec_power'], 
                            self.config['max_tec_power'])
        
        # Update state
        self.current_state['tec_power'] = final_power
        self.current_state['temperature'] = target_temperature
        
        return {
            'required_power': required_power,
            'final_power': final_power,
            'temperature_difference': temp_diff,
            'pid_output': pid_output,
            'efficiency': tec_efficiency
        }
    
    def _calculate_pid_control(self, error: float) -> float:
        """Calculate PID control output."""
        # Simplified PID control
        kp = self.config['pid_kp']
        ki = self.config['pid_ki']
        kd = self.config['pid_kd']
        
        # Proportional term
        p_term = kp * error
        
        # Integral term (simplified)
        i_term = ki * error * 0.001  # 1ms time step
        
        # Derivative term (simplified)
        d_term = kd * error / 0.001  # 1ms time step
        
        return p_term + i_term + d_term
    
    def calculate_wavelength_control(self, target_wavelength: float) -> Dict[str, float]:
        """Calculate wavelength control."""
        print("Calculating wavelength control...")
        
        # Current wavelength
        current_wavelength = self.current_state['wavelength']
        wavelength_diff = target_wavelength - current_wavelength
        
        # Temperature coefficient
        temp_coeff = self.config['temperature_coefficient']
        current_coeff = self.config['current_coefficient']
        
        # Required temperature change
        temp_change = wavelength_diff / temp_coeff
        
        # Required current change
        current_change = wavelength_diff / current_coeff
        
        # Update state
        new_temperature = self.current_state['temperature'] + temp_change
        new_current = self.current_state['laser_current'] + current_change
        
        # Apply limits
        new_temperature = np.clip(new_temperature, 0, 100)  # 0-100°C range
        new_current = np.clip(new_current, 
                            self.config['min_current'], 
                            self.config['max_current'])
        
        self.current_state['temperature'] = new_temperature
        self.current_state['laser_current'] = new_current
        self.current_state['wavelength'] = target_wavelength
        
        return {
            'wavelength_difference': wavelength_diff,
            'temperature_change': temp_change,
            'current_change': current_change,
            'new_temperature': new_temperature,
            'new_current': new_current,
            'temperature_coefficient': temp_coeff,
            'current_coefficient': current_coeff
        }
    
    def calculate_stability_analysis(self) -> Dict[str, float]:
        """Calculate stability analysis."""
        print("Calculating stability analysis...")
        
        # Stability metrics
        current_stability = np.std(self.current_state['laser_current'])
        temperature_stability = np.std(self.current_state['temperature'])
        power_stability = np.std(self.current_state['power'])
        wavelength_stability = np.std(self.current_state['wavelength'])
        
        # Overall stability
        overall_stability = (current_stability + temperature_stability + 
                           power_stability + wavelength_stability) / 4
        
        # Settling time (simplified)
        settling_time = self.config['settling_time']
        
        # Overshoot (simplified)
        overshoot = 0.0  # Would be calculated from step response
        
        # Steady state error
        steady_state_error = 0.0  # Would be calculated from step response
        
        # Update state
        self.current_state['stability'] = overall_stability
        self.current_state['settling_time'] = settling_time
        self.current_state['overshoot'] = overshoot
        self.current_state['steady_state_error'] = steady_state_error
        
        return {
            'current_stability': current_stability,
            'temperature_stability': temperature_stability,
            'power_stability': power_stability,
            'wavelength_stability': wavelength_stability,
            'overall_stability': overall_stability,
            'settling_time': settling_time,
            'overshoot': overshoot,
            'steady_state_error': steady_state_error
        }
    
    def calculate_calibration(self) -> Dict[str, np.ndarray]:
        """Calculate system calibration."""
        print("Calculating system calibration...")
        
        # Calibration parameters
        calibration_points = self.config['calibration_points']
        current_range = np.linspace(self.config['min_current'], 
                                  self.config['max_current'], 
                                  calibration_points)
        temperature_range = np.linspace(0, 100, calibration_points)
        
        # Laser power vs current
        power_vs_current = []
        for current in current_range:
            power = self._calculate_laser_power(current)
            power_vs_current.append(power)
        
        # Wavelength vs temperature
        wavelength_vs_temperature = []
        for temp in temperature_range:
            wavelength = self._calculate_wavelength(temp)
            wavelength_vs_temperature.append(wavelength)
        
        # Wavelength vs current
        wavelength_vs_current = []
        for current in current_range:
            wavelength = self._calculate_wavelength_from_current(current)
            wavelength_vs_current.append(wavelength)
        
        # Calibration curves
        power_calibration = np.polyfit(current_range, power_vs_current, 2)
        wavelength_temp_calibration = np.polyfit(temperature_range, wavelength_vs_temperature, 1)
        wavelength_current_calibration = np.polyfit(current_range, wavelength_vs_current, 1)
        
        self.calibration_data = {
            'current_range': current_range,
            'temperature_range': temperature_range,
            'power_vs_current': power_vs_current,
            'wavelength_vs_temperature': wavelength_vs_temperature,
            'wavelength_vs_current': wavelength_vs_current,
            'power_calibration': power_calibration,
            'wavelength_temp_calibration': wavelength_temp_calibration,
            'wavelength_current_calibration': wavelength_current_calibration
        }
        
        return self.calibration_data
    
    def _calculate_laser_power(self, current: float) -> float:
        """Calculate laser power from current."""
        threshold = self.config['laser_threshold']
        efficiency = self.config['laser_efficiency']
        
        if current < threshold:
            return 0.0
        else:
            return efficiency * (current - threshold)
    
    def _calculate_wavelength(self, temperature: float) -> float:
        """Calculate wavelength from temperature."""
        base_wavelength = self.config['laser_wavelength']
        temp_coeff = self.config['temperature_coefficient']
        base_temp = 25.0  # Base temperature in °C
        
        return base_wavelength + temp_coeff * (temperature - base_temp)
    
    def _calculate_wavelength_from_current(self, current: float) -> float:
        """Calculate wavelength from current."""
        base_wavelength = self.config['laser_wavelength']
        current_coeff = self.config['current_coefficient']
        base_current = self.config['laser_current']
        
        return base_wavelength + current_coeff * (current - base_current)
    
    def calculate_real_time_control(self, target_power: float, target_temperature: float, 
                                  target_wavelength: float) -> Dict[str, float]:
        """Calculate real-time control."""
        print("Calculating real-time control...")
        
        # Laser current control
        current_control = self.calculate_laser_current_control(target_power)
        
        # TEC control
        tec_control = self.calculate_tec_control(target_temperature)
        
        # Wavelength control
        wavelength_control = self.calculate_wavelength_control(target_wavelength)
        
        # Stability analysis
        stability_analysis = self.calculate_stability_analysis()
        
        # Store control history
        control_data = {
            'timestamp': len(self.control_history),
            'target_power': target_power,
            'target_temperature': target_temperature,
            'target_wavelength': target_wavelength,
            'current_control': current_control,
            'tec_control': tec_control,
            'wavelength_control': wavelength_control,
            'stability_analysis': stability_analysis
        }
        self.control_history.append(control_data)
        
        return control_data
    
    def plot_control_analysis(self, output_dir: str = "firmware") -> None:
        """Plot control analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Laser current vs power
        current_range = np.linspace(0, self.config['max_current'], 100)
        power_range = [self._calculate_laser_power(c) for c in current_range]
        
        axes[0, 0].plot(current_range * 1000, power_range * 1000, 'b-', linewidth=2)
        axes[0, 0].set_title('Laser Power vs Current')
        axes[0, 0].set_xlabel('Current (mA)')
        axes[0, 0].set_ylabel('Power (mW)')
        axes[0, 0].grid(True)
        
        # Wavelength vs temperature
        temp_range = np.linspace(0, 100, 100)
        wavelength_range = [self._calculate_wavelength(t) for t in temp_range]
        
        axes[0, 1].plot(temp_range, np.array(wavelength_range) * 1e9, 'r-', linewidth=2)
        axes[0, 1].set_title('Wavelength vs Temperature')
        axes[0, 1].set_xlabel('Temperature (°C)')
        axes[0, 1].set_ylabel('Wavelength (nm)')
        axes[0, 1].grid(True)
        
        # Wavelength vs current
        current_range = np.linspace(0, self.config['max_current'], 100)
        wavelength_range = [self._calculate_wavelength_from_current(c) for c in current_range]
        
        axes[0, 2].plot(current_range * 1000, np.array(wavelength_range) * 1e9, 'g-', linewidth=2)
        axes[0, 2].set_title('Wavelength vs Current')
        axes[0, 2].set_xlabel('Current (mA)')
        axes[0, 2].set_ylabel('Wavelength (nm)')
        axes[0, 2].grid(True)
        
        # Control history
        if self.control_history:
            timestamps = [c['timestamp'] for c in self.control_history]
            powers = [c['target_power'] for c in self.control_history]
            temperatures = [c['target_temperature'] for c in self.control_history]
            wavelengths = [c['target_wavelength'] for c in self.control_history]
            
            axes[1, 0].plot(timestamps, np.array(powers) * 1000, 'b-', linewidth=2, label='Power')
            axes[1, 0].set_title('Control History - Power')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Power (mW)')
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(timestamps, temperatures, 'r-', linewidth=2, label='Temperature')
            axes[1, 1].set_title('Control History - Temperature')
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('Temperature (°C)')
            axes[1, 1].grid(True)
            
            axes[1, 2].plot(timestamps, np.array(wavelengths) * 1e9, 'g-', linewidth=2, label='Wavelength')
            axes[1, 2].set_title('Control History - Wavelength')
            axes[1, 2].set_xlabel('Time Step')
            axes[1, 2].set_ylabel('Wavelength (nm)')
            axes[1, 2].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'No control history', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 1].text(0.5, 0.5, 'No control history', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 2].text(0.5, 0.5, 'No control history', ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/control_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Control analysis plot saved to {output_dir}/control_analysis.png")
    
    def plot_calibration_analysis(self, output_dir: str = "firmware") -> None:
        """Plot calibration analysis."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate calibration
        calibration_data = self.calculate_calibration()
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Power vs current calibration
        axes[0, 0].plot(np.array(calibration_data['current_range']) * 1000, 
                       np.array(calibration_data['power_vs_current']) * 1000, 'bo-', linewidth=2)
        axes[0, 0].set_title('Power vs Current Calibration')
        axes[0, 0].set_xlabel('Current (mA)')
        axes[0, 0].set_ylabel('Power (mW)')
        axes[0, 0].grid(True)
        
        # Wavelength vs temperature calibration
        axes[0, 1].plot(calibration_data['temperature_range'], 
                       np.array(calibration_data['wavelength_vs_temperature']) * 1e9, 'ro-', linewidth=2)
        axes[0, 1].set_title('Wavelength vs Temperature Calibration')
        axes[0, 1].set_xlabel('Temperature (°C)')
        axes[0, 1].set_ylabel('Wavelength (nm)')
        axes[0, 1].grid(True)
        
        # Wavelength vs current calibration
        axes[1, 0].plot(np.array(calibration_data['current_range']) * 1000, 
                       np.array(calibration_data['wavelength_vs_current']) * 1e9, 'go-', linewidth=2)
        axes[1, 0].set_title('Wavelength vs Current Calibration')
        axes[1, 0].set_xlabel('Current (mA)')
        axes[1, 0].set_ylabel('Wavelength (nm)')
        axes[1, 0].grid(True)
        
        # Calibration coefficients
        coeffs = ['Power (a)', 'Power (b)', 'Power (c)', 
                 'Wavelength-Temp (a)', 'Wavelength-Temp (b)',
                 'Wavelength-Current (a)', 'Wavelength-Current (b)']
        values = list(calibration_data['power_calibration']) + \
                list(calibration_data['wavelength_temp_calibration']) + \
                list(calibration_data['wavelength_current_calibration'])
        
        axes[1, 1].bar(coeffs, values, color=['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink'])
        axes[1, 1].set_title('Calibration Coefficients')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/calibration_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Calibration analysis plot saved to {output_dir}/calibration_analysis.png")

def main():
    """Main function to demonstrate laser controller."""
    print("FMCW LiDAR-on-a-Tabletop: Laser Controller")
    print("=" * 60)
    
    # Initialize laser controller
    controller = LaserController()
    
    # Test control
    target_power = 15e-3  # 15 mW
    target_temperature = 30.0  # 30°C
    target_wavelength = 1550.5e-9  # 1550.5 nm
    
    # Calculate control
    control_data = controller.calculate_real_time_control(target_power, target_temperature, target_wavelength)
    
    # Print results
    print(f"Target power: {target_power * 1000:.1f} mW")
    print(f"Target temperature: {target_temperature:.1f} °C")
    print(f"Target wavelength: {target_wavelength * 1e9:.1f} nm")
    print(f"Required current: {control_data['current_control']['required_current'] * 1000:.1f} mA")
    print(f"Required TEC power: {control_data['tec_control']['required_power']:.2f} W")
    print(f"Overall stability: {control_data['stability_analysis']['overall_stability']:.6f}")
    
    # Plot results
    controller.plot_control_analysis()
    controller.plot_calibration_analysis()
    
    print("Laser control complete!")

if __name__ == "__main__":
    main()