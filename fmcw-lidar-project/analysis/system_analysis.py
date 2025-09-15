#!/usr/bin/env python3
"""
FMCW LiDAR-on-a-Tabletop: System Analysis Module
Phase noise, RIN, ADC ENOB, TIA noise, and photonic loss budget analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FMCWSystemAnalyzer:
    """FMCW LiDAR system analyzer for performance metrics."""
    
    def __init__(self, config: Dict = None):
        """Initialize system analyzer."""
        self.config = config or {
            'wavelength': 1550e-9,           # Operating wavelength in m
            'bandwidth': 100e9,              # Chirp bandwidth in Hz
            'chirp_duration': 1e-3,          # Chirp duration in s
            'sampling_rate': 100e6,          # ADC sampling rate in Hz
            'adc_bits': 12,                  # ADC resolution in bits
            'adc_fs': 1.0,                   # ADC full scale in V
            'tia_gain': 1e3,                 # TIA gain in V/A
            'tia_noise': 1e-12,              # TIA noise current in A/√Hz
            'laser_power': 10e-3,            # Laser power in W
            'laser_rin': -150,               # Laser RIN in dB/Hz
            'photonic_loss': 3.0,            # Photonic loss in dB
            'detector_responsivity': 0.8,    # Detector responsivity in A/W
            'detector_dark_current': 1e-9,   # Detector dark current in A
            'detector_noise': 1e-12,         # Detector noise current in A/√Hz
            'phase_noise': -80,              # Phase noise in dBc/Hz
            'temperature': 25,               # Operating temperature in °C
            'range_resolution': 0.1,         # Range resolution in m
            'velocity_resolution': 0.1,      # Velocity resolution in m/s
            'max_range': 100.0,              # Maximum range in m
            'max_velocity': 50.0,            # Maximum velocity in m/s
            'snr_threshold': 10,             # SNR threshold in dB
            'detection_probability': 0.9,    # Detection probability
            'false_alarm_rate': 1e-6,        # False alarm rate
            'calibration_frequency': 1e6,    # Calibration frequency in Hz
            'noise_floor': -80,              # Noise floor in dB
            'target_threshold': -60,         # Target detection threshold in dB
            'super_resolution_factor': 4,    # Super-resolution factor
            'phase_unwrap_threshold': np.pi, # Phase unwrap threshold
            'control_frequency': 1e6,        # Control frequency in Hz
            'feedback_delay': 1e-6,          # Feedback delay in s
            'control_loop_delay': 1e-6,      # Control loop delay in s
            'stability_threshold': 0.001,    # Stability threshold
            'settling_time': 1e-3,           # Settling time in s
            'overshoot_limit': 0.05,         # Overshoot limit
            'steady_state_error': 0.001      # Steady state error limit
        }
        
        self.analysis_results = {}
        
    def calculate_phase_noise_analysis(self) -> Dict[str, float]:
        """Calculate phase noise analysis."""
        print("Calculating phase noise analysis...")
        
        # Phase noise parameters
        phase_noise_db = self.config['phase_noise']
        frequency_offset = np.logspace(0, 6, 1000)  # 1 Hz to 1 MHz
        
        # Phase noise in linear units
        phase_noise_linear = 10**(phase_noise_db / 10)
        
        # Phase noise spectrum
        phase_noise_spectrum = phase_noise_linear / frequency_offset
        
        # Integrated phase noise
        integrated_phase_noise = np.trapz(phase_noise_spectrum, frequency_offset)
        
        # RMS phase noise
        rms_phase_noise = np.sqrt(integrated_phase_noise)
        
        # Phase noise impact on range accuracy
        c = 3e8  # Speed of light
        wavelength = self.config['wavelength']
        range_accuracy = rms_phase_noise * wavelength / (4 * np.pi)
        
        return {
            'phase_noise_db': phase_noise_db,
            'integrated_phase_noise': integrated_phase_noise,
            'rms_phase_noise': rms_phase_noise,
            'range_accuracy': range_accuracy,
            'frequency_offset': frequency_offset,
            'phase_noise_spectrum': phase_noise_spectrum
        }
    
    def calculate_rin_analysis(self) -> Dict[str, float]:
        """Calculate RIN analysis."""
        print("Calculating RIN analysis...")
        
        # RIN parameters
        rin_db = self.config['laser_rin']
        laser_power = self.config['laser_power']
        
        # RIN in linear units
        rin_linear = 10**(rin_db / 10)
        
        # RIN noise power
        rin_noise_power = rin_linear * laser_power**2
        
        # RIN noise current
        detector_responsivity = self.config['detector_responsivity']
        rin_noise_current = np.sqrt(rin_noise_power) * detector_responsivity
        
        # RIN impact on SNR
        signal_current = laser_power * detector_responsivity
        rin_snr = 20 * np.log10(signal_current / rin_noise_current)
        
        return {
            'rin_db': rin_db,
            'rin_linear': rin_linear,
            'rin_noise_power': rin_noise_power,
            'rin_noise_current': rin_noise_current,
            'rin_snr': rin_snr,
            'signal_current': signal_current
        }
    
    def calculate_adc_enob_analysis(self) -> Dict[str, float]:
        """Calculate ADC ENOB analysis."""
        print("Calculating ADC ENOB analysis...")
        
        # ADC parameters
        adc_bits = self.config['adc_bits']
        adc_fs = self.config['adc_fs']
        sampling_rate = self.config['sampling_rate']
        
        # Theoretical ENOB
        theoretical_enob = adc_bits
        
        # Practical ENOB (considering noise and distortion)
        noise_floor = self.config['noise_floor']
        snr_limitation = noise_floor - 6.02 * adc_bits - 1.76
        practical_enob = (snr_limitation - 1.76) / 6.02
        
        # ADC quantization noise
        quantization_noise = adc_fs / (2**adc_bits * np.sqrt(12))
        
        # ADC dynamic range
        dynamic_range = 20 * np.log10(2**adc_bits)
        
        # ADC resolution
        adc_resolution = adc_fs / (2**adc_bits)
        
        return {
            'theoretical_enob': theoretical_enob,
            'practical_enob': practical_enob,
            'quantization_noise': quantization_noise,
            'dynamic_range': dynamic_range,
            'adc_resolution': adc_resolution,
            'snr_limitation': snr_limitation
        }
    
    def calculate_tia_noise_analysis(self) -> Dict[str, float]:
        """Calculate TIA noise analysis."""
        print("Calculating TIA noise analysis...")
        
        # TIA parameters
        tia_gain = self.config['tia_gain']
        tia_noise = self.config['tia_noise']
        bandwidth = self.config['bandwidth']
        
        # TIA noise current
        tia_noise_current = tia_noise * np.sqrt(bandwidth)
        
        # TIA noise voltage
        tia_noise_voltage = tia_noise_current * tia_gain
        
        # TIA noise power
        tia_noise_power = tia_noise_voltage**2
        
        # TIA noise figure
        kt = 1.38e-23 * 300  # Thermal noise at 300K
        tia_noise_figure = 10 * np.log10(1 + tia_noise_power / (kt * bandwidth))
        
        return {
            'tia_noise_current': tia_noise_current,
            'tia_noise_voltage': tia_noise_voltage,
            'tia_noise_power': tia_noise_power,
            'tia_noise_figure': tia_noise_figure,
            'tia_gain': tia_gain,
            'bandwidth': bandwidth
        }
    
    def calculate_photonic_loss_budget(self) -> Dict[str, float]:
        """Calculate photonic loss budget."""
        print("Calculating photonic loss budget...")
        
        # Photonic parameters
        laser_power = self.config['laser_power']
        photonic_loss = self.config['photonic_loss']
        detector_responsivity = self.config['detector_responsivity']
        
        # Loss budget
        total_loss_db = photonic_loss
        total_loss_linear = 10**(total_loss_db / 10)
        
        # Power at detector
        detector_power = laser_power / total_loss_linear
        
        # Current at detector
        detector_current = detector_power * detector_responsivity
        
        # Loss breakdown
        loss_breakdown = {
            'laser_to_fiber': 0.5,      # dB
            'fiber_loss': 1.0,          # dB
            'splitter_loss': 1.0,       # dB
            'mzi_loss': 0.5,            # dB
            'detector_loss': 0.0        # dB
        }
        
        total_calculated_loss = sum(loss_breakdown.values())
        
        return {
            'total_loss_db': total_loss_db,
            'total_loss_linear': total_loss_linear,
            'detector_power': detector_power,
            'detector_current': detector_current,
            'loss_breakdown': loss_breakdown,
            'total_calculated_loss': total_calculated_loss
        }
    
    def calculate_system_performance(self) -> Dict[str, float]:
        """Calculate overall system performance."""
        print("Calculating system performance...")
        
        # Get individual analyses
        phase_noise = self.calculate_phase_noise_analysis()
        rin = self.calculate_rin_analysis()
        adc = self.calculate_adc_enob_analysis()
        tia = self.calculate_tia_noise_analysis()
        photonic = self.calculate_photonic_loss_budget()
        
        # System SNR
        signal_power = photonic['detector_current']**2
        noise_power = (rin['rin_noise_current']**2 + 
                      tia['tia_noise_current']**2 + 
                      (adc['quantization_noise'] / tia['tia_gain'])**2)
        
        system_snr = 10 * np.log10(signal_power / noise_power)
        
        # Range accuracy
        range_accuracy = phase_noise['range_accuracy']
        
        # Velocity accuracy
        velocity_accuracy = range_accuracy / self.config['chirp_duration']
        
        # Detection range
        max_range = self.config['max_range']
        detection_range = max_range * (system_snr / self.config['snr_threshold'])
        
        # Detection probability
        detection_probability = self.config['detection_probability']
        
        # False alarm rate
        false_alarm_rate = self.config['false_alarm_rate']
        
        self.analysis_results = {
            'system_snr': system_snr,
            'range_accuracy': range_accuracy,
            'velocity_accuracy': velocity_accuracy,
            'detection_range': detection_range,
            'detection_probability': detection_probability,
            'false_alarm_rate': false_alarm_rate,
            'phase_noise': phase_noise,
            'rin': rin,
            'adc': adc,
            'tia': tia,
            'photonic': photonic
        }
        
        return self.analysis_results
    
    def plot_system_analysis(self, output_dir: str = "analysis") -> None:
        """Plot system analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate system performance
        results = self.calculate_system_performance()
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Phase noise spectrum
        axes[0, 0].semilogx(results['phase_noise']['frequency_offset'], 
                           10 * np.log10(results['phase_noise']['phase_noise_spectrum']), 'b-', linewidth=2)
        axes[0, 0].set_title('Phase Noise Spectrum')
        axes[0, 0].set_xlabel('Frequency Offset (Hz)')
        axes[0, 0].set_ylabel('Phase Noise (dBc/Hz)')
        axes[0, 0].grid(True)
        
        # RIN analysis
        axes[0, 1].bar(['RIN', 'Noise Power', 'Noise Current'], 
                      [results['rin']['rin_db'], 
                       10 * np.log10(results['rin']['rin_noise_power']),
                       20 * np.log10(results['rin']['rin_noise_current'])], 
                      color=['blue', 'green', 'orange'])
        axes[0, 1].set_title('RIN Analysis')
        axes[0, 1].set_ylabel('Value (dB)')
        axes[0, 1].grid(True)
        
        # ADC ENOB
        axes[0, 2].bar(['Theoretical', 'Practical'], 
                      [results['adc']['theoretical_enob'], results['adc']['practical_enob']], 
                      color=['blue', 'red'])
        axes[0, 2].set_title('ADC ENOB')
        axes[0, 2].set_ylabel('ENOB (bits)')
        axes[0, 2].grid(True)
        
        # TIA noise
        axes[1, 0].bar(['Noise Current', 'Noise Voltage', 'Noise Power'], 
                      [20 * np.log10(results['tia']['tia_noise_current']),
                       20 * np.log10(results['tia']['tia_noise_voltage']),
                       10 * np.log10(results['tia']['tia_noise_power'])], 
                      color=['blue', 'green', 'orange'])
        axes[1, 0].set_title('TIA Noise Analysis')
        axes[1, 0].set_ylabel('Value (dB)')
        axes[1, 0].grid(True)
        
        # Photonic loss budget
        loss_breakdown = results['photonic']['loss_breakdown']
        components = list(loss_breakdown.keys())
        losses = list(loss_breakdown.values())
        
        axes[1, 1].bar(components, losses, color=['blue', 'green', 'orange', 'red', 'purple'])
        axes[1, 1].set_title('Photonic Loss Budget')
        axes[1, 1].set_ylabel('Loss (dB)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True)
        
        # System performance summary
        performance_metrics = ['SNR', 'Range Accuracy', 'Velocity Accuracy', 'Detection Range']
        performance_values = [results['system_snr'], 
                            results['range_accuracy'] * 1000,  # Convert to mm
                            results['velocity_accuracy'] * 1000,  # Convert to mm/s
                            results['detection_range']]
        
        axes[1, 2].bar(performance_metrics, performance_values, 
                      color=['blue', 'green', 'orange', 'red'])
        axes[1, 2].set_title('System Performance')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/system_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"System analysis plot saved to {output_dir}/system_analysis.png")

def main():
    """Main function to demonstrate system analyzer."""
    print("FMCW LiDAR-on-a-Tabletop: System Analyzer")
    print("=" * 60)
    
    # Initialize system analyzer
    analyzer = FMCWSystemAnalyzer()
    
    # Calculate system performance
    results = analyzer.calculate_system_performance()
    
    # Print results
    print(f"System SNR: {results['system_snr']:.1f} dB")
    print(f"Range accuracy: {results['range_accuracy'] * 1000:.2f} mm")
    print(f"Velocity accuracy: {results['velocity_accuracy'] * 1000:.2f} mm/s")
    print(f"Detection range: {results['detection_range']:.1f} m")
    print(f"Detection probability: {results['detection_probability']:.1%}")
    print(f"False alarm rate: {results['false_alarm_rate']:.1e}")
    
    # Plot results
    analyzer.plot_system_analysis()
    
    print("System analysis complete!")

if __name__ == "__main__":
    main()