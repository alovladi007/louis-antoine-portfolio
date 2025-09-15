#!/usr/bin/env python3
"""
HBM3E/4 SI-PI Co-Design: Signal Integrity Module
Channel modeling for TSV stack + interposer + package + DIMM trace with IBIS-AMI.
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

class HBMChannelModel:
    """HBM channel model for SI analysis."""
    
    def __init__(self, config: Dict = None):
        """Initialize HBM channel model."""
        self.config = config or {
            'data_rate_gbps': 9.6,  # Data rate in Gbps
            'tsv_count': 1024,      # Number of TSVs
            'tsv_diameter': 10e-6,  # TSV diameter in meters
            'tsv_height': 50e-6,    # TSV height in meters
            'tsv_pitch': 20e-6,     # TSV pitch in meters
            'interposer_thickness': 100e-6,  # Interposer thickness in meters
            'package_thickness': 800e-6,     # Package thickness in meters
            'dimm_trace_length': 50e-3,      # DIMM trace length in meters
            'dimm_trace_width': 100e-6,      # DIMM trace width in meters
            'dimm_trace_thickness': 35e-6,   # DIMM trace thickness in meters
            'dielectric_constant': 3.9,      # SiO2 dielectric constant
            'conductivity_cu': 5.8e7,        # Copper conductivity S/m
            'frequency_points': 1000,        # Number of frequency points
            'max_frequency': 20e9,           # Maximum frequency in Hz
            'temperature': 85,               # Operating temperature in °C
            'supply_voltage': 1.2,           # Supply voltage in V
            'impedance_target': 50,          # Target impedance in ohms
            'eye_height_target': 0.2,        # Target eye height in V
            'eye_width_target': 0.4          # Target eye width in UI
        }
        
        self.frequency = None
        self.s_parameters = None
        self.impulse_response = None
        self.eye_diagram = None
        
    def calculate_tsv_impedance(self) -> Dict[str, np.ndarray]:
        """Calculate TSV impedance parameters."""
        print("Calculating TSV impedance...")
        
        # TSV geometric parameters
        r_tsv = self.config['tsv_diameter'] / 2
        h_tsv = self.config['tsv_height']
        pitch = self.config['tsv_pitch']
        
        # Material properties
        eps_r = self.config['dielectric_constant']
        sigma_cu = self.config['conductivity_cu']
        
        # Frequency range
        f = np.logspace(6, np.log10(self.config['max_frequency']), self.config['frequency_points'])
        omega = 2 * np.pi * f
        
        # TSV resistance (DC + skin effect)
        r_dc = 1 / (sigma_cu * np.pi * r_tsv**2)
        skin_depth = np.sqrt(2 / (omega * 4e-7 * np.pi * sigma_cu))
        r_ac = r_dc * (1 + h_tsv / (3 * skin_depth))
        
        # TSV inductance
        l_tsv = (4e-7 * np.pi * h_tsv / np.pi) * np.log(pitch / r_tsv)
        
        # TSV capacitance
        c_tsv = (2 * np.pi * eps_r * 8.854e-12 * h_tsv) / np.log(pitch / r_tsv)
        
        # TSV conductance (simplified)
        g_tsv = omega * c_tsv * 0.01  # 1% loss tangent
        
        # Characteristic impedance
        z0_tsv = np.sqrt((r_ac + 1j * omega * l_tsv) / (g_tsv + 1j * omega * c_tsv))
        
        return {
            'frequency': f,
            'resistance': r_ac,
            'inductance': l_tsv,
            'capacitance': c_tsv,
            'conductance': g_tsv,
            'impedance': z0_tsv
        }
    
    def calculate_interposer_model(self) -> Dict[str, np.ndarray]:
        """Calculate interposer transmission line model."""
        print("Calculating interposer model...")
        
        # Interposer parameters
        thickness = self.config['interposer_thickness']
        eps_r = self.config['dielectric_constant']
        
        # Microstrip approximation
        w = 10e-6  # Trace width
        h = thickness
        
        # Effective dielectric constant
        eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * 1 / np.sqrt(1 + 12 * h / w)
        
        # Characteristic impedance
        z0 = 377 / np.sqrt(eps_eff) * np.log(8 * h / w + w / (4 * h))
        
        # Propagation constant
        f = np.logspace(6, np.log10(self.config['max_frequency']), self.config['frequency_points'])
        beta = 2 * np.pi * f * np.sqrt(eps_eff) / 3e8
        
        # Loss (simplified)
        alpha = 0.1 * f / 1e9  # dB/m
        
        return {
            'frequency': f,
            'impedance': np.full_like(f, z0),
            'propagation_constant': beta,
            'attenuation': alpha,
            'effective_dielectric_constant': eps_eff
        }
    
    def calculate_package_model(self) -> Dict[str, np.ndarray]:
        """Calculate package transmission line model."""
        print("Calculating package model...")
        
        # Package parameters
        thickness = self.config['package_thickness']
        eps_r = 4.0  # Package dielectric constant
        
        # Microstrip model
        w = 50e-6  # Trace width
        h = thickness
        
        # Effective dielectric constant
        eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * 1 / np.sqrt(1 + 12 * h / w)
        
        # Characteristic impedance
        z0 = 377 / np.sqrt(eps_eff) * np.log(8 * h / w + w / (4 * h))
        
        # Frequency response
        f = np.logspace(6, np.log10(self.config['max_frequency']), self.config['frequency_points'])
        beta = 2 * np.pi * f * np.sqrt(eps_eff) / 3e8
        
        # Loss
        alpha = 0.2 * f / 1e9  # dB/m
        
        return {
            'frequency': f,
            'impedance': np.full_like(f, z0),
            'propagation_constant': beta,
            'attenuation': alpha,
            'effective_dielectric_constant': eps_eff
        }
    
    def calculate_dimm_model(self) -> Dict[str, np.ndarray]:
        """Calculate DIMM trace model."""
        print("Calculating DIMM trace model...")
        
        # DIMM parameters
        length = self.config['dimm_trace_length']
        width = self.config['dimm_trace_width']
        thickness = self.config['dimm_trace_thickness']
        eps_r = 4.5  # PCB dielectric constant
        
        # Microstrip model
        h = 1.6e-3  # PCB thickness
        
        # Effective dielectric constant
        eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * 1 / np.sqrt(1 + 12 * h / width)
        
        # Characteristic impedance
        z0 = 377 / np.sqrt(eps_eff) * np.log(8 * h / width + width / (4 * h))
        
        # Frequency response
        f = np.logspace(6, np.log10(self.config['max_frequency']), self.config['frequency_points'])
        beta = 2 * np.pi * f * np.sqrt(eps_eff) / 3e8
        
        # Loss (frequency dependent)
        alpha = 0.5 * np.sqrt(f / 1e9)  # dB/m
        
        return {
            'frequency': f,
            'impedance': np.full_like(f, z0),
            'propagation_constant': beta,
            'attenuation': alpha,
            'effective_dielectric_constant': eps_eff,
            'length': length
        }
    
    def calculate_s_parameters(self) -> Dict[str, np.ndarray]:
        """Calculate S-parameters for the complete channel."""
        print("Calculating S-parameters...")
        
        # Get individual models
        tsv_model = self.calculate_tsv_impedance()
        interposer_model = self.calculate_interposer_model()
        package_model = self.calculate_package_model()
        dimm_model = self.calculate_dimm_model()
        
        # Frequency vector
        f = tsv_model['frequency']
        omega = 2 * np.pi * f
        
        # Calculate S-parameters for each section
        s_tsv = self._impedance_to_s_parameters(tsv_model['impedance'], 50)
        s_interposer = self._transmission_line_to_s_parameters(
            interposer_model['impedance'], 
            interposer_model['propagation_constant'],
            interposer_model['attenuation'],
            1e-3  # 1mm length
        )
        s_package = self._transmission_line_to_s_parameters(
            package_model['impedance'],
            package_model['propagation_constant'],
            package_model['attenuation'],
            1e-3  # 1mm length
        )
        s_dimm = self._transmission_line_to_s_parameters(
            dimm_model['impedance'],
            dimm_model['propagation_constant'],
            dimm_model['attenuation'],
            dimm_model['length']
        )
        
        # Cascade S-parameters
        s_total = self._cascade_s_parameters([s_tsv, s_interposer, s_package, s_dimm])
        
        self.frequency = f
        self.s_parameters = s_total
        
        return {
            'frequency': f,
            's11': s_total['s11'],
            's12': s_total['s12'],
            's21': s_total['s21'],
            's22': s_total['s22']
        }
    
    def _impedance_to_s_parameters(self, z: np.ndarray, z0: float) -> Dict[str, np.ndarray]:
        """Convert impedance to S-parameters."""
        gamma = (z - z0) / (z + z0)
        s11 = gamma
        s12 = 1 - gamma
        s21 = 1 - gamma
        s22 = gamma
        
        return {
            's11': s11,
            's12': s12,
            's21': s21,
            's22': s22
        }
    
    def _transmission_line_to_s_parameters(self, z0: np.ndarray, beta: np.ndarray, 
                                         alpha: np.ndarray, length: float) -> Dict[str, np.ndarray]:
        """Convert transmission line parameters to S-parameters."""
        # Propagation constant
        gamma = alpha / 8.686 + 1j * beta  # Convert dB/m to Np/m
        
        # S-parameters
        s11 = (z0 - 50) / (z0 + 50) * (1 - np.exp(-2 * gamma * length))
        s12 = 2 * np.sqrt(z0 / 50) / (z0 + 50) * np.exp(-gamma * length)
        s21 = s12
        s22 = s11
        
        return {
            's11': s11,
            's12': s12,
            's21': s21,
            's22': s22
        }
    
    def _cascade_s_parameters(self, s_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Cascade multiple S-parameter matrices."""
        if not s_list:
            return None
        
        result = s_list[0]
        for s in s_list[1:]:
            result = self._cascade_two_s_parameters(result, s)
        
        return result
    
    def _cascade_two_s_parameters(self, s1: Dict[str, np.ndarray], 
                                 s2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Cascade two S-parameter matrices."""
        # Convert to T-parameters, cascade, convert back
        t1 = self._s_to_t_parameters(s1)
        t2 = self._s_to_t_parameters(s2)
        
        # Cascade T-parameters
        t_result = {
            't11': t1['t11'] * t2['t11'] + t1['t12'] * t2['t21'],
            't12': t1['t11'] * t2['t12'] + t1['t12'] * t2['t22'],
            't21': t1['t21'] * t2['t11'] + t1['t22'] * t2['t21'],
            't22': t1['t21'] * t2['t12'] + t1['t22'] * t2['t22']
        }
        
        # Convert back to S-parameters
        return self._t_to_s_parameters(t_result)
    
    def _s_to_t_parameters(self, s: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Convert S-parameters to T-parameters."""
        t11 = (s['s12'] * s['s21'] - s['s11'] * s['s22']) / s['s21']
        t12 = s['s11'] / s['s21']
        t21 = -s['s22'] / s['s21']
        t22 = 1 / s['s21']
        
        return {
            't11': t11,
            't12': t12,
            't21': t21,
            't22': t22
        }
    
    def _t_to_s_parameters(self, t: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Convert T-parameters to S-parameters."""
        s11 = t['t12'] / t['t22']
        s12 = (t['t11'] * t['t22'] - t['t12'] * t['t21']) / t['t22']
        s21 = 1 / t['t22']
        s22 = -t['t21'] / t['t22']
        
        return {
            's11': s11,
            's12': s12,
            's21': s21,
            's22': s22
        }
    
    def calculate_eye_diagram(self, data_rate: float = None) -> Dict[str, np.ndarray]:
        """Calculate eye diagram for the channel."""
        if data_rate is None:
            data_rate = self.config['data_rate_gbps']
        
        print(f"Calculating eye diagram at {data_rate} Gbps...")
        
        # Get S-parameters
        if self.s_parameters is None:
            self.calculate_s_parameters()
        
        # Time vector
        t_bit = 1 / (data_rate * 1e9)  # Bit period
        t = np.linspace(0, 2 * t_bit, 1000)
        
        # Generate PRBS pattern
        prbs_length = 1023
        prbs = self._generate_prbs(prbs_length)
        
        # Convert to analog signal
        signal_analog = self._digital_to_analog(prbs, t, t_bit)
        
        # Apply channel response
        signal_filtered = self._apply_channel_response(signal_analog, t)
        
        # Generate eye diagram
        eye_data = self._generate_eye_diagram(signal_filtered, t, t_bit)
        
        self.eye_diagram = eye_data
        
        return eye_data
    
    def _generate_prbs(self, length: int) -> np.ndarray:
        """Generate PRBS pattern."""
        # PRBS-7 generator
        state = 0x7F  # Initial state
        prbs = []
        
        for _ in range(length):
            bit = state & 1
            prbs.append(bit)
            state = ((state << 1) ^ (bit << 7)) & 0x7F
        
        return np.array(prbs)
    
    def _digital_to_analog(self, digital: np.ndarray, t: np.ndarray, t_bit: float) -> np.ndarray:
        """Convert digital signal to analog."""
        # Create time vector for digital signal
        t_digital = np.arange(len(digital)) * t_bit
        
        # Interpolate to analog time vector
        analog = np.interp(t, t_digital, digital)
        
        return analog
    
    def _apply_channel_response(self, signal: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Apply channel frequency response."""
        # FFT
        f = np.fft.fftfreq(len(t), t[1] - t[0])
        signal_fft = np.fft.fft(signal)
        
        # Get channel response
        s21_interp = np.interp(f, self.frequency, np.abs(self.s_parameters['s21']))
        
        # Apply response
        signal_filtered_fft = signal_fft * s21_interp
        
        # IFFT
        signal_filtered = np.real(np.fft.ifft(signal_filtered_fft))
        
        return signal_filtered
    
    def _generate_eye_diagram(self, signal: np.ndarray, t: np.ndarray, t_bit: float) -> Dict[str, np.ndarray]:
        """Generate eye diagram data."""
        # Find bit boundaries
        bit_indices = np.arange(0, len(signal), int(len(signal) / (len(signal) * t_bit / (t[-1] - t[0])))
        
        # Extract eye data
        eye_data = []
        for i in range(len(bit_indices) - 2):
            start_idx = bit_indices[i]
            end_idx = bit_indices[i + 2]
            if end_idx < len(signal):
                eye_section = signal[start_idx:end_idx]
                eye_time = t[start_idx:end_idx] - t[start_idx]
                eye_data.append({
                    'time': eye_time,
                    'voltage': eye_section
                })
        
        return {
            'eye_data': eye_data,
            'bit_period': t_bit,
            'signal': signal,
            'time': t
        }
    
    def plot_s_parameters(self, output_dir: str = "si") -> None:
        """Plot S-parameters."""
        if self.s_parameters is None:
            self.calculate_s_parameters()
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # S11
        axes[0, 0].semilogx(self.frequency, 20 * np.log10(np.abs(self.s_parameters['s11'])))
        axes[0, 0].set_title('S11 - Return Loss')
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('Magnitude (dB)')
        axes[0, 0].grid(True)
        
        # S21
        axes[0, 1].semilogx(self.frequency, 20 * np.log10(np.abs(self.s_parameters['s21'])))
        axes[0, 1].set_title('S21 - Insertion Loss')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Magnitude (dB)')
        axes[0, 1].grid(True)
        
        # S12
        axes[1, 0].semilogx(self.frequency, 20 * np.log10(np.abs(self.s_parameters['s12'])))
        axes[1, 0].set_title('S12 - Reverse Isolation')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Magnitude (dB)')
        axes[1, 0].grid(True)
        
        # S22
        axes[1, 1].semilogx(self.frequency, 20 * np.log10(np.abs(self.s_parameters['s22'])))
        axes[1, 1].set_title('S22 - Output Return Loss')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Magnitude (dB)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/s_parameters.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"S-parameters plot saved to {output_dir}/s_parameters.png")
    
    def plot_eye_diagram(self, output_dir: str = "si") -> None:
        """Plot eye diagram."""
        if self.eye_diagram is None:
            self.calculate_eye_diagram()
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot eye diagram
        for eye_data in self.eye_diagram['eye_data'][:100]:  # Limit to 100 eyes for clarity
            ax.plot(eye_data['time'] / self.eye_diagram['bit_period'], 
                   eye_data['voltage'], 'b-', alpha=0.3, linewidth=0.5)
        
        ax.set_title(f'Eye Diagram at {self.config["data_rate_gbps"]} Gbps')
        ax.set_xlabel('Time (UI)')
        ax.set_ylabel('Voltage (V)')
        ax.grid(True)
        ax.set_xlim(0, 2)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/eye_diagram.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Eye diagram saved to {output_dir}/eye_diagram.png")

def main():
    """Main function to demonstrate HBM channel model."""
    print("HBM3E/4 SI-PI Co-Design: Signal Integrity Analysis")
    print("=" * 60)
    
    # Initialize channel model
    channel = HBMChannelModel()
    
    # Calculate S-parameters
    s_params = channel.calculate_s_parameters()
    
    # Calculate eye diagram
    eye_data = channel.calculate_eye_diagram()
    
    # Plot results
    channel.plot_s_parameters()
    channel.plot_eye_diagram()
    
    print("Signal integrity analysis complete!")

if __name__ == "__main__":
    main()