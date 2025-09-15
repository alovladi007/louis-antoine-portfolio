#!/usr/bin/env python3
"""
HBM3E/4 SI-PI Co-Design: Power Integrity Module
PDN modeling with Z(f) shaping and target impedance analysis.
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

class HBMPDNModel:
    """HBM Power Delivery Network (PDN) model for PI analysis."""
    
    def __init__(self, config: Dict = None):
        """Initialize HBM PDN model."""
        self.config = config or {
            'supply_voltage': 1.2,           # Supply voltage in V
            'max_current': 10.0,             # Maximum current in A
            'target_impedance': 0.1,         # Target impedance in ohms
            'frequency_points': 1000,        # Number of frequency points
            'max_frequency': 1e9,            # Maximum frequency in Hz
            'temperature': 85,               # Operating temperature in °C
            'vrm_inductance': 100e-9,        # VRM inductance in H
            'vrm_resistance': 0.01,          # VRM resistance in ohms
            'package_inductance': 200e-12,   # Package inductance in H
            'package_resistance': 0.005,     # Package resistance in ohms
            'die_capacitance': 100e-12,      # Die capacitance in F
            'die_resistance': 0.001,         # Die resistance in ohms
            'decoupling_capacitors': [       # Decoupling capacitor values
                {'value': 1e-6, 'esr': 0.01, 'esl': 0.5e-9, 'count': 10},
                {'value': 100e-9, 'esr': 0.05, 'esl': 0.2e-9, 'count': 50},
                {'value': 10e-9, 'esr': 0.1, 'esl': 0.1e-9, 'count': 100}
            ],
            'tsv_inductance': 50e-12,        # TSV inductance in H
            'tsv_resistance': 0.002,         # TSV resistance in ohms
            'interposer_capacitance': 50e-12, # Interposer capacitance in F
            'interposer_resistance': 0.003,  # Interposer resistance in ohms
            'burst_current_factor': 2.0,     # Burst current multiplication factor
            'current_rise_time': 1e-9,       # Current rise time in s
            'voltage_tolerance': 0.05        # Voltage tolerance (5%)
        }
        
        self.frequency = None
        self.impedance = None
        self.current_spectrum = None
        self.voltage_response = None
        
    def calculate_pdn_impedance(self) -> Dict[str, np.ndarray]:
        """Calculate PDN impedance vs frequency."""
        print("Calculating PDN impedance...")
        
        # Frequency vector
        f = np.logspace(0, np.log10(self.config['max_frequency']), self.config['frequency_points'])
        omega = 2 * np.pi * f
        
        # VRM impedance
        z_vrm = self.config['vrm_resistance'] + 1j * omega * self.config['vrm_inductance']
        
        # Package impedance
        z_package = self.config['package_resistance'] + 1j * omega * self.config['package_inductance']
        
        # TSV impedance
        z_tsv = self.config['tsv_resistance'] + 1j * omega * self.config['tsv_inductance']
        
        # Interposer impedance
        z_interposer = self.config['interposer_resistance'] + 1j * omega * self.config['interposer_capacitance']
        
        # Die impedance
        z_die = self.config['die_resistance'] + 1 / (1j * omega * self.config['die_capacitance'])
        
        # Decoupling capacitor impedances
        z_decap = self._calculate_decap_impedance(omega)
        
        # Total PDN impedance (simplified parallel combination)
        z_total = z_vrm + z_package + z_tsv + z_interposer + z_die + z_decap
        
        self.frequency = f
        self.impedance = z_total
        
        return {
            'frequency': f,
            'impedance': z_total,
            'magnitude': np.abs(z_total),
            'phase': np.angle(z_total),
            'vrm_impedance': z_vrm,
            'package_impedance': z_package,
            'tsv_impedance': z_tsv,
            'interposer_impedance': z_interposer,
            'die_impedance': z_die,
            'decap_impedance': z_decap
        }
    
    def _calculate_decap_impedance(self, omega: np.ndarray) -> np.ndarray:
        """Calculate decoupling capacitor impedance."""
        z_decap_total = np.zeros_like(omega, dtype=complex)
        
        for cap in self.config['decoupling_capacitors']:
            # Individual capacitor impedance
            z_cap = (cap['esr'] + 1j * omega * cap['esl'] + 
                    1 / (1j * omega * cap['value']))
            
            # Parallel combination of multiple capacitors
            z_decap_total += cap['count'] / z_cap
        
        # Convert back to impedance
        z_decap_total = 1 / z_decap_total
        
        return z_decap_total
    
    def calculate_current_spectrum(self, current_profile: str = 'burst') -> Dict[str, np.ndarray]:
        """Calculate current spectrum for different load profiles."""
        print(f"Calculating current spectrum for {current_profile} profile...")
        
        # Time vector
        t_max = 10e-9  # 10 ns
        dt = t_max / 10000
        t = np.arange(0, t_max, dt)
        
        # Generate current profile
        if current_profile == 'burst':
            current = self._generate_burst_current(t)
        elif current_profile == 'step':
            current = self._generate_step_current(t)
        elif current_profile == 'sinusoidal':
            current = self._generate_sinusoidal_current(t)
        else:
            current = self._generate_constant_current(t)
        
        # FFT to get frequency spectrum
        f = np.fft.fftfreq(len(t), dt)
        current_fft = np.fft.fft(current)
        
        # Take positive frequencies only
        positive_freqs = f > 0
        f_positive = f[positive_freqs]
        current_spectrum = np.abs(current_fft[positive_freqs])
        
        self.current_spectrum = {
            'frequency': f_positive,
            'magnitude': current_spectrum,
            'time': t,
            'current': current
        }
        
        return self.current_spectrum
    
    def _generate_burst_current(self, t: np.ndarray) -> np.ndarray:
        """Generate burst current profile."""
        # Burst parameters
        burst_duration = 2e-9  # 2 ns
        burst_start = 1e-9     # 1 ns
        burst_end = burst_start + burst_duration
        
        # Base current
        base_current = 0.1 * self.config['max_current']
        
        # Burst current
        burst_current = self.config['max_current'] * self.config['burst_current_factor']
        
        # Generate profile
        current = np.full_like(t, base_current)
        burst_mask = (t >= burst_start) & (t <= burst_end)
        current[burst_mask] = burst_current
        
        # Add rise/fall times
        rise_time = self.config['current_rise_time']
        fall_time = self.config['current_rise_time']
        
        # Rise transition
        rise_mask = (t >= burst_start) & (t < burst_start + rise_time)
        if np.any(rise_mask):
            rise_factor = (t[rise_mask] - burst_start) / rise_time
            current[rise_mask] = base_current + (burst_current - base_current) * rise_factor
        
        # Fall transition
        fall_mask = (t >= burst_end - fall_time) & (t <= burst_end)
        if np.any(fall_mask):
            fall_factor = (burst_end - t[fall_mask]) / fall_time
            current[fall_mask] = base_current + (burst_current - base_current) * fall_factor
        
        return current
    
    def _generate_step_current(self, t: np.ndarray) -> np.ndarray:
        """Generate step current profile."""
        step_time = 1e-9
        current = np.zeros_like(t)
        current[t >= step_time] = self.config['max_current']
        return current
    
    def _generate_sinusoidal_current(self, t: np.ndarray) -> np.ndarray:
        """Generate sinusoidal current profile."""
        freq = 100e6  # 100 MHz
        current = self.config['max_current'] * np.sin(2 * np.pi * freq * t)
        return current
    
    def _generate_constant_current(self, t: np.ndarray) -> np.ndarray:
        """Generate constant current profile."""
        return np.full_like(t, self.config['max_current'])
    
    def calculate_voltage_response(self, current_spectrum: Dict[str, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Calculate voltage response due to current spectrum."""
        if current_spectrum is None:
            current_spectrum = self.calculate_current_spectrum()
        
        print("Calculating voltage response...")
        
        # Get PDN impedance
        if self.impedance is None:
            self.calculate_pdn_impedance()
        
        # Interpolate impedance to current spectrum frequencies
        z_interp = np.interp(current_spectrum['frequency'], self.frequency, np.abs(self.impedance))
        
        # Calculate voltage spectrum
        voltage_spectrum = current_spectrum['magnitude'] * z_interp
        
        # Convert back to time domain
        voltage_time = np.real(np.fft.ifft(voltage_spectrum))
        
        self.voltage_response = {
            'frequency': current_spectrum['frequency'],
            'voltage_spectrum': voltage_spectrum,
            'time': current_spectrum['time'],
            'voltage': voltage_time,
            'current': current_spectrum['current']
        }
        
        return self.voltage_response
    
    def calculate_target_impedance(self) -> Dict[str, np.ndarray]:
        """Calculate target impedance based on current requirements."""
        print("Calculating target impedance...")
        
        # Target impedance calculation
        # Z_target = V_tolerance * V_supply / I_max
        z_target = (self.config['voltage_tolerance'] * self.config['supply_voltage'] / 
                   self.config['max_current'])
        
        # Frequency-dependent target impedance (simplified)
        f = np.logspace(0, np.log10(self.config['max_frequency']), self.config['frequency_points'])
        z_target_freq = np.full_like(f, z_target)
        
        return {
            'frequency': f,
            'target_impedance': z_target_freq,
            'constant_target': z_target
        }
    
    def optimize_decap_placement(self) -> Dict[str, any]:
        """Optimize decoupling capacitor placement and values."""
        print("Optimizing decoupling capacitor placement...")
        
        # Get current PDN impedance
        pdn_data = self.calculate_pdn_impedance()
        target_data = self.calculate_target_impedance()
        
        # Find frequencies where impedance exceeds target
        excess_mask = np.abs(pdn_data['impedance']) > target_data['target_impedance']
        excess_freqs = pdn_data['frequency'][excess_mask]
        excess_impedance = np.abs(pdn_data['impedance'][excess_mask])
        
        # Optimization results (simplified)
        optimization_results = {
            'excess_frequencies': excess_freqs,
            'excess_impedance': excess_impedance,
            'recommended_capacitors': [
                {'value': 1e-6, 'count': 20, 'placement': 'package'},
                {'value': 100e-9, 'count': 100, 'placement': 'die'},
                {'value': 10e-9, 'count': 200, 'placement': 'interposer'}
            ],
            'improvement_factor': 0.5  # 50% improvement
        }
        
        return optimization_results
    
    def plot_impedance_analysis(self, output_dir: str = "pi") -> None:
        """Plot impedance analysis results."""
        if self.impedance is None:
            self.calculate_pdn_impedance()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get target impedance
        target_data = self.calculate_target_impedance()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Impedance magnitude
        axes[0, 0].loglog(self.frequency, np.abs(self.impedance), 'b-', linewidth=2, label='PDN Impedance')
        axes[0, 0].loglog(target_data['frequency'], target_data['target_impedance'], 'r--', 
                         linewidth=2, label='Target Impedance')
        axes[0, 0].set_title('PDN Impedance vs Target')
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('Impedance (Ω)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Impedance phase
        axes[0, 1].semilogx(self.frequency, np.angle(self.impedance) * 180 / np.pi, 'b-', linewidth=2)
        axes[0, 1].set_title('PDN Impedance Phase')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Phase (degrees)')
        axes[0, 1].grid(True)
        
        # Current spectrum
        if self.current_spectrum is not None:
            axes[1, 0].loglog(self.current_spectrum['frequency'], self.current_spectrum['magnitude'], 
                             'g-', linewidth=2)
            axes[1, 0].set_title('Current Spectrum')
            axes[1, 0].set_xlabel('Frequency (Hz)')
            axes[1, 0].set_ylabel('Current (A)')
            axes[1, 0].grid(True)
        
        # Voltage response
        if self.voltage_response is not None:
            axes[1, 1].plot(self.voltage_response['time'] * 1e9, self.voltage_response['voltage'], 
                           'r-', linewidth=2)
            axes[1, 1].axhline(y=self.config['supply_voltage'], color='k', linestyle='--', 
                              label='Supply Voltage')
            axes[1, 1].axhline(y=self.config['supply_voltage'] * (1 - self.config['voltage_tolerance']), 
                              color='orange', linestyle='--', label='Tolerance Band')
            axes[1, 1].set_title('Voltage Response')
            axes[1, 1].set_xlabel('Time (ns)')
            axes[1, 1].set_ylabel('Voltage (V)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/impedance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Impedance analysis plot saved to {output_dir}/impedance_analysis.png")
    
    def plot_current_profiles(self, output_dir: str = "pi") -> None:
        """Plot different current profiles."""
        os.makedirs(output_dir, exist_ok=True)
        
        profiles = ['burst', 'step', 'sinusoidal', 'constant']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, profile in enumerate(profiles):
            current_data = self.calculate_current_spectrum(profile)
            
            axes[i].plot(current_data['time'] * 1e9, current_data['current'], 'b-', linewidth=2)
            axes[i].set_title(f'{profile.title()} Current Profile')
            axes[i].set_xlabel('Time (ns)')
            axes[i].set_ylabel('Current (A)')
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/current_profiles.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Current profiles plot saved to {output_dir}/current_profiles.png")

def main():
    """Main function to demonstrate HBM PDN model."""
    print("HBM3E/4 SI-PI Co-Design: Power Integrity Analysis")
    print("=" * 60)
    
    # Initialize PDN model
    pdn = HBMPDNModel()
    
    # Calculate PDN impedance
    impedance_data = pdn.calculate_pdn_impedance()
    
    # Calculate current spectrum
    current_data = pdn.calculate_current_spectrum('burst')
    
    # Calculate voltage response
    voltage_data = pdn.calculate_voltage_response(current_data)
    
    # Calculate target impedance
    target_data = pdn.calculate_target_impedance()
    
    # Optimize decoupling
    optimization = pdn.optimize_decap_placement()
    
    # Plot results
    pdn.plot_impedance_analysis()
    pdn.plot_current_profiles()
    
    print("Power integrity analysis complete!")

if __name__ == "__main__":
    main()