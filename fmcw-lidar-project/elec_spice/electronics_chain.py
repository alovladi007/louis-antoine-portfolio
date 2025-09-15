#!/usr/bin/env python3
"""
FMCW LiDAR-on-a-Tabletop: Electronics Chain Module
PLL/chirp synthesis, laser control, TIA/ADC, and behavioral SPICE models.
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

class FMCWElectronicsChain:
    """FMCW LiDAR electronics chain with PLL, laser control, and receiver AFE."""
    
    def __init__(self, config: Dict = None):
        """Initialize FMCW electronics chain."""
        self.config = config or {
            'wavelength': 1550e-9,           # Operating wavelength in m
            'frequency': 193.4e12,           # Operating frequency in Hz
            'bandwidth': 100e9,              # Chirp bandwidth in Hz
            'chirp_duration': 1e-3,          # Chirp duration in s
            'sampling_rate': 100e6,          # ADC sampling rate in Hz
            'adc_bits': 14,                  # ADC resolution in bits
            'adc_fs': 2.0,                   # ADC full scale in V
            'tia_gain': 1e3,                 # TIA gain in V/A
            'tia_bandwidth': 500e6,          # TIA bandwidth in Hz
            'tia_noise': 1e-12,              # TIA input noise in A/√Hz
            'pll_phase_noise': -80,          # PLL phase noise in dBc/Hz
            'pll_jitter': 1e-12,             # PLL jitter in s
            'laser_rin': -150,               # Laser RIN in dB/Hz
            'laser_linewidth': 1e6,          # Laser linewidth in Hz
            'tec_gain': 0.1,                 # TEC gain in A/K
            'tec_time_constant': 1e-3,       # TEC time constant in s
            'pid_kp': 1.0,                   # PID proportional gain
            'pid_ki': 0.1,                   # PID integral gain
            'pid_kd': 0.01,                  # PID derivative gain
            'supply_voltage': 3.3,           # Supply voltage in V
            'current_driver_gain': 0.1,      # Current driver gain in A/V
            'current_driver_bandwidth': 100e6, # Current driver bandwidth in Hz
            'frequency_points': 1000,        # Number of frequency points
            'time_points': 10000,            # Number of time points
            'temperature': 25,               # Operating temperature in °C
            'noise_temperature': 300,        # Noise temperature in K
            'load_resistance': 50,           # Load resistance in ohms
            'matching_network_q': 10,        # Matching network Q factor
            'psrr': 60,                      # Power supply rejection ratio in dB
            'cmrr': 80,                      # Common mode rejection ratio in dB
            'input_impedance': 1e6,          # Input impedance in ohms
            'output_impedance': 50           # Output impedance in ohms
        }
        
        self.frequency = None
        self.time = None
        self.chirp_signal = None
        self.laser_current = None
        self.tia_output = None
        self.adc_output = None
        
    def calculate_pll_response(self) -> Dict[str, np.ndarray]:
        """Calculate PLL response for chirp synthesis."""
        print("Calculating PLL response...")
        
        # Frequency vector
        f = np.linspace(1e3, self.config['bandwidth'], self.config['frequency_points'])
        
        # PLL parameters
        phase_noise_db = self.config['pll_phase_noise']
        jitter = self.config['pll_jitter']
        
        # Phase noise model (simplified)
        phase_noise_linear = 10**(phase_noise_db / 10)
        phase_noise_freq = phase_noise_linear / f  # 1/f noise
        
        # PLL transfer function (simplified)
        # H(s) = K / (s + K) where K is the loop bandwidth
        loop_bandwidth = 1e6  # 1 MHz loop bandwidth
        s = 1j * 2 * np.pi * f
        h_pll = loop_bandwidth / (s + loop_bandwidth)
        
        # Jitter contribution
        jitter_phase_noise = (2 * np.pi * f * jitter)**2
        
        # Total phase noise
        total_phase_noise = phase_noise_freq + jitter_phase_noise
        
        return {
            'frequency': f,
            'phase_noise': total_phase_noise,
            'phase_noise_db': 10 * np.log10(total_phase_noise),
            'transfer_function': h_pll,
            'loop_bandwidth': loop_bandwidth,
            'jitter': jitter
        }
    
    def calculate_chirp_synthesis(self) -> Dict[str, np.ndarray]:
        """Calculate chirp synthesis with predistortion."""
        print("Calculating chirp synthesis...")
        
        # Time vector
        t = np.linspace(0, self.config['chirp_duration'], self.config['time_points'])
        
        # Chirp parameters
        f0 = self.config['frequency']
        bandwidth = self.config['bandwidth']
        duration = self.config['chirp_duration']
        chirp_slope = bandwidth / duration
        
        # Ideal chirp
        instantaneous_freq = f0 + chirp_slope * t
        phase = 2 * np.pi * (f0 * t + 0.5 * chirp_slope * t**2)
        ideal_chirp = np.exp(1j * phase)
        
        # Predistortion LUT (simplified)
        # In practice, this would be a lookup table to linearize the chirp
        predistortion_factor = 1.0 + 0.01 * np.sin(2 * np.pi * t / duration)  # 1% nonlinearity
        predistorted_phase = phase * predistortion_factor
        predistorted_chirp = np.exp(1j * predistorted_phase)
        
        # Add phase noise
        pll_response = self.calculate_pll_response()
        phase_noise = np.interp(instantaneous_freq, pll_response['frequency'], 
                               pll_response['phase_noise'])
        phase_noise_time = np.random.normal(0, np.sqrt(phase_noise), len(t))
        
        # Final chirp signal
        final_chirp = predistorted_chirp * np.exp(1j * phase_noise_time)
        
        self.time = t
        self.chirp_signal = final_chirp
        
        return {
            'time': t,
            'ideal_chirp': ideal_chirp,
            'predistorted_chirp': predistorted_chirp,
            'final_chirp': final_chirp,
            'instantaneous_freq': instantaneous_freq,
            'chirp_slope': chirp_slope,
            'phase_noise': phase_noise_time
        }
    
    def calculate_laser_control(self) -> Dict[str, np.ndarray]:
        """Calculate laser current control and TEC loop."""
        print("Calculating laser control...")
        
        # Time vector
        t = np.linspace(0, self.config['chirp_duration'], self.config['time_points'])
        
        # Laser parameters
        laser_rin_db = self.config['laser_rin']
        laser_linewidth = self.config['laser_linewidth']
        tec_gain = self.config['tec_gain']
        tec_time_constant = self.config['tec_time_constant']
        
        # Current driver response
        current_driver_gain = self.config['current_driver_gain']
        current_driver_bandwidth = self.config['current_driver_bandwidth']
        
        # Current driver transfer function
        s = 1j * 2 * np.pi * np.logspace(0, 9, 1000)
        h_current = current_driver_gain / (1 + s / current_driver_bandwidth)
        
        # Laser current (simplified)
        base_current = 100e-3  # 100 mA
        modulation_current = 10e-3 * np.sin(2 * np.pi * 1e6 * t)  # 1 MHz modulation
        laser_current = base_current + modulation_current
        
        # TEC control (PID)
        target_temperature = 25  # °C
        actual_temperature = target_temperature + 0.1 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz drift
        
        # PID control
        error = target_temperature - actual_temperature
        pid_output = self._calculate_pid_control(error, t)
        
        # TEC current
        tec_current = tec_gain * pid_output
        
        # Laser RIN
        rin_linear = 10**(laser_rin_db / 10)
        rin_noise = np.sqrt(rin_linear) * np.random.randn(len(t))
        
        self.laser_current = laser_current
        
        return {
            'time': t,
            'laser_current': laser_current,
            'tec_current': tec_current,
            'temperature': actual_temperature,
            'target_temperature': target_temperature,
            'pid_output': pid_output,
            'rin_noise': rin_noise,
            'current_driver_response': h_current
        }
    
    def _calculate_pid_control(self, error: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Calculate PID control output."""
        dt = t[1] - t[0]
        
        # PID parameters
        kp = self.config['pid_kp']
        ki = self.config['pid_ki']
        kd = self.config['pid_kd']
        
        # Initialize
        integral = 0
        previous_error = 0
        pid_output = np.zeros_like(error)
        
        for i in range(len(error)):
            # Proportional term
            proportional = kp * error[i]
            
            # Integral term
            integral += error[i] * dt
            integral_term = ki * integral
            
            # Derivative term
            if i > 0:
                derivative = (error[i] - previous_error) / dt
            else:
                derivative = 0
            derivative_term = kd * derivative
            
            # PID output
            pid_output[i] = proportional + integral_term + derivative_term
            
            # Update previous error
            previous_error = error[i]
        
        return pid_output
    
    def calculate_tia_response(self) -> Dict[str, np.ndarray]:
        """Calculate TIA response and noise."""
        print("Calculating TIA response...")
        
        # Frequency vector
        f = np.logspace(0, 9, self.config['frequency_points'])
        
        # TIA parameters
        gain = self.config['tia_gain']
        bandwidth = self.config['tia_bandwidth']
        input_noise = self.config['tia_noise']
        load_resistance = self.config['load_resistance']
        
        # TIA transfer function
        s = 1j * 2 * np.pi * f
        h_tia = gain / (1 + s / bandwidth)
        
        # Noise analysis
        # Input-referred noise
        input_noise_density = input_noise  # A/√Hz
        
        # Thermal noise
        k_boltzmann = 1.38e-23
        temperature = self.config['noise_temperature']
        thermal_noise_density = np.sqrt(4 * k_boltzmann * temperature / load_resistance)
        
        # Total input noise
        total_input_noise = np.sqrt(input_noise_density**2 + thermal_noise_density**2)
        
        # Output noise
        output_noise_density = total_input_noise * np.abs(h_tia)
        
        # SNR calculation
        signal_power = 1e-6  # 1 μW input power
        signal_current = 0.8 * signal_power  # 0.8 A/W responsivity
        signal_voltage = signal_current * gain
        noise_voltage = output_noise_density * np.sqrt(bandwidth)
        snr = 20 * np.log10(signal_voltage / noise_voltage)
        
        return {
            'frequency': f,
            'transfer_function': h_tia,
            'gain_db': 20 * np.log10(np.abs(h_tia)),
            'phase_deg': np.angle(h_tia) * 180 / np.pi,
            'input_noise_density': input_noise_density,
            'thermal_noise_density': thermal_noise_density,
            'total_input_noise': total_input_noise,
            'output_noise_density': output_noise_density,
            'snr': snr,
            'bandwidth': bandwidth
        }
    
    def calculate_adc_response(self) -> Dict[str, np.ndarray]:
        """Calculate ADC response and quantization noise."""
        print("Calculating ADC response...")
        
        # ADC parameters
        bits = self.config['adc_bits']
        fs = self.config['adc_fs']
        sampling_rate = self.config['sampling_rate']
        
        # Quantization parameters
        lsb = fs / (2**bits)
        quantization_noise = lsb / np.sqrt(12)
        
        # ENOB calculation
        enob = bits - 0.5  # Simplified ENOB calculation
        
        # Frequency response
        f = np.logspace(0, 9, self.config['frequency_points'])
        nyquist_freq = sampling_rate / 2
        
        # Anti-aliasing filter response
        filter_order = 4
        filter_cutoff = nyquist_freq * 0.8
        filter_response = 1 / (1 + (f / filter_cutoff)**(2 * filter_order))
        
        # ADC frequency response
        adc_response = filter_response * np.sinc(f / sampling_rate)
        
        # SNR calculation
        signal_power = 0.5  # 0.5 V RMS
        noise_power = quantization_noise**2
        snr = 10 * np.log10(signal_power / noise_power)
        
        return {
            'frequency': f,
            'transfer_function': adc_response,
            'gain_db': 20 * np.log10(np.abs(adc_response)),
            'lsb': lsb,
            'quantization_noise': quantization_noise,
            'enob': enob,
            'snr': snr,
            'sampling_rate': sampling_rate,
            'nyquist_freq': nyquist_freq
        }
    
    def calculate_system_response(self) -> Dict[str, np.ndarray]:
        """Calculate complete system response."""
        print("Calculating complete system response...")
        
        # Get individual responses
        tia_response = self.calculate_tia_response()
        adc_response = self.calculate_adc_response()
        
        # System frequency response
        system_response = tia_response['transfer_function'] * adc_response['transfer_function']
        
        # System noise
        system_noise = np.sqrt(tia_response['output_noise_density']**2 + 
                              adc_response['quantization_noise']**2)
        
        # System SNR
        system_snr = 20 * np.log10(1.0 / system_noise)
        
        return {
            'frequency': tia_response['frequency'],
            'system_response': system_response,
            'system_noise': system_noise,
            'system_snr': system_snr,
            'tia_response': tia_response,
            'adc_response': adc_response
        }
    
    def calculate_psrr_cmrr(self) -> Dict[str, float]:
        """Calculate PSRR and CMRR."""
        print("Calculating PSRR and CMRR...")
        
        # PSRR and CMRR values
        psrr = self.config['psrr']
        cmrr = self.config['cmrr']
        
        # Supply voltage variations
        supply_variations = np.array([0.1, 0.2, 0.5, 1.0])  # V
        psrr_response = 20 * np.log10(supply_variations / (supply_variations / (10**(psrr/20))))
        
        # Common mode variations
        cm_variations = np.array([0.1, 0.2, 0.5, 1.0])  # V
        cmrr_response = 20 * np.log10(cm_variations / (cm_variations / (10**(cmrr/20))))
        
        return {
            'psrr': psrr,
            'cmrr': cmrr,
            'supply_variations': supply_variations,
            'psrr_response': psrr_response,
            'cm_variations': cm_variations,
            'cmrr_response': cmrr_response
        }
    
    def plot_electronics_analysis(self, output_dir: str = "elec_spice") -> None:
        """Plot electronics analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate responses
        pll_response = self.calculate_pll_response()
        chirp_data = self.calculate_chirp_synthesis()
        laser_control = self.calculate_laser_control()
        tia_response = self.calculate_tia_response()
        adc_response = self.calculate_adc_response()
        system_response = self.calculate_system_response()
        psrr_cmrr = self.calculate_psrr_cmrr()
        
        # Create plots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # PLL phase noise
        axes[0, 0].semilogx(pll_response['frequency'], pll_response['phase_noise_db'], 'b-', linewidth=2)
        axes[0, 0].set_title('PLL Phase Noise')
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('Phase Noise (dBc/Hz)')
        axes[0, 0].grid(True)
        
        # Chirp signal
        axes[0, 1].plot(chirp_data['time'] * 1e6, np.real(chirp_data['final_chirp']), 'g-', linewidth=2)
        axes[0, 1].set_title('Chirp Signal')
        axes[0, 1].set_xlabel('Time (μs)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True)
        
        # Laser current
        axes[0, 2].plot(laser_control['time'] * 1e6, laser_control['laser_current'] * 1e3, 'r-', linewidth=2)
        axes[0, 2].set_title('Laser Current')
        axes[0, 2].set_xlabel('Time (μs)')
        axes[0, 2].set_ylabel('Current (mA)')
        axes[0, 2].grid(True)
        
        # TIA response
        axes[1, 0].semilogx(tia_response['frequency'], tia_response['gain_db'], 'purple', linewidth=2)
        axes[1, 0].set_title('TIA Frequency Response')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Gain (dB)')
        axes[1, 0].grid(True)
        
        # TIA noise
        axes[1, 1].semilogx(tia_response['frequency'], tia_response['output_noise_density'] * 1e9, 'orange', linewidth=2)
        axes[1, 1].set_title('TIA Output Noise')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Noise (nV/√Hz)')
        axes[1, 1].grid(True)
        
        # ADC response
        axes[1, 2].semilogx(adc_response['frequency'], adc_response['gain_db'], 'brown', linewidth=2)
        axes[1, 2].set_title('ADC Frequency Response')
        axes[1, 2].set_xlabel('Frequency (Hz)')
        axes[1, 2].set_ylabel('Gain (dB)')
        axes[1, 2].grid(True)
        
        # System response
        axes[2, 0].semilogx(system_response['frequency'], 20 * np.log10(np.abs(system_response['system_response'])), 'black', linewidth=2)
        axes[2, 0].set_title('System Frequency Response')
        axes[2, 0].set_xlabel('Frequency (Hz)')
        axes[2, 0].set_ylabel('Gain (dB)')
        axes[2, 0].grid(True)
        
        # System noise
        axes[2, 1].semilogx(system_response['frequency'], system_response['system_noise'] * 1e9, 'gray', linewidth=2)
        axes[2, 1].set_title('System Noise')
        axes[2, 1].set_xlabel('Frequency (Hz)')
        axes[2, 1].set_ylabel('Noise (nV/√Hz)')
        axes[2, 1].grid(True)
        
        # PSRR/CMRR
        axes[2, 2].plot(psrr_cmrr['supply_variations'], psrr_cmrr['psrr_response'], 'b-', linewidth=2, label='PSRR')
        axes[2, 2].plot(psrr_cmrr['cm_variations'], psrr_cmrr['cmrr_response'], 'r-', linewidth=2, label='CMRR')
        axes[2, 2].set_title('PSRR and CMRR')
        axes[2, 2].set_xlabel('Variation (V)')
        axes[2, 2].set_ylabel('Response (dB)')
        axes[2, 2].legend()
        axes[2, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/electronics_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Electronics analysis plot saved to {output_dir}/electronics_analysis.png")

def main():
    """Main function to demonstrate FMCW electronics chain."""
    print("FMCW LiDAR-on-a-Tabletop: Electronics Chain")
    print("=" * 60)
    
    # Initialize electronics chain
    electronics = FMCWElectronicsChain()
    
    # Calculate responses
    pll_response = electronics.calculate_pll_response()
    chirp_data = electronics.calculate_chirp_synthesis()
    laser_control = electronics.calculate_laser_control()
    tia_response = electronics.calculate_tia_response()
    adc_response = electronics.calculate_adc_response()
    system_response = electronics.calculate_system_response()
    
    # Print key metrics
    print(f"PLL Phase Noise: {pll_response['phase_noise_db'][0]:.1f} dBc/Hz")
    print(f"TIA Gain: {tia_response['gain_db'][0]:.1f} dB")
    print(f"TIA Bandwidth: {tia_response['bandwidth']/1e6:.1f} MHz")
    print(f"ADC ENOB: {adc_response['enob']:.1f} bits")
    print(f"ADC SNR: {adc_response['snr']:.1f} dB")
    print(f"System SNR: {system_response['system_snr'][0]:.1f} dB")
    
    # Plot results
    electronics.plot_electronics_analysis()
    
    print("Electronics chain analysis complete!")

if __name__ == "__main__":
    main()