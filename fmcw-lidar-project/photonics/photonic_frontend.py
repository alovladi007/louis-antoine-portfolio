#!/usr/bin/env python3
"""
FMCW LiDAR-on-a-Tabletop: Photonic Frontend Module
Silicon photonics front-end with splitters, MZI, and coherent receiver.
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

class FMCWPhotonicFrontend:
    """FMCW LiDAR photonic frontend with silicon photonics components."""
    
    def __init__(self, config: Dict = None):
        """Initialize FMCW photonic frontend."""
        self.config = config or {
            'wavelength': 1550e-9,           # Operating wavelength in m
            'frequency': 193.4e12,           # Operating frequency in Hz
            'bandwidth': 100e9,              # Chirp bandwidth in Hz
            'chirp_duration': 1e-3,          # Chirp duration in s
            'repetition_rate': 1000,         # Repetition rate in Hz
            'power_lo': 1e-3,                # LO power in W
            'power_tx': 10e-3,               # TX power in W
            'splitter_loss': 0.5,            # Splitter loss in dB
            'mzi_loss': 1.0,                 # MZI loss in dB
            'hybrid_loss': 1.5,              # 90° hybrid loss in dB
            'waveguide_loss': 0.1,           # Waveguide loss in dB/cm
            'coupling_loss': 3.0,            # Fiber coupling loss in dB
            'detector_responsivity': 0.8,    # Detector responsivity in A/W
            'detector_dark_current': 1e-9,   # Detector dark current in A
            'detector_noise_factor': 1.5,    # Detector noise factor
            'fiber_length': 1.0,             # Fiber length in m
            'target_distance': 10.0,         # Target distance in m
            'target_reflectivity': 0.1,      # Target reflectivity
            'atmospheric_loss': 0.1,         # Atmospheric loss in dB/km
            'beam_divergence': 1e-3,         # Beam divergence in rad
            'aperture_diameter': 5e-3,       # Aperture diameter in m
            'temperature': 25,               # Operating temperature in °C
            'phase_noise_level': -80,        # Phase noise level in dBc/Hz
            'rin_level': -150,               # RIN level in dB/Hz
            'frequency_points': 1000,        # Number of frequency points
            'time_points': 10000             # Number of time points
        }
        
        self.frequency = None
        self.time = None
        self.optical_power = None
        self.phase = None
        self.beat_signal = None
        
    def calculate_splitter_response(self) -> Dict[str, np.ndarray]:
        """Calculate 1x2 splitter response."""
        print("Calculating 1x2 splitter response...")
        
        # Frequency vector
        f = np.linspace(self.config['frequency'] - self.config['bandwidth']/2,
                       self.config['frequency'] + self.config['bandwidth']/2,
                       self.config['frequency_points'])
        
        # Splitter parameters
        splitter_loss_db = self.config['splitter_loss']
        splitter_loss_linear = 10**(splitter_loss_db / 10)
        
        # Splitter response (ideal)
        s11 = 0  # No reflection
        s12 = s21 = np.sqrt(1 / splitter_loss_linear)  # Equal splitting
        s22 = 0  # No reflection
        
        return {
            'frequency': f,
            's11': s11,
            's12': s12,
            's21': s21,
            's22': s22,
            'insertion_loss': splitter_loss_db,
            'splitting_ratio': 0.5
        }
    
    def calculate_mzi_response(self, phase_trim: float = 0.0) -> Dict[str, np.ndarray]:
        """Calculate MZI response with thermo-optic phase trim."""
        print("Calculating MZI response...")
        
        # Frequency vector
        f = np.linspace(self.config['frequency'] - self.config['bandwidth']/2,
                       self.config['frequency'] + self.config['bandwidth']/2,
                       self.config['frequency_points'])
        
        # MZI parameters
        mzi_loss_db = self.config['mzi_loss']
        mzi_loss_linear = 10**(mzi_loss_db / 10)
        
        # Phase difference (including trim)
        phase_diff = phase_trim
        
        # MZI response
        s11 = 0  # No reflection
        s12 = s21 = np.sqrt(1 / mzi_loss_linear) * np.cos(phase_diff / 2)
        s22 = 0  # No reflection
        
        return {
            'frequency': f,
            's11': s11,
            's12': s12,
            's21': s21,
            's22': s22,
            'insertion_loss': mzi_loss_db,
            'phase_trim': phase_trim,
            'transmission': np.abs(s21)**2
        }
    
    def calculate_hybrid_response(self) -> Dict[str, np.ndarray]:
        """Calculate 2x2 90° hybrid response."""
        print("Calculating 90° hybrid response...")
        
        # Frequency vector
        f = np.linspace(self.config['frequency'] - self.config['bandwidth']/2,
                       self.config['frequency'] + self.config['bandwidth']/2,
                       self.config['frequency_points'])
        
        # Hybrid parameters
        hybrid_loss_db = self.config['hybrid_loss']
        hybrid_loss_linear = 10**(hybrid_loss_db / 10)
        
        # 90° hybrid response
        s11 = s22 = 0  # No reflection
        s12 = s21 = np.sqrt(1 / hybrid_loss_linear) / np.sqrt(2)  # Equal splitting
        s13 = s31 = np.sqrt(1 / hybrid_loss_linear) / np.sqrt(2)  # Equal splitting
        s14 = s41 = np.sqrt(1 / hybrid_loss_linear) / np.sqrt(2) * 1j  # 90° phase shift
        s23 = s32 = np.sqrt(1 / hybrid_loss_linear) / np.sqrt(2) * 1j  # 90° phase shift
        s24 = s42 = np.sqrt(1 / hybrid_loss_linear) / np.sqrt(2)  # No phase shift
        
        return {
            'frequency': f,
            's11': s11, 's12': s12, 's13': s13, 's14': s14,
            's21': s21, 's22': s22, 's23': s23, 's24': s24,
            's31': s31, 's32': s32, 's33': s11, 's34': s12,
            's41': s41, 's42': s42, 's43': s21, 's44': s22,
            'insertion_loss': hybrid_loss_db
        }
    
    def calculate_fiber_propagation(self, length: float = None) -> Dict[str, np.ndarray]:
        """Calculate fiber propagation response."""
        if length is None:
            length = self.config['fiber_length']
        
        print(f"Calculating fiber propagation for {length} m...")
        
        # Frequency vector
        f = np.linspace(self.config['frequency'] - self.config['bandwidth']/2,
                       self.config['frequency'] + self.config['bandwidth']/2,
                       self.config['frequency_points'])
        
        # Fiber parameters
        waveguide_loss_db = self.config['waveguide_loss']  # dB/cm
        waveguide_loss_linear = 10**(waveguide_loss_db / 10)  # Linear
        
        # Propagation constant
        c = 3e8  # Speed of light
        n_eff = 1.45  # Effective index
        beta = 2 * np.pi * f * n_eff / c
        
        # Attenuation
        alpha = waveguide_loss_linear * length * 100  # Convert to cm
        
        # Fiber response
        s21 = np.exp(-alpha / 2) * np.exp(-1j * beta * length)
        
        return {
            'frequency': f,
            's21': s21,
            'insertion_loss': 20 * np.log10(np.abs(s21)),
            'phase': np.angle(s21),
            'length': length
        }
    
    def calculate_target_reflection(self, distance: float = None) -> Dict[str, np.ndarray]:
        """Calculate target reflection response."""
        if distance is None:
            distance = self.config['target_distance']
        
        print(f"Calculating target reflection for {distance} m...")
        
        # Frequency vector
        f = np.linspace(self.config['frequency'] - self.config['bandwidth']/2,
                       self.config['frequency'] + self.config['bandwidth']/2,
                       self.config['frequency_points'])
        
        # Target parameters
        reflectivity = self.config['target_reflectivity']
        atmospheric_loss_db = self.config['atmospheric_loss'] * distance / 1000  # dB
        atmospheric_loss_linear = 10**(atmospheric_loss_db / 10)
        
        # Beam divergence loss
        beam_area = np.pi * (distance * self.config['beam_divergence'] / 2)**2
        aperture_area = np.pi * (self.config['aperture_diameter'] / 2)**2
        divergence_loss = min(1.0, aperture_area / beam_area)
        
        # Round-trip propagation
        c = 3e8
        n_air = 1.0
        round_trip_distance = 2 * distance
        propagation_phase = 2 * np.pi * f * n_air * round_trip_distance / c
        
        # Target response
        s21 = np.sqrt(reflectivity * atmospheric_loss_linear * divergence_loss) * np.exp(-1j * propagation_phase)
        
        return {
            'frequency': f,
            's21': s21,
            'insertion_loss': 20 * np.log10(np.abs(s21)),
            'phase': np.angle(s21),
            'distance': distance,
            'reflectivity': reflectivity
        }
    
    def calculate_coherent_detection(self, tx_signal: np.ndarray, lo_signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate coherent detection response."""
        print("Calculating coherent detection...")
        
        # Detector parameters
        responsivity = self.config['detector_responsivity']
        dark_current = self.config['detector_dark_current']
        noise_factor = self.config['detector_noise_factor']
        
        # Balanced detection
        i_plus = responsivity * (tx_signal + lo_signal)**2
        i_minus = responsivity * (tx_signal - lo_signal)**2
        i_balanced = i_plus - i_minus
        
        # Add dark current
        i_total = i_balanced + dark_current
        
        # Calculate noise
        shot_noise = np.sqrt(2 * 1.6e-19 * responsivity * (np.abs(tx_signal)**2 + np.abs(lo_signal)**2))
        thermal_noise = np.sqrt(4 * 1.38e-23 * 300 * 50 * 1e9)  # 50 ohm, 1 GHz
        total_noise = np.sqrt(shot_noise**2 + thermal_noise**2) * noise_factor
        
        return {
            'current': i_total,
            'current_balanced': i_balanced,
            'shot_noise': shot_noise,
            'thermal_noise': thermal_noise,
            'total_noise': total_noise,
            'snr': np.abs(i_balanced) / total_noise,
            'responsivity': responsivity
        }
    
    def calculate_beat_signal(self, chirp_slope: float = None) -> Dict[str, np.ndarray]:
        """Calculate beat signal for FMCW LiDAR."""
        if chirp_slope is None:
            chirp_slope = self.config['bandwidth'] / self.config['chirp_duration']
        
        print(f"Calculating beat signal with chirp slope {chirp_slope/1e6:.1f} MHz/μs...")
        
        # Time vector
        t = np.linspace(0, self.config['chirp_duration'], self.config['time_points'])
        
        # Chirp parameters
        f0 = self.config['frequency']
        bandwidth = self.config['bandwidth']
        duration = self.config['chirp_duration']
        
        # Generate chirp signal
        instantaneous_freq = f0 + chirp_slope * t
        phase = 2 * np.pi * (f0 * t + 0.5 * chirp_slope * t**2)
        chirp_signal = np.exp(1j * phase)
        
        # Calculate beat frequency
        target_distance = self.config['target_distance']
        c = 3e8
        beat_freq = 2 * chirp_slope * target_distance / c
        
        # Generate beat signal
        beat_phase = 2 * np.pi * beat_freq * t
        beat_signal = np.cos(beat_phase)
        
        # Add noise
        noise_level = 0.1
        noise = noise_level * np.random.randn(len(t))
        beat_signal_noisy = beat_signal + noise
        
        self.time = t
        self.beat_signal = beat_signal_noisy
        
        return {
            'time': t,
            'chirp_signal': chirp_signal,
            'beat_signal': beat_signal,
            'beat_signal_noisy': beat_signal_noisy,
            'beat_frequency': beat_freq,
            'target_distance': target_distance,
            'chirp_slope': chirp_slope
        }
    
    def calculate_range_resolution(self) -> float:
        """Calculate range resolution."""
        c = 3e8
        bandwidth = self.config['bandwidth']
        range_resolution = c / (2 * bandwidth)
        return range_resolution
    
    def calculate_max_range(self) -> float:
        """Calculate maximum unambiguous range."""
        c = 3e8
        chirp_duration = self.config['chirp_duration']
        max_range = c * chirp_duration / 2
        return max_range
    
    def calculate_velocity_resolution(self, observation_time: float = None) -> float:
        """Calculate velocity resolution."""
        if observation_time is None:
            observation_time = self.config['chirp_duration']
        
        c = 3e8
        wavelength = self.config['wavelength']
        velocity_resolution = wavelength / (2 * observation_time)
        return velocity_resolution
    
    def calculate_snr(self, target_distance: float = None) -> float:
        """Calculate signal-to-noise ratio."""
        if target_distance is None:
            target_distance = self.config['target_distance']
        
        # System parameters
        tx_power = self.config['power_tx']
        lo_power = self.config['power_lo']
        responsivity = self.config['detector_responsivity']
        target_reflectivity = self.config['target_reflectivity']
        
        # Losses
        splitter_loss = 10**(self.config['splitter_loss'] / 10)
        mzi_loss = 10**(self.config['mzi_loss'] / 10)
        hybrid_loss = 10**(self.config['hybrid_loss'] / 10)
        coupling_loss = 10**(self.config['coupling_loss'] / 10)
        atmospheric_loss = 10**(self.config['atmospheric_loss'] * target_distance / 1000 / 10)
        
        # Total loss
        total_loss = splitter_loss * mzi_loss * hybrid_loss * coupling_loss * atmospheric_loss
        
        # Received power
        received_power = tx_power * target_reflectivity / total_loss
        
        # Signal current
        signal_current = responsivity * received_power
        
        # LO current
        lo_current = responsivity * lo_power / total_loss
        
        # Beat signal current
        beat_current = 2 * np.sqrt(signal_current * lo_current)
        
        # Noise current
        shot_noise = np.sqrt(2 * 1.6e-19 * responsivity * (received_power + lo_power / total_loss))
        thermal_noise = np.sqrt(4 * 1.38e-23 * 300 * 50 * 1e9)
        total_noise = np.sqrt(shot_noise**2 + thermal_noise**2)
        
        # SNR
        snr = beat_current / total_noise
        
        return snr
    
    def plot_photonic_response(self, output_dir: str = "photonics") -> None:
        """Plot photonic response analysis."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate responses
        splitter_response = self.calculate_splitter_response()
        mzi_response = self.calculate_mzi_response()
        hybrid_response = self.calculate_hybrid_response()
        fiber_response = self.calculate_fiber_propagation()
        target_response = self.calculate_target_reflection()
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Splitter response
        axes[0, 0].plot(splitter_response['frequency'] / 1e12, 
                       20 * np.log10(np.abs(splitter_response['s21'])), 'b-', linewidth=2)
        axes[0, 0].set_title('1x2 Splitter Response')
        axes[0, 0].set_xlabel('Frequency (THz)')
        axes[0, 0].set_ylabel('Insertion Loss (dB)')
        axes[0, 0].grid(True)
        
        # MZI response
        axes[0, 1].plot(mzi_response['frequency'] / 1e12, 
                       mzi_response['transmission'], 'g-', linewidth=2)
        axes[0, 1].set_title('MZI Transmission')
        axes[0, 1].set_xlabel('Frequency (THz)')
        axes[0, 1].set_ylabel('Transmission')
        axes[0, 1].grid(True)
        
        # Hybrid response
        axes[0, 2].plot(hybrid_response['frequency'] / 1e12, 
                       20 * np.log10(np.abs(hybrid_response['s21'])), 'r-', linewidth=2)
        axes[0, 2].set_title('90° Hybrid Response')
        axes[0, 2].set_xlabel('Frequency (THz)')
        axes[0, 2].set_ylabel('Insertion Loss (dB)')
        axes[0, 2].grid(True)
        
        # Fiber response
        axes[1, 0].plot(fiber_response['frequency'] / 1e12, 
                       fiber_response['insertion_loss'], 'purple', linewidth=2)
        axes[1, 0].set_title('Fiber Propagation')
        axes[1, 0].set_xlabel('Frequency (THz)')
        axes[1, 0].set_ylabel('Insertion Loss (dB)')
        axes[1, 0].grid(True)
        
        # Target response
        axes[1, 1].plot(target_response['frequency'] / 1e12, 
                       target_response['insertion_loss'], 'orange', linewidth=2)
        axes[1, 1].set_title('Target Reflection')
        axes[1, 1].set_xlabel('Frequency (THz)')
        axes[1, 1].set_ylabel('Insertion Loss (dB)')
        axes[1, 1].grid(True)
        
        # Beat signal
        beat_data = self.calculate_beat_signal()
        axes[1, 2].plot(beat_data['time'] * 1e6, beat_data['beat_signal_noisy'], 'b-', linewidth=1)
        axes[1, 2].set_title('Beat Signal')
        axes[1, 2].set_xlabel('Time (μs)')
        axes[1, 2].set_ylabel('Amplitude')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/photonic_response.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Photonic response plot saved to {output_dir}/photonic_response.png")
    
    def plot_beat_signal_analysis(self, output_dir: str = "photonics") -> None:
        """Plot beat signal analysis."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate beat signal
        beat_data = self.calculate_beat_signal()
        
        # FFT analysis
        fft = np.fft.fft(beat_data['beat_signal_noisy'])
        freqs = np.fft.fftfreq(len(beat_data['time']), beat_data['time'][1] - beat_data['time'][0])
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time domain
        axes[0, 0].plot(beat_data['time'] * 1e6, beat_data['beat_signal'], 'b-', linewidth=2, label='Clean')
        axes[0, 0].plot(beat_data['time'] * 1e6, beat_data['beat_signal_noisy'], 'r-', linewidth=1, alpha=0.7, label='Noisy')
        axes[0, 0].set_title('Beat Signal Time Domain')
        axes[0, 0].set_xlabel('Time (μs)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Frequency domain
        positive_freqs = freqs > 0
        axes[0, 1].plot(freqs[positive_freqs] / 1e6, np.abs(fft[positive_freqs]), 'g-', linewidth=2)
        axes[0, 1].set_title('Beat Signal Frequency Domain')
        axes[0, 1].set_xlabel('Frequency (MHz)')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].grid(True)
        
        # Chirp signal
        axes[1, 0].plot(beat_data['time'] * 1e6, np.real(beat_data['chirp_signal']), 'purple', linewidth=2)
        axes[1, 0].set_title('Chirp Signal')
        axes[1, 0].set_xlabel('Time (μs)')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].grid(True)
        
        # System parameters
        range_res = self.calculate_range_resolution()
        max_range = self.calculate_max_range()
        velocity_res = self.calculate_velocity_resolution()
        snr = self.calculate_snr()
        
        axes[1, 1].text(0.1, 0.9, f'Range Resolution: {range_res*1000:.1f} mm', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.8, f'Max Range: {max_range:.1f} m', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f'Velocity Resolution: {velocity_res:.3f} m/s', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'SNR: {snr:.1f} dB', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f'Beat Frequency: {beat_data["beat_frequency"]/1e6:.1f} MHz', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f'Target Distance: {beat_data["target_distance"]:.1f} m', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('System Parameters')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/beat_signal_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Beat signal analysis plot saved to {output_dir}/beat_signal_analysis.png")

def main():
    """Main function to demonstrate FMCW photonic frontend."""
    print("FMCW LiDAR-on-a-Tabletop: Photonic Frontend")
    print("=" * 60)
    
    # Initialize photonic frontend
    frontend = FMCWPhotonicFrontend()
    
    # Calculate responses
    splitter_response = frontend.calculate_splitter_response()
    mzi_response = frontend.calculate_mzi_response()
    hybrid_response = frontend.calculate_hybrid_response()
    fiber_response = frontend.calculate_fiber_propagation()
    target_response = frontend.calculate_target_reflection()
    beat_data = frontend.calculate_beat_signal()
    
    # Calculate system parameters
    range_resolution = frontend.calculate_range_resolution()
    max_range = frontend.calculate_max_range()
    velocity_resolution = frontend.calculate_velocity_resolution()
    snr = frontend.calculate_snr()
    
    print(f"Range Resolution: {range_resolution*1000:.1f} mm")
    print(f"Max Range: {max_range:.1f} m")
    print(f"Velocity Resolution: {velocity_resolution:.3f} m/s")
    print(f"SNR: {snr:.1f} dB")
    print(f"Beat Frequency: {beat_data['beat_frequency']/1e6:.1f} MHz")
    
    # Plot results
    frontend.plot_photonic_response()
    frontend.plot_beat_signal_analysis()
    
    print("Photonic frontend analysis complete!")

if __name__ == "__main__":
    main()