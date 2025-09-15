#!/usr/bin/env python3
"""
FMCW LiDAR-on-a-Tabletop: DSP Module
Range-Doppler FFT, CFAR detection, calibration, and real-time processing.
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

class FMCWDSPProcessor:
    """FMCW LiDAR DSP processor for range and velocity extraction."""
    
    def __init__(self, config: Dict = None):
        """Initialize FMCW DSP processor."""
        self.config = config or {
            'sampling_rate': 100e6,          # ADC sampling rate in Hz
            'chirp_duration': 1e-3,          # Chirp duration in s
            'bandwidth': 100e9,              # Chirp bandwidth in Hz
            'wavelength': 1550e-9,           # Operating wavelength in m
            'fft_size': 1024,                # FFT size
            'window_type': 'hann',           # Window type
            'cfar_threshold': 0.1,           # CFAR threshold
            'cfar_guard_cells': 4,           # CFAR guard cells
            'cfar_training_cells': 8,        # CFAR training cells
            'range_bins': 512,               # Number of range bins
            'velocity_bins': 256,            # Number of velocity bins
            'max_range': 100.0,              # Maximum range in m
            'max_velocity': 50.0,            # Maximum velocity in m/s
            'calibration_points': 100,       # Number of calibration points
            'noise_floor': -80,              # Noise floor in dB
            'target_threshold': -60,         # Target detection threshold in dB
            'range_resolution': 0.1,         # Range resolution in m
            'velocity_resolution': 0.1,      # Velocity resolution in m/s
            'super_resolution_factor': 4,    # Super-resolution factor
            'phase_unwrap_threshold': np.pi, # Phase unwrap threshold
            'calibration_frequency': 1e6,    # Calibration frequency in Hz
            'temperature': 25,               # Operating temperature in °C
            'pressure': 101325,              # Atmospheric pressure in Pa
            'humidity': 50                   # Relative humidity in %
        }
        
        self.range_fft = None
        self.velocity_fft = None
        self.range_doppler_map = None
        self.detected_targets = None
        self.calibration_data = None
        
    def window_signal(self, signal: np.ndarray, window_type: str = None) -> np.ndarray:
        """Apply windowing to signal."""
        if window_type is None:
            window_type = self.config['window_type']
        
        if window_type == 'hann':
            window = np.hanning(len(signal))
        elif window_type == 'hamming':
            window = np.hamming(len(signal))
        elif window_type == 'blackman':
            window = np.blackman(len(signal))
        elif window_type == 'kaiser':
            window = np.kaiser(len(signal), 5)
        else:
            window = np.ones(len(signal))
        
        return signal * window
    
    def calculate_range_fft(self, beat_signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate range FFT."""
        print("Calculating range FFT...")
        
        # Apply windowing
        windowed_signal = self.window_signal(beat_signal)
        
        # Zero-padding to FFT size
        fft_size = self.config['fft_size']
        if len(windowed_signal) < fft_size:
            padded_signal = np.pad(windowed_signal, (0, fft_size - len(windowed_signal)))
        else:
            padded_signal = windowed_signal[:fft_size]
        
        # FFT
        fft_result = np.fft.fft(padded_signal)
        fft_magnitude = np.abs(fft_result)
        fft_phase = np.angle(fft_result)
        
        # Frequency bins
        sampling_rate = self.config['sampling_rate']
        freq_bins = np.fft.fftfreq(fft_size, 1/sampling_rate)
        
        # Range bins
        c = 3e8  # Speed of light
        chirp_slope = self.config['bandwidth'] / self.config['chirp_duration']
        range_bins = freq_bins * c / (2 * chirp_slope)
        
        # Keep only positive frequencies
        positive_mask = freq_bins >= 0
        range_bins = range_bins[positive_mask]
        fft_magnitude = fft_magnitude[positive_mask]
        fft_phase = fft_phase[positive_mask]
        
        self.range_fft = {
            'range_bins': range_bins,
            'magnitude': fft_magnitude,
            'phase': fft_phase,
            'frequency_bins': freq_bins[positive_mask]
        }
        
        return self.range_fft
    
    def calculate_velocity_fft(self, range_fft_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate velocity FFT from range FFT data."""
        print("Calculating velocity FFT...")
        
        # For velocity calculation, we need multiple chirps
        # This is a simplified version using a single chirp
        range_magnitude = range_fft_data['magnitude']
        
        # Simulate multiple chirps for velocity calculation
        n_chirps = self.config['velocity_bins']
        range_doppler_data = np.zeros((n_chirps, len(range_magnitude)))
        
        # Generate synthetic velocity data
        for i in range(n_chirps):
            # Add velocity-dependent phase shift
            velocity = (i - n_chirps//2) * self.config['max_velocity'] / n_chirps
            phase_shift = 4 * np.pi * velocity * self.config['chirp_duration'] / self.config['wavelength']
            range_doppler_data[i] = range_magnitude * np.exp(1j * phase_shift)
        
        # FFT along velocity dimension
        velocity_fft = np.fft.fft(range_doppler_data, axis=0)
        velocity_magnitude = np.abs(velocity_fft)
        
        # Velocity bins
        velocity_bins = np.linspace(-self.config['max_velocity'], 
                                   self.config['max_velocity'], n_chirps)
        
        self.velocity_fft = {
            'velocity_bins': velocity_bins,
            'range_doppler_map': velocity_magnitude,
            'range_bins': range_fft_data['range_bins']
        }
        
        return self.velocity_fft
    
    def calculate_cfar_detection(self, range_fft_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate CFAR detection."""
        print("Calculating CFAR detection...")
        
        magnitude = range_fft_data['magnitude']
        threshold = self.config['cfar_threshold']
        guard_cells = self.config['cfar_guard_cells']
        training_cells = self.config['cfar_training_cells']
        
        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-12)
        
        # CFAR detection
        detections = np.zeros_like(magnitude_db, dtype=bool)
        cfar_threshold = np.zeros_like(magnitude_db)
        
        for i in range(len(magnitude_db)):
            # Training cells
            start_idx = max(0, i - guard_cells - training_cells)
            end_idx = min(len(magnitude_db), i + guard_cells + training_cells)
            
            # Exclude guard cells
            training_indices = list(range(start_idx, i - guard_cells)) + \
                             list(range(i + guard_cells, end_idx))
            
            if len(training_indices) > 0:
                # Calculate noise floor
                noise_floor = np.mean(magnitude_db[training_indices])
                cfar_threshold[i] = noise_floor + threshold
                
                # Detection
                detections[i] = magnitude_db[i] > cfar_threshold[i]
        
        return {
            'detections': detections,
            'cfar_threshold': cfar_threshold,
            'magnitude_db': magnitude_db,
            'noise_floor': np.mean(magnitude_db)
        }
    
    def calculate_phase_unwrap(self, phase: np.ndarray) -> np.ndarray:
        """Calculate phase unwrapping for super-resolution."""
        print("Calculating phase unwrapping...")
        
        # Phase unwrapping
        unwrapped_phase = np.unwrap(phase)
        
        # Super-resolution using phase
        phase_gradient = np.gradient(unwrapped_phase)
        
        # Sub-bin resolution
        sub_bin_offset = phase_gradient / (2 * np.pi)
        
        return {
            'unwrapped_phase': unwrapped_phase,
            'phase_gradient': phase_gradient,
            'sub_bin_offset': sub_bin_offset
        }
    
    def calculate_range_velocity_extraction(self, range_fft_data: Dict[str, np.ndarray], 
                                          velocity_fft_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate range and velocity extraction."""
        print("Calculating range and velocity extraction...")
        
        # Range extraction
        range_magnitude = range_fft_data['magnitude']
        range_bins = range_fft_data['range_bins']
        
        # Find peaks in range
        range_peaks, _ = signal.find_peaks(range_magnitude, height=np.max(range_magnitude) * 0.1)
        
        # Velocity extraction
        velocity_magnitude = velocity_fft_data['range_doppler_map']
        velocity_bins = velocity_fft_data['velocity_bins']
        
        # Find peaks in velocity
        velocity_peaks = []
        for i in range(len(range_peaks)):
            range_idx = range_peaks[i]
            velocity_profile = velocity_magnitude[:, range_idx]
            peaks, _ = signal.find_peaks(velocity_profile, height=np.max(velocity_profile) * 0.1)
            velocity_peaks.extend(peaks)
        
        # Extract target information
        targets = []
        for range_peak in range_peaks:
            for velocity_peak in velocity_peaks:
                range_val = range_bins[range_peak]
                velocity_val = velocity_bins[velocity_peak]
                
                # Calculate SNR
                signal_power = range_magnitude[range_peak]**2
                noise_power = np.mean(range_magnitude**2)
                snr = 10 * np.log10(signal_power / noise_power)
                
                if snr > self.config['target_threshold']:
                    targets.append({
                        'range': range_val,
                        'velocity': velocity_val,
                        'snr': snr,
                        'range_bin': range_peak,
                        'velocity_bin': velocity_peak
                    })
        
        self.detected_targets = targets
        
        return {
            'targets': targets,
            'range_peaks': range_peaks,
            'velocity_peaks': velocity_peaks,
            'num_targets': len(targets)
        }
    
    def calculate_calibration(self) -> Dict[str, np.ndarray]:
        """Calculate system calibration."""
        print("Calculating system calibration...")
        
        # Calibration parameters
        calibration_freq = self.config['calibration_frequency']
        calibration_points = self.config['calibration_points']
        
        # Generate calibration signal
        t = np.linspace(0, self.config['chirp_duration'], calibration_points)
        calibration_signal = np.sin(2 * np.pi * calibration_freq * t)
        
        # Apply system response
        system_response = self._calculate_system_response(calibration_freq)
        calibrated_signal = calibration_signal * system_response
        
        # Calculate calibration factors
        calibration_factor = np.mean(calibrated_signal) / np.mean(calibration_signal)
        phase_calibration = np.angle(calibration_factor)
        amplitude_calibration = np.abs(calibration_factor)
        
        # Temperature compensation
        temperature = self.config['temperature']
        temp_coefficient = 0.001  # 0.1% per °C
        temp_compensation = 1 + temp_coefficient * (temperature - 25)
        
        self.calibration_data = {
            'calibration_factor': calibration_factor,
            'phase_calibration': phase_calibration,
            'amplitude_calibration': amplitude_calibration,
            'temperature_compensation': temp_compensation,
            'calibration_signal': calibration_signal,
            'calibrated_signal': calibrated_signal
        }
        
        return self.calibration_data
    
    def _calculate_system_response(self, frequency: float) -> complex:
        """Calculate system response at given frequency."""
        # Simplified system response
        # In practice, this would be measured or calculated from component models
        
        # Frequency response
        bandwidth = self.config['bandwidth']
        response = 1 / (1 + 1j * frequency / bandwidth)
        
        # Phase response
        phase = np.angle(response)
        
        # Amplitude response
        amplitude = np.abs(response)
        
        return amplitude * np.exp(1j * phase)
    
    def calculate_real_time_processing(self, beat_signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate real-time processing pipeline."""
        print("Calculating real-time processing pipeline...")
        
        # Step 1: Range FFT
        range_fft_data = self.calculate_range_fft(beat_signal)
        
        # Step 2: CFAR detection
        cfar_data = self.calculate_cfar_detection(range_fft_data)
        
        # Step 3: Velocity FFT
        velocity_fft_data = self.calculate_velocity_fft(range_fft_data)
        
        # Step 4: Phase unwrapping
        phase_unwrap_data = self.calculate_phase_unwrap(range_fft_data['phase'])
        
        # Step 5: Range and velocity extraction
        target_data = self.calculate_range_velocity_extraction(range_fft_data, velocity_fft_data)
        
        # Step 6: Calibration
        calibration_data = self.calculate_calibration()
        
        return {
            'range_fft': range_fft_data,
            'cfar_detection': cfar_data,
            'velocity_fft': velocity_fft_data,
            'phase_unwrap': phase_unwrap_data,
            'target_extraction': target_data,
            'calibration': calibration_data
        }
    
    def plot_dsp_analysis(self, processing_data: Dict[str, np.ndarray], output_dir: str = "dsp") -> None:
        """Plot DSP analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data
        range_fft = processing_data['range_fft']
        cfar_data = processing_data['cfar_detection']
        velocity_fft = processing_data['velocity_fft']
        target_data = processing_data['target_extraction']
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Range FFT
        axes[0, 0].plot(range_fft['range_bins'], 20 * np.log10(range_fft['magnitude'] + 1e-12), 'b-', linewidth=2)
        axes[0, 0].set_title('Range FFT')
        axes[0, 0].set_xlabel('Range (m)')
        axes[0, 0].set_ylabel('Magnitude (dB)')
        axes[0, 0].grid(True)
        
        # CFAR detection
        axes[0, 1].plot(range_fft['range_bins'], cfar_data['magnitude_db'], 'b-', linewidth=2, label='Signal')
        axes[0, 1].plot(range_fft['range_bins'], cfar_data['cfar_threshold'], 'r--', linewidth=2, label='Threshold')
        axes[0, 1].set_title('CFAR Detection')
        axes[0, 1].set_xlabel('Range (m)')
        axes[0, 1].set_ylabel('Magnitude (dB)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Range-Doppler map
        im = axes[0, 2].imshow(20 * np.log10(velocity_fft['range_doppler_map'] + 1e-12), 
                               aspect='auto', origin='lower', cmap='jet')
        axes[0, 2].set_title('Range-Doppler Map')
        axes[0, 2].set_xlabel('Range Bin')
        axes[0, 2].set_ylabel('Velocity Bin')
        plt.colorbar(im, ax=axes[0, 2], label='Magnitude (dB)')
        
        # Phase unwrapping
        phase_unwrap = processing_data['phase_unwrap']
        axes[1, 0].plot(range_fft['range_bins'], phase_unwrap['unwrapped_phase'], 'g-', linewidth=2)
        axes[1, 0].set_title('Phase Unwrapping')
        axes[1, 0].set_xlabel('Range (m)')
        axes[1, 0].set_ylabel('Phase (rad)')
        axes[1, 0].grid(True)
        
        # Sub-bin resolution
        axes[1, 1].plot(range_fft['range_bins'], phase_unwrap['sub_bin_offset'], 'purple', linewidth=2)
        axes[1, 1].set_title('Sub-bin Resolution')
        axes[1, 1].set_xlabel('Range (m)')
        axes[1, 1].set_ylabel('Sub-bin Offset')
        axes[1, 1].grid(True)
        
        # Detected targets
        if target_data['targets']:
            targets = target_data['targets']
            ranges = [t['range'] for t in targets]
            velocities = [t['velocity'] for t in targets]
            snrs = [t['snr'] for t in targets]
            
            scatter = axes[1, 2].scatter(ranges, velocities, c=snrs, cmap='viridis', s=100)
            axes[1, 2].set_title('Detected Targets')
            axes[1, 2].set_xlabel('Range (m)')
            axes[1, 2].set_ylabel('Velocity (m/s)')
            plt.colorbar(scatter, ax=axes[1, 2], label='SNR (dB)')
        else:
            axes[1, 2].text(0.5, 0.5, 'No targets detected', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Detected Targets')
            axes[1, 2].set_xlabel('Range (m)')
            axes[1, 2].set_ylabel('Velocity (m/s)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dsp_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"DSP analysis plot saved to {output_dir}/dsp_analysis.png")
    
    def plot_calibration_analysis(self, output_dir: str = "dsp") -> None:
        """Plot calibration analysis."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate calibration
        calibration_data = self.calculate_calibration()
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Calibration signal
        axes[0, 0].plot(calibration_data['calibration_signal'], 'b-', linewidth=2, label='Input')
        axes[0, 0].plot(calibration_data['calibrated_signal'], 'r-', linewidth=2, label='Output')
        axes[0, 0].set_title('Calibration Signal')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Calibration factors
        factors = ['Amplitude', 'Phase', 'Temperature']
        values = [calibration_data['amplitude_calibration'], 
                 calibration_data['phase_calibration'],
                 calibration_data['temperature_compensation']]
        
        axes[0, 1].bar(factors, values, color=['blue', 'green', 'orange'])
        axes[0, 1].set_title('Calibration Factors')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].grid(True)
        
        # Frequency response
        frequencies = np.logspace(0, 9, 1000)
        responses = [self._calculate_system_response(f) for f in frequencies]
        magnitudes = [np.abs(r) for r in responses]
        phases = [np.angle(r) for r in responses]
        
        axes[1, 0].semilogx(frequencies, 20 * np.log10(magnitudes), 'b-', linewidth=2)
        axes[1, 0].set_title('System Frequency Response')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Magnitude (dB)')
        axes[1, 0].grid(True)
        
        axes[1, 1].semilogx(frequencies, np.array(phases) * 180 / np.pi, 'r-', linewidth=2)
        axes[1, 1].set_title('System Phase Response')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Phase (degrees)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/calibration_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Calibration analysis plot saved to {output_dir}/calibration_analysis.png")

def main():
    """Main function to demonstrate FMCW DSP processor."""
    print("FMCW LiDAR-on-a-Tabletop: DSP Processor")
    print("=" * 60)
    
    # Initialize DSP processor
    dsp = FMCWDSPProcessor()
    
    # Generate test signal
    t = np.linspace(0, dsp.config['chirp_duration'], 1000)
    beat_freq = 1e6  # 1 MHz beat frequency
    beat_signal = np.sin(2 * np.pi * beat_freq * t) + 0.1 * np.random.randn(len(t))
    
    # Process signal
    processing_data = dsp.calculate_real_time_processing(beat_signal)
    
    # Print results
    target_data = processing_data['target_extraction']
    print(f"Number of targets detected: {target_data['num_targets']}")
    
    if target_data['targets']:
        for i, target in enumerate(target_data['targets']):
            print(f"Target {i+1}: Range = {target['range']:.2f} m, "
                  f"Velocity = {target['velocity']:.2f} m/s, SNR = {target['snr']:.1f} dB")
    
    # Plot results
    dsp.plot_dsp_analysis(processing_data)
    dsp.plot_calibration_analysis()
    
    print("DSP processing complete!")

if __name__ == "__main__":
    main()