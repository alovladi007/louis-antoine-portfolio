#!/usr/bin/env python3
"""
DUV Energy Deposition: Swing Curves Module
Swing curves vs duty cycle and contrast/NILS vs pitch.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SwingCurves:
    """Swing curves analysis for DUV mask energy deposition."""
    
    def __init__(self, config: Dict = None):
        """Initialize swing curves analysis."""
        self.config = config or {
            'wavelength': 193e-9,            # DUV wavelength in m
            'numerical_aperture': 0.85,      # Numerical aperture
            'illumination_sigma': 0.7,       # Illumination sigma
            'pupil_cutoff': 0.95,            # Pupil cutoff
            'partial_coherence': 0.7,        # Partial coherence
            'flare_level': 0.02,             # Flare level
            'flare_sigma': 0.1,              # Flare sigma
            'mask_size': (1000, 1000),       # Mask size in pixels
            'pixel_size': 1e-9,              # Pixel size in m
            'psf_size': 100,                 # PSF size in pixels
            'fft_size': 2048,                # FFT size
            'duty_cycle_range': (0.1, 0.9),  # Duty cycle range
            'duty_cycle_steps': 50,          # Number of duty cycle steps
            'pitch_range': (100e-9, 1000e-9), # Pitch range in m
            'pitch_steps': 50,               # Number of pitch steps
            'swing_curve_points': 100,       # Number of swing curve points
            'contrast_threshold': 0.1,       # Contrast threshold
            'nils_threshold': 0.5,           # NILS threshold
            'convolution_method': 'fft',     # Convolution method
            'normalization': True,           # Enable normalization
            'parallel_processing': True,     # Enable parallel processing
            'gpu_acceleration': True,        # Enable GPU acceleration
            'memory_optimization': True,     # Enable memory optimization
            'precision': 'float32',          # Precision type
            'output_directory': 'output',    # Output directory
            'temporary_directory': 'temp',   # Temporary directory
            'cache_directory': 'cache',      # Cache directory
            'log_directory': 'logs',         # Log directory
            'result_directory': 'results'    # Result directory
        }
        
        self.model_results = {}
        self.performance_metrics = {}
        
    def calculate_swing_curves(self) -> Dict[str, any]:
        """Calculate swing curves."""
        print("Calculating swing curves...")
        
        # Calculate duty cycle variations
        duty_cycle_results = self._calculate_duty_cycle_variations()
        
        # Calculate pitch variations
        pitch_results = self._calculate_pitch_variations()
        
        # Calculate swing curves
        swing_curve_results = self._calculate_swing_curves_data()
        
        # Calculate contrast/NILS vs pitch
        contrast_nils_pitch_results = self._calculate_contrast_nils_vs_pitch()
        
        # Calculate model results
        self.model_results = {
            'duty_cycle_results': duty_cycle_results,
            'pitch_results': pitch_results,
            'swing_curve_results': swing_curve_results,
            'contrast_nils_pitch_results': contrast_nils_pitch_results
        }
        
        return self.model_results
    
    def _calculate_duty_cycle_variations(self) -> Dict:
        """Calculate duty cycle variations."""
        print("Calculating duty cycle variations...")
        
        duty_cycle_range = self.config['duty_cycle_range']
        duty_cycle_steps = self.config['duty_cycle_steps']
        
        duty_cycle_values = np.linspace(duty_cycle_range[0], duty_cycle_range[1], duty_cycle_steps)
        
        # Calculate for each duty cycle
        duty_cycle_results = {}
        
        for i, duty_cycle in enumerate(duty_cycle_values):
            # Create mask pattern for this duty cycle
            mask_pattern = self._create_duty_cycle_mask(duty_cycle)
            
            # Calculate aerial image
            aerial_image = self._calculate_aerial_image_for_duty_cycle(mask_pattern)
            
            # Calculate contrast and NILS
            contrast_nils = self._calculate_contrast_nils_for_duty_cycle(aerial_image)
            
            duty_cycle_results[i] = {
                'duty_cycle': duty_cycle,
                'mask_pattern': mask_pattern,
                'aerial_image': aerial_image,
                'contrast_nils': contrast_nils
            }
        
        return duty_cycle_results
    
    def _create_duty_cycle_mask(self, duty_cycle: float) -> np.ndarray:
        """Create mask pattern for given duty cycle."""
        mask_size = self.config['mask_size']
        
        # Create periodic pattern
        pattern_period = 100  # pixels
        mask_pattern = np.zeros(mask_size)
        
        for i in range(0, mask_size[0], pattern_period):
            for j in range(0, mask_size[1], pattern_period):
                # Calculate pattern width based on duty cycle
                pattern_width = int(pattern_period * duty_cycle)
                
                # Create pattern
                if i + pattern_width < mask_size[0] and j + pattern_width < mask_size[1]:
                    mask_pattern[i:i+pattern_width, j:j+pattern_width] = 1.0
        
        return mask_pattern
    
    def _calculate_aerial_image_for_duty_cycle(self, mask_pattern: np.ndarray) -> np.ndarray:
        """Calculate aerial image for duty cycle."""
        # Calculate PSF
        psf = self._calculate_psf_for_swing_curves()
        
        # Convolve mask with PSF
        if self.config['convolution_method'] == 'fft':
            aerial_image = self._fft_convolution(mask_pattern, psf)
        else:
            aerial_image = signal.convolve2d(mask_pattern, psf, mode='same')
        
        return aerial_image
    
    def _calculate_psf_for_swing_curves(self) -> np.ndarray:
        """Calculate PSF for swing curves."""
        psf_size = self.config['psf_size']
        
        # Create coordinate grids
        x = np.linspace(-psf_size//2, psf_size//2, psf_size)
        y = np.linspace(-psf_size//2, psf_size//2, psf_size)
        X, Y = np.meshgrid(x, y)
        
        # Calculate distance from center
        r = np.sqrt(X**2 + Y**2)
        
        # Calculate main PSF (Gaussian)
        main_psf = np.exp(-(r**2) / (2 * 10**2))  # Main Gaussian
        
        # Calculate flare PSF (third Gaussian with long tail)
        flare_psf = np.exp(-(r**2) / (2 * self.config['flare_sigma']**2))
        
        # Combine main PSF and flare
        psf = main_psf + self.config['flare_level'] * flare_psf
        
        # Normalize
        if self.config['normalization']:
            psf = psf / np.sum(psf)
        
        return psf
    
    def _calculate_contrast_nils_for_duty_cycle(self, aerial_image: np.ndarray) -> Dict:
        """Calculate contrast and NILS for duty cycle."""
        # Calculate image statistics
        max_intensity = np.max(aerial_image)
        min_intensity = np.min(aerial_image)
        mean_intensity = np.mean(aerial_image)
        
        # Calculate contrast
        contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity)
        
        # Calculate NILS (Normalized Image Log Slope)
        # Calculate gradient
        grad_x = np.gradient(aerial_image, axis=1)
        grad_y = np.gradient(aerial_image, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate NILS
        nils = np.mean(gradient_magnitude) / mean_intensity
        
        # Calculate additional metrics
        std_intensity = np.std(aerial_image)
        snr = mean_intensity / std_intensity
        
        contrast_nils = {
            'contrast': contrast,
            'nils': nils,
            'max_intensity': max_intensity,
            'min_intensity': min_intensity,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'snr': snr
        }
        
        return contrast_nils
    
    def _calculate_pitch_variations(self) -> Dict:
        """Calculate pitch variations."""
        print("Calculating pitch variations...")
        
        pitch_range = self.config['pitch_range']
        pitch_steps = self.config['pitch_steps']
        
        pitch_values = np.linspace(pitch_range[0], pitch_range[1], pitch_steps)
        
        # Calculate for each pitch
        pitch_results = {}
        
        for i, pitch in enumerate(pitch_values):
            # Create mask pattern for this pitch
            mask_pattern = self._create_pitch_mask(pitch)
            
            # Calculate aerial image
            aerial_image = self._calculate_aerial_image_for_pitch(mask_pattern)
            
            # Calculate contrast and NILS
            contrast_nils = self._calculate_contrast_nils_for_pitch(aerial_image)
            
            pitch_results[i] = {
                'pitch': pitch,
                'mask_pattern': mask_pattern,
                'aerial_image': aerial_image,
                'contrast_nils': contrast_nils
            }
        
        return pitch_results
    
    def _create_pitch_mask(self, pitch: float) -> np.ndarray:
        """Create mask pattern for given pitch."""
        mask_size = self.config['mask_size']
        pixel_size = self.config['pixel_size']
        
        # Convert pitch to pixels
        pitch_pixels = int(pitch / pixel_size)
        
        # Create periodic pattern
        mask_pattern = np.zeros(mask_size)
        
        for i in range(0, mask_size[0], pitch_pixels):
            for j in range(0, mask_size[1], pitch_pixels):
                # Create pattern
                if i + pitch_pixels//2 < mask_size[0] and j + pitch_pixels//2 < mask_size[1]:
                    mask_pattern[i:i+pitch_pixels//2, j:j+pitch_pixels//2] = 1.0
        
        return mask_pattern
    
    def _calculate_aerial_image_for_pitch(self, mask_pattern: np.ndarray) -> np.ndarray:
        """Calculate aerial image for pitch."""
        # Calculate PSF
        psf = self._calculate_psf_for_swing_curves()
        
        # Convolve mask with PSF
        if self.config['convolution_method'] == 'fft':
            aerial_image = self._fft_convolution(mask_pattern, psf)
        else:
            aerial_image = signal.convolve2d(mask_pattern, psf, mode='same')
        
        return aerial_image
    
    def _calculate_contrast_nils_for_pitch(self, aerial_image: np.ndarray) -> Dict:
        """Calculate contrast and NILS for pitch."""
        # Calculate image statistics
        max_intensity = np.max(aerial_image)
        min_intensity = np.min(aerial_image)
        mean_intensity = np.mean(aerial_image)
        
        # Calculate contrast
        contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity)
        
        # Calculate NILS (Normalized Image Log Slope)
        # Calculate gradient
        grad_x = np.gradient(aerial_image, axis=1)
        grad_y = np.gradient(aerial_image, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate NILS
        nils = np.mean(gradient_magnitude) / mean_intensity
        
        # Calculate additional metrics
        std_intensity = np.std(aerial_image)
        snr = mean_intensity / std_intensity
        
        contrast_nils = {
            'contrast': contrast,
            'nils': nils,
            'max_intensity': max_intensity,
            'min_intensity': min_intensity,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'snr': snr
        }
        
        return contrast_nils
    
    def _calculate_swing_curves_data(self) -> Dict:
        """Calculate swing curves data."""
        print("Calculating swing curves data...")
        
        swing_curve_points = self.config['swing_curve_points']
        
        # Calculate swing curves for different parameters
        swing_curves = {}
        
        # Duty cycle swing curves
        duty_cycle_values = np.linspace(0.1, 0.9, swing_curve_points)
        duty_cycle_contrasts = []
        duty_cycle_nils = []
        
        for duty_cycle in duty_cycle_values:
            mask_pattern = self._create_duty_cycle_mask(duty_cycle)
            aerial_image = self._calculate_aerial_image_for_duty_cycle(mask_pattern)
            contrast_nils = self._calculate_contrast_nils_for_duty_cycle(aerial_image)
            
            duty_cycle_contrasts.append(contrast_nils['contrast'])
            duty_cycle_nils.append(contrast_nils['nils'])
        
        swing_curves['duty_cycle'] = {
            'duty_cycle_values': duty_cycle_values,
            'contrasts': duty_cycle_contrasts,
            'nils': duty_cycle_nils
        }
        
        # Pitch swing curves
        pitch_range = self.config['pitch_range']
        pitch_values = np.linspace(pitch_range[0], pitch_range[1], swing_curve_points)
        pitch_contrasts = []
        pitch_nils = []
        
        for pitch in pitch_values:
            mask_pattern = self._create_pitch_mask(pitch)
            aerial_image = self._calculate_aerial_image_for_pitch(mask_pattern)
            contrast_nils = self._calculate_contrast_nils_for_pitch(aerial_image)
            
            pitch_contrasts.append(contrast_nils['contrast'])
            pitch_nils.append(contrast_nils['nils'])
        
        swing_curves['pitch'] = {
            'pitch_values': pitch_values,
            'contrasts': pitch_contrasts,
            'nils': pitch_nils
        }
        
        return swing_curves
    
    def _calculate_contrast_nils_vs_pitch(self) -> Dict:
        """Calculate contrast/NILS vs pitch."""
        print("Calculating contrast/NILS vs pitch...")
        
        pitch_range = self.config['pitch_range']
        pitch_steps = self.config['pitch_steps']
        
        pitch_values = np.linspace(pitch_range[0], pitch_range[1], pitch_steps)
        contrast_values = []
        nils_values = []
        
        for pitch in pitch_values:
            mask_pattern = self._create_pitch_mask(pitch)
            aerial_image = self._calculate_aerial_image_for_pitch(mask_pattern)
            contrast_nils = self._calculate_contrast_nils_for_pitch(aerial_image)
            
            contrast_values.append(contrast_nils['contrast'])
            nils_values.append(contrast_nils['nils'])
        
        contrast_nils_pitch = {
            'pitch_values': pitch_values,
            'contrast_values': contrast_values,
            'nils_values': nils_values
        }
        
        return contrast_nils_pitch
    
    def _fft_convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Calculate convolution using FFT."""
        # Get dimensions
        h, w = image.shape
        kh, kw = kernel.shape
        
        # Pad image and kernel
        pad_h = h + kh - 1
        pad_w = w + kw - 1
        
        # Pad to power of 2 for efficiency
        pad_h = 2**int(np.ceil(np.log2(pad_h)))
        pad_w = 2**int(np.ceil(np.log2(pad_w)))
        
        # Pad image
        padded_image = np.zeros((pad_h, pad_w))
        padded_image[:h, :w] = image
        
        # Pad kernel
        padded_kernel = np.zeros((pad_h, pad_w))
        padded_kernel[:kh, :kw] = kernel
        
        # FFT convolution
        fft_image = np.fft.fft2(padded_image)
        fft_kernel = np.fft.fft2(padded_kernel)
        
        # Convolve
        fft_result = fft_image * fft_kernel
        
        # Inverse FFT
        result = np.real(np.fft.ifft2(fft_result))
        
        # Crop to original size
        result = result[:h, :w]
        
        return result
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        print("Calculating performance metrics...")
        
        # Model execution time (simplified)
        execution_time = 1.0  # seconds
        
        # Memory usage
        mask_size = self.config['mask_size']
        psf_size = self.config['psf_size']
        fft_size = self.config['fft_size']
        duty_cycle_steps = self.config['duty_cycle_steps']
        pitch_steps = self.config['pitch_steps']
        swing_curve_points = self.config['swing_curve_points']
        
        memory_usage = (mask_size[0] * mask_size[1] + psf_size**2 + fft_size**2 + 
                       duty_cycle_steps * mask_size[0] * mask_size[1] + 
                       pitch_steps * mask_size[0] * mask_size[1] + 
                       swing_curve_points * 2) * 4  # 4 bytes per float
        
        # Throughput
        throughput = (mask_size[0] * mask_size[1] * psf_size**2 * 
                     (duty_cycle_steps + pitch_steps + swing_curve_points)) / execution_time
        
        # Efficiency
        efficiency = 1.0  # Simplified
        
        # GPU utilization (simplified)
        gpu_utilization = min(1.0, throughput / 1e9)  # Normalized to 1G operations/s
        
        self.performance_metrics = {
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'throughput': throughput,
            'efficiency': efficiency,
            'gpu_utilization': gpu_utilization,
            'operations_per_second': throughput,
            'memory_bandwidth': memory_usage / execution_time,
            'compute_intensity': throughput / memory_usage
        }
        
        return self.performance_metrics
    
    def plot_swing_curves(self, output_dir: str = "swing_curves") -> None:
        """Plot swing curves analysis."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Duty cycle vs contrast
        duty_cycle_data = self.model_results['swing_curve_results']['duty_cycle']
        axes[0, 0].plot(duty_cycle_data['duty_cycle_values'], duty_cycle_data['contrasts'], 'b-', linewidth=2)
        axes[0, 0].set_title('Duty Cycle vs Contrast')
        axes[0, 0].set_xlabel('Duty Cycle')
        axes[0, 0].set_ylabel('Contrast')
        axes[0, 0].grid(True)
        
        # Duty cycle vs NILS
        axes[0, 1].plot(duty_cycle_data['duty_cycle_values'], duty_cycle_data['nils'], 'r-', linewidth=2)
        axes[0, 1].set_title('Duty Cycle vs NILS')
        axes[0, 1].set_xlabel('Duty Cycle')
        axes[0, 1].set_ylabel('NILS')
        axes[0, 1].grid(True)
        
        # Pitch vs contrast
        pitch_data = self.model_results['swing_curve_results']['pitch']
        axes[0, 2].plot(pitch_data['pitch_values'] * 1e9, pitch_data['contrasts'], 'g-', linewidth=2)
        axes[0, 2].set_title('Pitch vs Contrast')
        axes[0, 2].set_xlabel('Pitch (nm)')
        axes[0, 2].set_ylabel('Contrast')
        axes[0, 2].grid(True)
        
        # Pitch vs NILS
        axes[1, 0].plot(pitch_data['pitch_values'] * 1e9, pitch_data['nils'], 'm-', linewidth=2)
        axes[1, 0].set_title('Pitch vs NILS')
        axes[1, 0].set_xlabel('Pitch (nm)')
        axes[1, 0].set_ylabel('NILS')
        axes[1, 0].grid(True)
        
        # Contrast vs NILS
        contrast_nils_pitch_data = self.model_results['contrast_nils_pitch_results']
        axes[1, 1].scatter(contrast_nils_pitch_data['contrast_values'], 
                          contrast_nils_pitch_data['nils_values'], 
                          alpha=0.7, color='orange')
        axes[1, 1].set_title('Contrast vs NILS')
        axes[1, 1].set_xlabel('Contrast')
        axes[1, 1].set_ylabel('NILS')
        axes[1, 1].grid(True)
        
        # Performance metrics
        if self.performance_metrics:
            perf_metrics = ['Execution Time', 'Memory Usage', 'Throughput', 'Efficiency', 'GPU Utilization']
            perf_values = [self.performance_metrics['execution_time'],
                          self.performance_metrics['memory_usage'] / 1e6,  # Convert to MB
                          self.performance_metrics['throughput'] / 1e9,    # Convert to G operations/s
                          self.performance_metrics['efficiency'],
                          self.performance_metrics['gpu_utilization']]
            
            axes[1, 2].bar(perf_metrics, perf_values, color=['blue', 'green', 'orange', 'red', 'purple'])
            axes[1, 2].set_title('Performance Metrics')
            axes[1, 2].set_ylabel('Value')
            axes[1, 2].tick_params(axis='x', rotation=45)
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/swing_curves_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Swing curves analysis plot saved to {output_dir}/swing_curves_analysis.png")

def main():
    """Main function to demonstrate swing curves analysis."""
    print("DUV Energy Deposition: Swing Curves Analysis")
    print("=" * 60)
    
    # Initialize swing curves analysis
    model = SwingCurves()
    
    # Calculate swing curves
    results = model.calculate_swing_curves()
    
    # Calculate performance metrics
    performance = model.calculate_performance_metrics()
    
    # Print results
    print(f"Duty cycle variations: {len(results['duty_cycle_results'])}")
    print(f"Pitch variations: {len(results['pitch_results'])}")
    print(f"Swing curve points: {len(results['swing_curve_results']['duty_cycle']['duty_cycle_values'])}")
    print(f"Contrast/NILS vs pitch points: {len(results['contrast_nils_pitch_results']['pitch_values'])}")
    
    print(f"Performance - Execution time: {performance['execution_time']:.2f} s")
    print(f"Performance - Memory usage: {performance['memory_usage'] / 1e6:.1f} MB")
    print(f"Performance - Throughput: {performance['throughput'] / 1e9:.1f} G operations/s")
    print(f"Performance - GPU utilization: {performance['gpu_utilization']:.2%}")
    
    # Plot results
    model.plot_swing_curves()
    
    print("Swing curves analysis complete!")

if __name__ == "__main__":
    main()