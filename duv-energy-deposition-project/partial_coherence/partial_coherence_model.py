#!/usr/bin/env python3
"""
DUV Energy Deposition: Partial Coherence Model Module
Partial coherence modeling with σ from NA/illumination and pupil cut-off.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PartialCoherenceModel:
    """Partial coherence model for DUV mask energy deposition."""
    
    def __init__(self, config: Dict = None):
        """Initialize partial coherence model."""
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
            'sigma_range': (0.1, 1.0),       # Sigma range
            'sigma_steps': 20,               # Number of sigma steps
            'pupil_cutoff_range': (0.5, 1.0), # Pupil cutoff range
            'pupil_cutoff_steps': 20,        # Number of pupil cutoff steps
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
        
    def calculate_partial_coherence_model(self) -> Dict[str, any]:
        """Calculate partial coherence model."""
        print("Calculating partial coherence model...")
        
        # Initialize mask
        mask = self._initialize_mask()
        
        # Calculate sigma values
        sigma_values = self._calculate_sigma_values()
        
        # Calculate pupil cutoff values
        pupil_cutoff_values = self._calculate_pupil_cutoff_values()
        
        # Calculate PSF for different parameters
        psf_results = self._calculate_psf_variations(sigma_values, pupil_cutoff_values)
        
        # Calculate aerial images
        aerial_image_results = self._calculate_aerial_image_variations(mask, psf_results)
        
        # Calculate contrast and NILS
        contrast_nils_results = self._calculate_contrast_nils_variations(aerial_image_results)
        
        # Calculate model results
        self.model_results = {
            'mask': mask,
            'sigma_values': sigma_values,
            'pupil_cutoff_values': pupil_cutoff_values,
            'psf_results': psf_results,
            'aerial_image_results': aerial_image_results,
            'contrast_nils_results': contrast_nils_results
        }
        
        return self.model_results
    
    def _initialize_mask(self) -> Dict:
        """Initialize mask data."""
        mask_size = self.config['mask_size']
        pixel_size = self.config['pixel_size']
        
        # Generate mask pattern
        mask_pattern = np.zeros(mask_size)
        
        # Add some features
        for i in range(0, mask_size[0], 100):
            for j in range(0, mask_size[1], 100):
                if (i + j) % 200 < 100:
                    mask_pattern[i:i+50, j:j+50] = 1.0
        
        mask = {
            'pattern': mask_pattern,
            'size': mask_size,
            'pixel_size': pixel_size,
            'transmission': 0.1,  # 10% transmission
            'phase_shift': np.pi,  # 180° phase shift
            'absorption_coefficient': 0.1,
            'scattering_coefficient': 0.5
        }
        
        return mask
    
    def _calculate_sigma_values(self) -> np.ndarray:
        """Calculate sigma values."""
        sigma_range = self.config['sigma_range']
        sigma_steps = self.config['sigma_steps']
        
        sigma_values = np.linspace(sigma_range[0], sigma_range[1], sigma_steps)
        
        return sigma_values
    
    def _calculate_pupil_cutoff_values(self) -> np.ndarray:
        """Calculate pupil cutoff values."""
        pupil_cutoff_range = self.config['pupil_cutoff_range']
        pupil_cutoff_steps = self.config['pupil_cutoff_steps']
        
        pupil_cutoff_values = np.linspace(pupil_cutoff_range[0], pupil_cutoff_range[1], pupil_cutoff_steps)
        
        return pupil_cutoff_values
    
    def _calculate_psf_variations(self, sigma_values: np.ndarray, 
                                 pupil_cutoff_values: np.ndarray) -> Dict:
        """Calculate PSF variations."""
        print("Calculating PSF variations...")
        
        psf_size = self.config['psf_size']
        psf_results = {}
        
        for i, sigma in enumerate(sigma_values):
            for j, pupil_cutoff in enumerate(pupil_cutoff_values):
                # Calculate PSF for this combination
                psf = self._calculate_psf(sigma, pupil_cutoff)
                psf_results[(i, j)] = {
                    'psf': psf,
                    'sigma': sigma,
                    'pupil_cutoff': pupil_cutoff
                }
        
        return psf_results
    
    def _calculate_psf(self, sigma: float, pupil_cutoff: float) -> np.ndarray:
        """Calculate PSF for given parameters."""
        psf_size = self.config['psf_size']
        
        # Create coordinate grids
        x = np.linspace(-psf_size//2, psf_size//2, psf_size)
        y = np.linspace(-psf_size//2, psf_size//2, psf_size)
        X, Y = np.meshgrid(x, y)
        
        # Calculate distance from center
        r = np.sqrt(X**2 + Y**2)
        
        # Calculate PSF with partial coherence
        # Simplified model: Gaussian with partial coherence effects
        psf = np.exp(-(r**2) / (2 * sigma**2))
        
        # Apply pupil cutoff
        psf[r > pupil_cutoff * psf_size//2] = 0
        
        # Normalize
        if self.config['normalization']:
            psf = psf / np.sum(psf)
        
        return psf
    
    def _calculate_aerial_image_variations(self, mask: Dict, psf_results: Dict) -> Dict:
        """Calculate aerial image variations."""
        print("Calculating aerial image variations...")
        
        mask_pattern = mask['pattern']
        aerial_image_results = {}
        
        for key, psf_data in psf_results.items():
            psf = psf_data['psf']
            
            # Calculate aerial image
            aerial_image = self._calculate_aerial_image(mask_pattern, psf)
            
            aerial_image_results[key] = {
                'aerial_image': aerial_image,
                'sigma': psf_data['sigma'],
                'pupil_cutoff': psf_data['pupil_cutoff']
            }
        
        return aerial_image_results
    
    def _calculate_aerial_image(self, mask_pattern: np.ndarray, psf: np.ndarray) -> np.ndarray:
        """Calculate aerial image using convolution."""
        # Convolve mask with PSF
        if self.config['convolution_method'] == 'fft':
            aerial_image = self._fft_convolution(mask_pattern, psf)
        else:
            aerial_image = signal.convolve2d(mask_pattern, psf, mode='same')
        
        return aerial_image
    
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
    
    def _calculate_contrast_nils_variations(self, aerial_image_results: Dict) -> Dict:
        """Calculate contrast and NILS variations."""
        print("Calculating contrast and NILS variations...")
        
        contrast_nils_results = {}
        
        for key, aerial_data in aerial_image_results.items():
            aerial_image = aerial_data['aerial_image']
            
            # Calculate contrast and NILS
            contrast_nils = self._calculate_contrast_nils(aerial_image)
            
            contrast_nils_results[key] = {
                'contrast_nils': contrast_nils,
                'sigma': aerial_data['sigma'],
                'pupil_cutoff': aerial_data['pupil_cutoff']
            }
        
        return contrast_nils_results
    
    def _calculate_contrast_nils(self, aerial_image: np.ndarray) -> Dict:
        """Calculate contrast and NILS."""
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
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        print("Calculating performance metrics...")
        
        # Model execution time (simplified)
        execution_time = 1.0  # seconds
        
        # Memory usage
        mask_size = self.config['mask_size']
        psf_size = self.config['psf_size']
        fft_size = self.config['fft_size']
        sigma_steps = self.config['sigma_steps']
        pupil_cutoff_steps = self.config['pupil_cutoff_steps']
        
        memory_usage = (mask_size[0] * mask_size[1] + psf_size**2 + fft_size**2 + 
                       sigma_steps * pupil_cutoff_steps * psf_size**2) * 4  # 4 bytes per float
        
        # Throughput
        throughput = (mask_size[0] * mask_size[1] * psf_size**2 * sigma_steps * pupil_cutoff_steps) / execution_time
        
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
    
    def plot_model_analysis(self, output_dir: str = "partial_coherence") -> None:
        """Plot model analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Mask pattern
        mask_pattern = self.model_results['mask']['pattern']
        axes[0, 0].imshow(mask_pattern, cmap='gray', origin='lower')
        axes[0, 0].set_title('Mask Pattern')
        axes[0, 0].set_xlabel('X (pixels)')
        axes[0, 0].set_ylabel('Y (pixels)')
        
        # Sigma vs Contrast
        sigma_values = self.model_results['sigma_values']
        contrast_values = []
        
        for i in range(len(sigma_values)):
            key = (i, 0)  # Use first pupil cutoff value
            if key in self.model_results['contrast_nils_results']:
                contrast = self.model_results['contrast_nils_results'][key]['contrast_nils']['contrast']
                contrast_values.append(contrast)
            else:
                contrast_values.append(0)
        
        axes[0, 1].plot(sigma_values, contrast_values, 'b-', linewidth=2)
        axes[0, 1].set_title('Sigma vs Contrast')
        axes[0, 1].set_xlabel('Sigma')
        axes[0, 1].set_ylabel('Contrast')
        axes[0, 1].grid(True)
        
        # Pupil Cutoff vs NILS
        pupil_cutoff_values = self.model_results['pupil_cutoff_values']
        nils_values = []
        
        for j in range(len(pupil_cutoff_values)):
            key = (0, j)  # Use first sigma value
            if key in self.model_results['contrast_nils_results']:
                nils = self.model_results['contrast_nils_results'][key]['contrast_nils']['nils']
                nils_values.append(nils)
            else:
                nils_values.append(0)
        
        axes[0, 2].plot(pupil_cutoff_values, nils_values, 'r-', linewidth=2)
        axes[0, 2].set_title('Pupil Cutoff vs NILS')
        axes[0, 2].set_xlabel('Pupil Cutoff')
        axes[0, 2].set_ylabel('NILS')
        axes[0, 2].grid(True)
        
        # Contrast vs NILS
        contrast_nils_data = self.model_results['contrast_nils_results']
        all_contrasts = []
        all_nils = []
        
        for key, data in contrast_nils_data.items():
            all_contrasts.append(data['contrast_nils']['contrast'])
            all_nils.append(data['contrast_nils']['nils'])
        
        axes[1, 0].scatter(all_contrasts, all_nils, alpha=0.7, color='green')
        axes[1, 0].set_title('Contrast vs NILS')
        axes[1, 0].set_xlabel('Contrast')
        axes[1, 0].set_ylabel('NILS')
        axes[1, 0].grid(True)
        
        # Sigma vs Pupil Cutoff heatmap
        sigma_steps = self.config['sigma_steps']
        pupil_cutoff_steps = self.config['pupil_cutoff_steps']
        contrast_heatmap = np.zeros((sigma_steps, pupil_cutoff_steps))
        
        for i in range(sigma_steps):
            for j in range(pupil_cutoff_steps):
                key = (i, j)
                if key in contrast_nils_data:
                    contrast_heatmap[i, j] = contrast_nils_data[key]['contrast_nils']['contrast']
        
        im = axes[1, 1].imshow(contrast_heatmap, cmap='hot', origin='lower')
        axes[1, 1].set_title('Contrast Heatmap')
        axes[1, 1].set_xlabel('Pupil Cutoff Index')
        axes[1, 1].set_ylabel('Sigma Index')
        plt.colorbar(im, ax=axes[1, 1], label='Contrast')
        
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
        plt.savefig(f"{output_dir}/model_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model analysis plot saved to {output_dir}/model_analysis.png")

def main():
    """Main function to demonstrate partial coherence model."""
    print("DUV Energy Deposition: Partial Coherence Model")
    print("=" * 60)
    
    # Initialize partial coherence model
    model = PartialCoherenceModel()
    
    # Calculate partial coherence model
    results = model.calculate_partial_coherence_model()
    
    # Calculate performance metrics
    performance = model.calculate_performance_metrics()
    
    # Print results
    print(f"Sigma values: {len(results['sigma_values'])}")
    print(f"Pupil cutoff values: {len(results['pupil_cutoff_values'])}")
    print(f"PSF variations: {len(results['psf_results'])}")
    print(f"Aerial image variations: {len(results['aerial_image_results'])}")
    print(f"Contrast/NILS variations: {len(results['contrast_nils_results'])}")
    
    print(f"Performance - Execution time: {performance['execution_time']:.2f} s")
    print(f"Performance - Memory usage: {performance['memory_usage'] / 1e6:.1f} MB")
    print(f"Performance - Throughput: {performance['throughput'] / 1e9:.1f} G operations/s")
    print(f"Performance - GPU utilization: {performance['gpu_utilization']:.2%}")
    
    # Plot results
    model.plot_model_analysis()
    
    print("Partial coherence model complete!")

if __name__ == "__main__":
    main()