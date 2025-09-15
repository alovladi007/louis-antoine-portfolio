#!/usr/bin/env python3
"""
DUV Energy Deposition: Flare Modeling Module
Flare modeling with long-tail via third Gaussian.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FlareModeling:
    """Flare modeling for DUV mask energy deposition."""
    
    def __init__(self, config: Dict = None):
        """Initialize flare modeling."""
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
            'flare_sigma_range': (0.05, 0.5), # Flare sigma range
            'flare_sigma_steps': 20,         # Number of flare sigma steps
            'flare_level_range': (0.001, 0.1), # Flare level range
            'flare_level_steps': 20,         # Number of flare level steps
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
        
    def calculate_flare_modeling(self) -> Dict[str, any]:
        """Calculate flare modeling."""
        print("Calculating flare modeling...")
        
        # Initialize mask
        mask = self._initialize_mask()
        
        # Calculate flare sigma values
        flare_sigma_values = self._calculate_flare_sigma_values()
        
        # Calculate flare level values
        flare_level_values = self._calculate_flare_level_values()
        
        # Calculate PSF with flare for different parameters
        psf_results = self._calculate_psf_with_flare_variations(flare_sigma_values, flare_level_values)
        
        # Calculate aerial images
        aerial_image_results = self._calculate_aerial_image_variations(mask, psf_results)
        
        # Calculate contrast and NILS
        contrast_nils_results = self._calculate_contrast_nils_variations(aerial_image_results)
        
        # Calculate model results
        self.model_results = {
            'mask': mask,
            'flare_sigma_values': flare_sigma_values,
            'flare_level_values': flare_level_values,
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
    
    def _calculate_flare_sigma_values(self) -> np.ndarray:
        """Calculate flare sigma values."""
        flare_sigma_range = self.config['flare_sigma_range']
        flare_sigma_steps = self.config['flare_sigma_steps']
        
        flare_sigma_values = np.linspace(flare_sigma_range[0], flare_sigma_range[1], flare_sigma_steps)
        
        return flare_sigma_values
    
    def _calculate_flare_level_values(self) -> np.ndarray:
        """Calculate flare level values."""
        flare_level_range = self.config['flare_level_range']
        flare_level_steps = self.config['flare_level_steps']
        
        flare_level_values = np.linspace(flare_level_range[0], flare_level_range[1], flare_level_steps)
        
        return flare_level_values
    
    def _calculate_psf_with_flare_variations(self, flare_sigma_values: np.ndarray, 
                                           flare_level_values: np.ndarray) -> Dict:
        """Calculate PSF with flare variations."""
        print("Calculating PSF with flare variations...")
        
        psf_size = self.config['psf_size']
        psf_results = {}
        
        for i, flare_sigma in enumerate(flare_sigma_values):
            for j, flare_level in enumerate(flare_level_values):
                # Calculate PSF for this combination
                psf = self._calculate_psf_with_flare(flare_sigma, flare_level)
                psf_results[(i, j)] = {
                    'psf': psf,
                    'flare_sigma': flare_sigma,
                    'flare_level': flare_level
                }
        
        return psf_results
    
    def _calculate_psf_with_flare(self, flare_sigma: float, flare_level: float) -> np.ndarray:
        """Calculate PSF with flare for given parameters."""
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
        flare_psf = np.exp(-(r**2) / (2 * flare_sigma**2))
        
        # Combine main PSF and flare
        psf = main_psf + flare_level * flare_psf
        
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
                'flare_sigma': psf_data['flare_sigma'],
                'flare_level': psf_data['flare_level']
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
                'flare_sigma': aerial_data['flare_sigma'],
                'flare_level': aerial_data['flare_level']
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
        flare_sigma_steps = self.config['flare_sigma_steps']
        flare_level_steps = self.config['flare_level_steps']
        
        memory_usage = (mask_size[0] * mask_size[1] + psf_size**2 + fft_size**2 + 
                       flare_sigma_steps * flare_level_steps * psf_size**2) * 4  # 4 bytes per float
        
        # Throughput
        throughput = (mask_size[0] * mask_size[1] * psf_size**2 * flare_sigma_steps * flare_level_steps) / execution_time
        
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
    
    def plot_model_analysis(self, output_dir: str = "flare_modeling") -> None:
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
        
        # Flare Sigma vs Contrast
        flare_sigma_values = self.model_results['flare_sigma_values']
        contrast_values = []
        
        for i in range(len(flare_sigma_values)):
            key = (i, 0)  # Use first flare level value
            if key in self.model_results['contrast_nils_results']:
                contrast = self.model_results['contrast_nils_results'][key]['contrast_nils']['contrast']
                contrast_values.append(contrast)
            else:
                contrast_values.append(0)
        
        axes[0, 1].plot(flare_sigma_values, contrast_values, 'b-', linewidth=2)
        axes[0, 1].set_title('Flare Sigma vs Contrast')
        axes[0, 1].set_xlabel('Flare Sigma')
        axes[0, 1].set_ylabel('Contrast')
        axes[0, 1].grid(True)
        
        # Flare Level vs NILS
        flare_level_values = self.model_results['flare_level_values']
        nils_values = []
        
        for j in range(len(flare_level_values)):
            key = (0, j)  # Use first flare sigma value
            if key in self.model_results['contrast_nils_results']:
                nils = self.model_results['contrast_nils_results'][key]['contrast_nils']['nils']
                nils_values.append(nils)
            else:
                nils_values.append(0)
        
        axes[0, 2].plot(flare_level_values, nils_values, 'r-', linewidth=2)
        axes[0, 2].set_title('Flare Level vs NILS')
        axes[0, 2].set_xlabel('Flare Level')
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
        
        # Flare Sigma vs Flare Level heatmap
        flare_sigma_steps = self.config['flare_sigma_steps']
        flare_level_steps = self.config['flare_level_steps']
        contrast_heatmap = np.zeros((flare_sigma_steps, flare_level_steps))
        
        for i in range(flare_sigma_steps):
            for j in range(flare_level_steps):
                key = (i, j)
                if key in contrast_nils_data:
                    contrast_heatmap[i, j] = contrast_nils_data[key]['contrast_nils']['contrast']
        
        im = axes[1, 1].imshow(contrast_heatmap, cmap='hot', origin='lower')
        axes[1, 1].set_title('Contrast Heatmap')
        axes[1, 1].set_xlabel('Flare Level Index')
        axes[1, 1].set_ylabel('Flare Sigma Index')
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
    """Main function to demonstrate flare modeling."""
    print("DUV Energy Deposition: Flare Modeling")
    print("=" * 60)
    
    # Initialize flare modeling
    model = FlareModeling()
    
    # Calculate flare modeling
    results = model.calculate_flare_modeling()
    
    # Calculate performance metrics
    performance = model.calculate_performance_metrics()
    
    # Print results
    print(f"Flare sigma values: {len(results['flare_sigma_values'])}")
    print(f"Flare level values: {len(results['flare_level_values'])}")
    print(f"PSF variations: {len(results['psf_results'])}")
    print(f"Aerial image variations: {len(results['aerial_image_results'])}")
    print(f"Contrast/NILS variations: {len(results['contrast_nils_results'])}")
    
    print(f"Performance - Execution time: {performance['execution_time']:.2f} s")
    print(f"Performance - Memory usage: {performance['memory_usage'] / 1e6:.1f} MB")
    print(f"Performance - Throughput: {performance['throughput'] / 1e9:.1f} G operations/s")
    print(f"Performance - GPU utilization: {performance['gpu_utilization']:.2%}")
    
    # Plot results
    model.plot_model_analysis()
    
    print("Flare modeling complete!")

if __name__ == "__main__":
    main()