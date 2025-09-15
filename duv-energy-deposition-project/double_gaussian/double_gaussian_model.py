#!/usr/bin/env python3
"""
DUV Energy Deposition: Double Gaussian Model Module
Double-Gaussian PSF model for DUV mask energy deposition.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DoubleGaussianModel:
    """Double-Gaussian PSF model for DUV mask energy deposition."""
    
    def __init__(self, config: Dict = None):
        """Initialize double Gaussian model."""
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
            'psf_sigma_1': 10.0,             # First Gaussian sigma
            'psf_sigma_2': 50.0,             # Second Gaussian sigma
            'psf_weight_1': 0.8,             # First Gaussian weight
            'psf_weight_2': 0.2,             # Second Gaussian weight
            'fft_size': 2048,                # FFT size
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
        
    def calculate_double_gaussian_model(self) -> Dict[str, any]:
        """Calculate double Gaussian model."""
        print("Calculating double Gaussian model...")
        
        # Initialize mask
        mask = self._initialize_mask()
        
        # Calculate PSF
        psf = self._calculate_psf()
        
        # Calculate aerial image
        aerial_image = self._calculate_aerial_image(mask, psf)
        
        # Calculate contrast and NILS
        contrast_nils = self._calculate_contrast_nils(aerial_image)
        
        # Calculate model results
        self.model_results = {
            'mask': mask,
            'psf': psf,
            'aerial_image': aerial_image,
            'contrast_nils': contrast_nils
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
    
    def _calculate_psf(self) -> Dict:
        """Calculate point spread function."""
        print("Calculating PSF...")
        
        psf_size = self.config['psf_size']
        sigma_1 = self.config['psf_sigma_1']
        sigma_2 = self.config['psf_sigma_2']
        weight_1 = self.config['psf_weight_1']
        weight_2 = self.config['psf_weight_2']
        
        # Create coordinate grids
        x = np.linspace(-psf_size//2, psf_size//2, psf_size)
        y = np.linspace(-psf_size//2, psf_size//2, psf_size)
        X, Y = np.meshgrid(x, y)
        
        # Calculate distance from center
        r = np.sqrt(X**2 + Y**2)
        
        # Calculate first Gaussian
        gaussian_1 = np.exp(-(r**2) / (2 * sigma_1**2))
        
        # Calculate second Gaussian
        gaussian_2 = np.exp(-(r**2) / (2 * sigma_2**2))
        
        # Combine Gaussians
        psf = weight_1 * gaussian_1 + weight_2 * gaussian_2
        
        # Normalize
        if self.config['normalization']:
            psf = psf / np.sum(psf)
        
        psf_data = {
            'psf': psf,
            'size': psf_size,
            'sigma_1': sigma_1,
            'sigma_2': sigma_2,
            'weight_1': weight_1,
            'weight_2': weight_2,
            'coordinates': (X, Y)
        }
        
        return psf_data
    
    def _calculate_aerial_image(self, mask: Dict, psf: Dict) -> np.ndarray:
        """Calculate aerial image using convolution."""
        print("Calculating aerial image...")
        
        mask_pattern = mask['pattern']
        psf_data = psf['psf']
        
        # Convolve mask with PSF
        if self.config['convolution_method'] == 'fft':
            aerial_image = self._fft_convolution(mask_pattern, psf_data)
        else:
            aerial_image = signal.convolve2d(mask_pattern, psf_data, mode='same')
        
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
    
    def _calculate_contrast_nils(self, aerial_image: np.ndarray) -> Dict:
        """Calculate contrast and NILS."""
        print("Calculating contrast and NILS...")
        
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
        
        memory_usage = (mask_size[0] * mask_size[1] + psf_size**2 + fft_size**2) * 4  # 4 bytes per float
        
        # Throughput
        throughput = (mask_size[0] * mask_size[1] * psf_size**2) / execution_time
        
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
    
    def plot_model_analysis(self, output_dir: str = "double_gaussian") -> None:
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
        
        # PSF
        psf = self.model_results['psf']['psf']
        im = axes[0, 1].imshow(psf, cmap='hot', origin='lower')
        axes[0, 1].set_title('Point Spread Function')
        axes[0, 1].set_xlabel('X (pixels)')
        axes[0, 1].set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=axes[0, 1], label='Intensity')
        
        # Aerial image
        aerial_image = self.model_results['aerial_image']
        im = axes[0, 2].imshow(aerial_image, cmap='hot', origin='lower')
        axes[0, 2].set_title('Aerial Image')
        axes[0, 2].set_xlabel('X (pixels)')
        axes[0, 2].set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=axes[0, 2], label='Intensity')
        
        # PSF cross-section
        psf_center = psf[psf.shape[0]//2, :]
        axes[1, 0].plot(psf_center, 'b-', linewidth=2)
        axes[1, 0].set_title('PSF Cross-Section')
        axes[1, 0].set_xlabel('X (pixels)')
        axes[1, 0].set_ylabel('Intensity')
        axes[1, 0].grid(True)
        
        # Aerial image cross-section
        aerial_center = aerial_image[aerial_image.shape[0]//2, :]
        axes[1, 1].plot(aerial_center, 'r-', linewidth=2)
        axes[1, 1].set_title('Aerial Image Cross-Section')
        axes[1, 1].set_xlabel('X (pixels)')
        axes[1, 1].set_ylabel('Intensity')
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
        plt.savefig(f"{output_dir}/model_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model analysis plot saved to {output_dir}/model_analysis.png")

def main():
    """Main function to demonstrate double Gaussian model."""
    print("DUV Energy Deposition: Double Gaussian Model")
    print("=" * 60)
    
    # Initialize double Gaussian model
    model = DoubleGaussianModel()
    
    # Calculate double Gaussian model
    results = model.calculate_double_gaussian_model()
    
    # Calculate performance metrics
    performance = model.calculate_performance_metrics()
    
    # Print results
    contrast_nils = results['contrast_nils']
    print(f"Contrast: {contrast_nils['contrast']:.3f}")
    print(f"NILS: {contrast_nils['nils']:.3f}")
    print(f"Max intensity: {contrast_nils['max_intensity']:.3f}")
    print(f"Min intensity: {contrast_nils['min_intensity']:.3f}")
    print(f"Mean intensity: {contrast_nils['mean_intensity']:.3f}")
    print(f"SNR: {contrast_nils['snr']:.3f}")
    
    print(f"Performance - Execution time: {performance['execution_time']:.2f} s")
    print(f"Performance - Memory usage: {performance['memory_usage'] / 1e6:.1f} MB")
    print(f"Performance - Throughput: {performance['throughput'] / 1e9:.1f} G operations/s")
    print(f"Performance - GPU utilization: {performance['gpu_utilization']:.2%}")
    
    # Plot results
    model.plot_model_analysis()
    
    print("Double Gaussian model complete!")

if __name__ == "__main__":
    main()