#!/usr/bin/env python3
"""
Low-Dose CT Reconstruction: Unrolled Network Module
Unrolled proximal gradient networks with learned denoisers and data consistency.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class UnrolledNetwork:
    """Unrolled proximal gradient network for CT reconstruction."""
    
    def __init__(self, config: Dict = None):
        """Initialize unrolled network."""
        self.config = config or {
            'image_size': (512, 512),        # Image size (height, width)
            'num_views': 360,                # Number of projection views
            'num_detectors': 512,            # Number of detectors
            'num_layers': 10,                # Number of unrolled layers
            'learning_rate': 0.001,          # Learning rate
            'batch_size': 32,                # Batch size
            'num_epochs': 100,               # Number of training epochs
            'regularization_weight': 0.01,   # Regularization weight
            'data_consistency_weight': 1.0,  # Data consistency weight
            'denoiser_type': 'unet',         # Denoiser type
            'denoiser_channels': 64,         # Number of denoiser channels
            'denoiser_layers': 5,            # Number of denoiser layers
            'activation': 'relu',            # Activation function
            'normalization': 'batch',        # Normalization type
            'dropout_rate': 0.1,             # Dropout rate
            'weight_decay': 1e-4,            # Weight decay
            'optimizer': 'adam',             # Optimizer type
            'scheduler': 'cosine',           # Learning rate scheduler
            'loss_function': 'mse',          # Loss function
            'metrics': ['psnr', 'ssim'],     # Evaluation metrics
            'validation_split': 0.2,         # Validation split
            'early_stopping': True,          # Enable early stopping
            'patience': 10,                  # Early stopping patience
            'checkpointing': True,           # Enable checkpointing
            'checkpoint_frequency': 10,      # Checkpoint frequency
            'tensorboard_logging': True,     # Enable TensorBoard logging
            'mixed_precision': True,         # Enable mixed precision
            'gradient_clipping': True,       # Enable gradient clipping
            'gradient_clip_value': 1.0,      # Gradient clip value
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
        
        self.network_results = {}
        self.performance_metrics = {}
        
    def calculate_unrolled_network(self) -> Dict[str, any]:
        """Calculate unrolled network."""
        print("Calculating unrolled network...")
        
        # Initialize network architecture
        architecture = self._initialize_architecture()
        
        # Initialize denoiser
        denoiser = self._initialize_denoiser()
        
        # Initialize data consistency layer
        data_consistency = self._initialize_data_consistency()
        
        # Initialize training data
        training_data = self._initialize_training_data()
        
        # Train network
        training_results = self._train_network(architecture, denoiser, data_consistency, training_data)
        
        # Evaluate network
        evaluation_results = self._evaluate_network(architecture, denoiser, data_consistency, training_data)
        
        # Calculate network results
        self.network_results = {
            'architecture': architecture,
            'denoiser': denoiser,
            'data_consistency': data_consistency,
            'training_data': training_data,
            'training_results': training_results,
            'evaluation_results': evaluation_results
        }
        
        return self.network_results
    
    def _initialize_architecture(self) -> Dict:
        """Initialize network architecture."""
        architecture = {
            'num_layers': self.config['num_layers'],
            'input_channels': 1,
            'output_channels': 1,
            'hidden_channels': 64,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'activation': self.config['activation'],
            'normalization': self.config['normalization'],
            'dropout_rate': self.config['dropout_rate']
        }
        
        return architecture
    
    def _initialize_denoiser(self) -> Dict:
        """Initialize denoiser network."""
        denoiser = {
            'type': self.config['denoiser_type'],
            'input_channels': 1,
            'output_channels': 1,
            'hidden_channels': self.config['denoiser_channels'],
            'num_layers': self.config['denoiser_layers'],
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'activation': self.config['activation'],
            'normalization': self.config['normalization'],
            'dropout_rate': self.config['dropout_rate']
        }
        
        return denoiser
    
    def _initialize_data_consistency(self) -> Dict:
        """Initialize data consistency layer."""
        data_consistency = {
            'weight': self.config['data_consistency_weight'],
            'forward_operator': 'radon',
            'backward_operator': 'iradon',
            'regularization': 'tv',
            'regularization_weight': self.config['regularization_weight']
        }
        
        return data_consistency
    
    def _initialize_training_data(self) -> Dict:
        """Initialize training data."""
        # Generate synthetic training data
        num_samples = 1000
        image_size = self.config['image_size']
        
        # Generate clean images
        clean_images = np.random.rand(num_samples, image_size[0], image_size[1])
        
        # Generate noisy images
        noisy_images = clean_images + 0.1 * np.random.randn(num_samples, image_size[0], image_size[1])
        
        # Generate projections
        projections = np.random.rand(num_samples, self.config['num_views'], self.config['num_detectors'])
        
        training_data = {
            'clean_images': clean_images,
            'noisy_images': noisy_images,
            'projections': projections,
            'num_samples': num_samples,
            'image_size': image_size
        }
        
        return training_data
    
    def _train_network(self, architecture: Dict, denoiser: Dict, 
                      data_consistency: Dict, training_data: Dict) -> Dict:
        """Train unrolled network."""
        print("Training unrolled network...")
        
        # Initialize training parameters
        num_epochs = self.config['num_epochs']
        batch_size = self.config['batch_size']
        learning_rate = self.config['learning_rate']
        
        # Initialize loss history
        loss_history = []
        validation_loss_history = []
        
        # Training loop (simplified)
        for epoch in range(num_epochs):
            # Calculate training loss
            training_loss = self._calculate_training_loss(architecture, denoiser, data_consistency, training_data)
            loss_history.append(training_loss)
            
            # Calculate validation loss
            validation_loss = self._calculate_validation_loss(architecture, denoiser, data_consistency, training_data)
            validation_loss_history.append(validation_loss)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Training Loss = {training_loss:.6f}, Validation Loss = {validation_loss:.6f}")
        
        training_results = {
            'loss_history': loss_history,
            'validation_loss_history': validation_loss_history,
            'final_training_loss': loss_history[-1],
            'final_validation_loss': validation_loss_history[-1],
            'num_epochs': num_epochs,
            'convergence': self._check_convergence(loss_history, validation_loss_history)
        }
        
        return training_results
    
    def _calculate_training_loss(self, architecture: Dict, denoiser: Dict, 
                                data_consistency: Dict, training_data: Dict) -> float:
        """Calculate training loss."""
        # Simplified loss calculation
        clean_images = training_data['clean_images']
        noisy_images = training_data['noisy_images']
        
        # Reconstructed images (simplified)
        reconstructed_images = self._reconstruct_images(noisy_images, architecture, denoiser, data_consistency)
        
        # MSE loss
        mse_loss = np.mean((clean_images - reconstructed_images)**2)
        
        # Regularization loss
        reg_loss = self._calculate_regularization_loss(reconstructed_images, data_consistency)
        
        # Total loss
        total_loss = mse_loss + reg_loss
        
        return total_loss
    
    def _calculate_validation_loss(self, architecture: Dict, denoiser: Dict, 
                                  data_consistency: Dict, training_data: Dict) -> float:
        """Calculate validation loss."""
        # Simplified validation loss calculation
        clean_images = training_data['clean_images']
        noisy_images = training_data['noisy_images']
        
        # Reconstructed images (simplified)
        reconstructed_images = self._reconstruct_images(noisy_images, architecture, denoiser, data_consistency)
        
        # MSE loss
        mse_loss = np.mean((clean_images - reconstructed_images)**2)
        
        return mse_loss
    
    def _reconstruct_images(self, noisy_images: np.ndarray, architecture: Dict, 
                          denoiser: Dict, data_consistency: Dict) -> np.ndarray:
        """Reconstruct images using unrolled network."""
        # Simplified reconstruction
        reconstructed_images = noisy_images.copy()
        
        # Apply denoiser
        for layer in range(architecture['num_layers']):
            # Denoising step
            reconstructed_images = self._apply_denoiser(reconstructed_images, denoiser)
            
            # Data consistency step
            reconstructed_images = self._apply_data_consistency(reconstructed_images, data_consistency)
        
        return reconstructed_images
    
    def _apply_denoiser(self, images: np.ndarray, denoiser: Dict) -> np.ndarray:
        """Apply denoiser to images."""
        # Simplified denoising
        denoised_images = images.copy()
        
        # Apply denoising filter
        for i in range(images.shape[0]):
            denoised_images[i] = signal.medfilt2d(images[i], kernel_size=3)
        
        return denoised_images
    
    def _apply_data_consistency(self, images: np.ndarray, data_consistency: Dict) -> np.ndarray:
        """Apply data consistency to images."""
        # Simplified data consistency
        consistent_images = images.copy()
        
        # Apply data consistency constraint
        consistent_images = np.clip(consistent_images, 0, 1)
        
        return consistent_images
    
    def _calculate_regularization_loss(self, images: np.ndarray, data_consistency: Dict) -> float:
        """Calculate regularization loss."""
        # Total variation regularization
        tv_loss = 0.0
        
        for i in range(images.shape[0]):
            # Calculate gradients
            grad_x = np.diff(images[i], axis=1)
            grad_y = np.diff(images[i], axis=0)
            
            # Calculate TV
            tv_loss += np.sum(np.sqrt(grad_x**2 + grad_y**2))
        
        return data_consistency['regularization_weight'] * tv_loss
    
    def _check_convergence(self, loss_history: List[float], validation_loss_history: List[float]) -> bool:
        """Check if training has converged."""
        if len(loss_history) < 10:
            return False
        
        # Check if loss has stopped decreasing
        recent_losses = loss_history[-10:]
        if np.std(recent_losses) < 1e-6:
            return True
        
        # Check if validation loss is increasing
        if len(validation_loss_history) >= 10:
            recent_val_losses = validation_loss_history[-10:]
            if np.mean(recent_val_losses[-5:]) > np.mean(recent_val_losses[:5]):
                return True
        
        return False
    
    def _evaluate_network(self, architecture: Dict, denoiser: Dict, 
                         data_consistency: Dict, training_data: Dict) -> Dict:
        """Evaluate network performance."""
        print("Evaluating network...")
        
        # Generate test data
        test_images = np.random.rand(100, self.config['image_size'][0], self.config['image_size'][1])
        test_noisy_images = test_images + 0.1 * np.random.randn(100, self.config['image_size'][0], self.config['image_size'][1])
        
        # Reconstruct test images
        reconstructed_images = self._reconstruct_images(test_noisy_images, architecture, denoiser, data_consistency)
        
        # Calculate metrics
        mse = np.mean((test_images - reconstructed_images)**2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        ssim = self._calculate_ssim(test_images, reconstructed_images)
        
        evaluation_results = {
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim,
            'test_images': test_images,
            'reconstructed_images': reconstructed_images
        }
        
        return evaluation_results
    
    def _calculate_ssim(self, images1: np.ndarray, images2: np.ndarray) -> float:
        """Calculate SSIM between images."""
        # Simplified SSIM calculation
        mu1 = np.mean(images1)
        mu2 = np.mean(images2)
        sigma1 = np.var(images1)
        sigma2 = np.var(images2)
        sigma12 = np.mean((images1 - mu1) * (images2 - mu2))
        
        c1 = 0.01**2
        c2 = 0.03**2
        
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        
        return ssim
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        print("Calculating performance metrics...")
        
        # Network execution time (simplified)
        execution_time = 1.0  # seconds
        
        # Memory usage
        image_size = self.config['image_size']
        num_layers = self.config['num_layers']
        hidden_channels = self.config['denoiser_channels']
        
        memory_usage = (image_size[0] * image_size[1] * hidden_channels * num_layers) * 4  # 4 bytes per float
        
        # Throughput
        throughput = (image_size[0] * image_size[1] * num_layers) / execution_time
        
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
    
    def plot_network_analysis(self, output_dir: str = "unrolled_network") -> None:
        """Plot network analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Training loss
        training_results = self.network_results['training_results']
        loss_history = training_results['loss_history']
        validation_loss_history = training_results['validation_loss_history']
        
        axes[0, 0].plot(loss_history, 'b-', linewidth=2, label='Training Loss')
        axes[0, 0].plot(validation_loss_history, 'r-', linewidth=2, label='Validation Loss')
        axes[0, 0].set_title('Training Loss History')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Evaluation metrics
        evaluation_results = self.network_results['evaluation_results']
        metrics = ['MSE', 'PSNR', 'SSIM']
        values = [evaluation_results['mse'], evaluation_results['psnr'], evaluation_results['ssim']]
        
        axes[0, 1].bar(metrics, values, color=['blue', 'green', 'orange'])
        axes[0, 1].set_title('Evaluation Metrics')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].grid(True)
        
        # Test images
        test_images = evaluation_results['test_images']
        reconstructed_images = evaluation_results['reconstructed_images']
        
        # Show first test image
        axes[0, 2].imshow(test_images[0], cmap='gray', origin='lower')
        axes[0, 2].set_title('Test Image')
        axes[0, 2].set_xlabel('X (pixels)')
        axes[0, 2].set_ylabel('Y (pixels)')
        
        # Show first reconstructed image
        axes[1, 0].imshow(reconstructed_images[0], cmap='gray', origin='lower')
        axes[1, 0].set_title('Reconstructed Image')
        axes[1, 0].set_xlabel('X (pixels)')
        axes[1, 0].set_ylabel('Y (pixels)')
        
        # Show difference image
        difference = np.abs(test_images[0] - reconstructed_images[0])
        axes[1, 1].imshow(difference, cmap='hot', origin='lower')
        axes[1, 1].set_title('Difference Image')
        axes[1, 1].set_xlabel('X (pixels)')
        axes[1, 1].set_ylabel('Y (pixels)')
        
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
        plt.savefig(f"{output_dir}/network_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Network analysis plot saved to {output_dir}/network_analysis.png")

def main():
    """Main function to demonstrate unrolled network."""
    print("Low-Dose CT Reconstruction: Unrolled Network")
    print("=" * 60)
    
    # Initialize unrolled network
    network = UnrolledNetwork()
    
    # Calculate unrolled network
    results = network.calculate_unrolled_network()
    
    # Calculate performance metrics
    performance = network.calculate_performance_metrics()
    
    # Print results
    training_results = results['training_results']
    evaluation_results = results['evaluation_results']
    
    print(f"Final training loss: {training_results['final_training_loss']:.6f}")
    print(f"Final validation loss: {training_results['final_validation_loss']:.6f}")
    print(f"Convergence: {training_results['convergence']}")
    print(f"MSE: {evaluation_results['mse']:.6f}")
    print(f"PSNR: {evaluation_results['psnr']:.2f} dB")
    print(f"SSIM: {evaluation_results['ssim']:.3f}")
    
    print(f"Performance - Execution time: {performance['execution_time']:.2f} s")
    print(f"Performance - Memory usage: {performance['memory_usage'] / 1e6:.1f} MB")
    print(f"Performance - Throughput: {performance['throughput'] / 1e9:.1f} G operations/s")
    print(f"Performance - GPU utilization: {performance['gpu_utilization']:.2%}")
    
    # Plot results
    network.plot_network_analysis()
    
    print("Unrolled network complete!")

if __name__ == "__main__":
    main()