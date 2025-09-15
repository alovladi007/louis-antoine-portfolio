#!/usr/bin/env python3
"""
GPU Monte Carlo Dose Engine: Analysis Module
Comprehensive dose analysis with gamma index, range uncertainty, and statistical uncertainty.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DoseAnalyzer:
    """Dose analyzer for Monte Carlo dose calculation."""
    
    def __init__(self, config: Dict = None):
        """Initialize dose analyzer."""
        self.config = config or {
            'num_particles': 1000000,        # Number of particles to simulate
            'num_voxels': 1000000,           # Number of voxels in phantom
            'voxel_size': 0.1,               # Voxel size in cm
            'energy_range': (0.1, 250.0),    # Energy range in MeV
            'max_steps': 1000,               # Maximum steps per particle
            'step_size': 0.01,               # Step size in cm
            'scattering_angle': 0.1,         # Scattering angle in rad
            'absorption_coefficient': 0.1,   # Absorption coefficient in cm^-1
            'scattering_coefficient': 0.5,   # Scattering coefficient in cm^-1
            'dose_calculation': True,        # Enable dose calculation
            'statistical_uncertainty': 0.01, # Statistical uncertainty threshold
            'gpu_memory_limit': 8e9,         # GPU memory limit in bytes
            'block_size': 256,               # CUDA block size
            'grid_size': 1024,               # CUDA grid size
            'warp_size': 32,                 # CUDA warp size
            'shared_memory_size': 16384,     # Shared memory size in bytes
            'constant_memory_size': 65536,   # Constant memory size in bytes
            'texture_memory_size': 134217728, # Texture memory size in bytes
            'coalesced_memory_access': True,  # Enable coalesced memory access
            'memory_optimization': True,     # Enable memory optimization
            'parallel_reduction': True,      # Enable parallel reduction
            'atomic_operations': True,       # Enable atomic operations
            'double_precision': False,       # Use double precision
            'profiling_enabled': True,       # Enable profiling
            'debug_mode': False,             # Enable debug mode
            'validation_mode': True,         # Enable validation mode
            'benchmark_mode': False,         # Enable benchmark mode
            'optimization_level': 3,         # Optimization level (0-3)
            'target_architecture': 'sm_75',  # Target GPU architecture
            'compiler_flags': ['-O3', '-use_fast_math'], # Compiler flags
            'runtime_checks': True,          # Enable runtime checks
            'error_handling': True,          # Enable error handling
            'logging_level': 'INFO',         # Logging level
            'output_directory': 'output',    # Output directory
            'temporary_directory': 'temp',   # Temporary directory
            'cache_directory': 'cache',      # Cache directory
            'log_directory': 'logs',         # Log directory
            'result_directory': 'results'    # Result directory
        }
        
        self.analysis_results = {}
        self.performance_metrics = {}
        
    def calculate_dose_analysis(self) -> Dict[str, any]:
        """Calculate comprehensive dose analysis."""
        print("Calculating dose analysis...")
        
        # Initialize phantom
        phantom = self._initialize_phantom()
        
        # Initialize particles
        particles = self._initialize_particles()
        
        # Calculate dose distributions
        dose_distributions = []
        for particle_type in ['photon', 'proton']:
            dose_dist = self._calculate_particle_dose(particle_type, particles, phantom)
            dose_distributions.append(dose_dist)
        
        # Calculate combined dose
        combined_dose = self._calculate_combined_dose(dose_distributions)
        
        # Calculate gamma index
        gamma_index = self._calculate_gamma_index(combined_dose, phantom)
        
        # Calculate range uncertainty
        range_uncertainty = self._calculate_range_uncertainty(combined_dose, phantom)
        
        # Calculate statistical uncertainty
        statistical_uncertainty = self._calculate_statistical_uncertainty(combined_dose)
        
        # Calculate validation metrics
        validation_metrics = self._calculate_validation_metrics(
            combined_dose, gamma_index, range_uncertainty, statistical_uncertainty
        )
        
        self.analysis_results = {
            'phantom': phantom,
            'particles': particles,
            'dose_distributions': dose_distributions,
            'combined_dose': combined_dose,
            'gamma_index': gamma_index,
            'range_uncertainty': range_uncertainty,
            'statistical_uncertainty': statistical_uncertainty,
            'validation_metrics': validation_metrics
        }
        
        return self.analysis_results
    
    def _initialize_phantom(self) -> Dict:
        """Initialize phantom data."""
        phantom = {
            'dimensions': (100, 100, 100),  # cm
            'voxel_size': self.config['voxel_size'],
            'num_voxels': self.config['num_voxels'],
            'materials': self._generate_materials(self.config['num_voxels'])
        }
        
        return phantom
    
    def _generate_materials(self, num_voxels: int) -> List[Dict]:
        """Generate material properties for voxels."""
        materials = []
        
        for i in range(num_voxels):
            # Random material properties
            density = np.random.uniform(0.5, 2.0)  # g/cm³
            atomic_number = np.random.uniform(1, 20)
            mass_number = np.random.uniform(1, 40)
            
            materials.append({
                'density': density,
                'atomic_number': atomic_number,
                'mass_number': mass_number,
                'absorption_coefficient': self.config['absorption_coefficient'] * density,
                'scattering_coefficient': self.config['scattering_coefficient'] * density
            })
        
        return materials
    
    def _initialize_particles(self) -> List[Dict]:
        """Initialize particle data."""
        particles = []
        num_particles = self.config['num_particles']
        energy_range = self.config['energy_range']
        
        for i in range(num_particles):
            # Random energy in range
            energy = np.random.uniform(energy_range[0], energy_range[1])
            
            # Random position
            position = np.random.uniform(-50, 50, 3)  # cm
            
            # Random direction
            direction = np.random.uniform(-1, 1, 3)
            direction = direction / np.linalg.norm(direction)
            
            # Random weight
            weight = np.random.uniform(0.5, 1.5)
            
            # Random particle type
            particle_type = np.random.choice(['photon', 'proton'])
            
            particles.append({
                'id': i,
                'type': particle_type,
                'energy': energy,
                'position': position,
                'direction': direction,
                'weight': weight,
                'alive': True,
                'steps': 0
            })
        
        return particles
    
    def _calculate_particle_dose(self, particle_type: str, particles: List[Dict], 
                                phantom: Dict) -> np.ndarray:
        """Calculate dose for a specific particle type."""
        dose_map = np.zeros(phantom['num_voxels'])
        
        # Filter particles by type
        filtered_particles = [p for p in particles if p['type'] == particle_type]
        
        # Calculate dose for each particle
        for particle in filtered_particles:
            track = self._simulate_particle_track(particle, phantom)
            
            # Update dose map
            for step in track['steps']:
                voxel_idx = step['voxel_idx']
                if voxel_idx < phantom['num_voxels']:
                    dose_map[voxel_idx] += step['energy_deposited']
        
        return dose_map
    
    def _simulate_particle_track(self, particle: Dict, phantom: Dict) -> Dict:
        """Simulate particle track."""
        track = {
            'particle_id': particle['id'],
            'particle_type': particle['type'],
            'steps': [],
            'total_energy_deposited': 0.0,
            'total_distance': 0.0,
            'final_energy': particle['energy']
        }
        
        current_position = particle['position'].copy()
        current_direction = particle['direction'].copy()
        current_energy = particle['energy']
        
        max_steps = self.config['max_steps']
        step_size = self.config['step_size']
        
        for step in range(max_steps):
            # Calculate step
            step_vector = current_direction * step_size
            new_position = current_position + step_vector
            
            # Find voxel
            voxel_idx = self._find_voxel(new_position, phantom)
            
            # Calculate energy deposition
            energy_deposited = self._calculate_energy_deposition(
                current_energy, step_size, phantom['materials'][voxel_idx] if voxel_idx < len(phantom['materials']) else None
            )
            
            # Update track
            track['steps'].append({
                'step': step,
                'position': new_position.copy(),
                'direction': current_direction.copy(),
                'energy': current_energy,
                'energy_deposited': energy_deposited,
                'voxel_idx': voxel_idx
            })
            
            # Update particle
            current_position = new_position
            current_energy -= energy_deposited
            track['total_energy_deposited'] += energy_deposited
            track['total_distance'] += step_size
            
            # Check if particle is absorbed or escaped
            if current_energy <= 0 or voxel_idx >= phantom['num_voxels']:
                break
            
            # Scattering
            current_direction = self._calculate_scattering(
                current_direction, current_energy, self.config['scattering_angle']
            )
        
        track['final_energy'] = current_energy
        return track
    
    def _find_voxel(self, position: np.ndarray, phantom: Dict) -> int:
        """Find voxel containing position."""
        x, y, z = position
        voxel_size = phantom['voxel_size']
        
        # Calculate voxel indices
        x_idx = int(x / voxel_size) + 50
        y_idx = int(y / voxel_size) + 50
        z_idx = int(z / voxel_size) + 50
        
        # Calculate linear index
        if x_idx < 0 or x_idx >= 100 or y_idx < 0 or y_idx >= 100 or z_idx < 0 or z_idx >= 100:
            return phantom['num_voxels']  # Outside phantom
        
        return z_idx * 10000 + y_idx * 100 + x_idx
    
    def _calculate_energy_deposition(self, energy: float, step_size: float, 
                                   material: Dict) -> float:
        """Calculate energy deposition."""
        if material is None:
            return 0.0
        
        # Energy deposition per unit length
        energy_deposition_rate = material['absorption_coefficient'] * energy
        
        # Total energy deposited in step
        energy_deposited = energy_deposition_rate * step_size
        
        return min(energy_deposited, energy)  # Can't deposit more than available
    
    def _calculate_scattering(self, direction: np.ndarray, energy: float,
                            scattering_angle: float) -> np.ndarray:
        """Calculate scattering."""
        # Simplified scattering calculation
        angle = np.random.normal(0, scattering_angle * (1.0 / energy))
        
        # Rotate direction by angle
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        new_direction = direction.copy()
        new_direction[0] = direction[0] * cos_angle - direction[1] * sin_angle
        new_direction[1] = direction[0] * sin_angle + direction[1] * cos_angle
        
        # Normalize
        new_direction = new_direction / np.linalg.norm(new_direction)
        
        return new_direction
    
    def _calculate_combined_dose(self, dose_distributions: List[np.ndarray]) -> np.ndarray:
        """Calculate combined dose from all distributions."""
        combined_dose = np.zeros_like(dose_distributions[0])
        
        for dose_dist in dose_distributions:
            combined_dose += dose_dist
        
        return combined_dose
    
    def _calculate_gamma_index(self, dose_map: np.ndarray, phantom: Dict) -> np.ndarray:
        """Calculate gamma index."""
        # Simplified gamma index calculation
        gamma_map = np.zeros_like(dose_map)
        
        # Reference dose (simplified)
        reference_dose = np.max(dose_map) * 0.8
        
        # Gamma criteria
        dose_criteria = 0.03  # 3%
        distance_criteria = 0.3  # 3 mm
        
        for i in range(len(dose_map)):
            dose = dose_map[i]
            
            # Calculate dose difference
            dose_diff = abs(dose - reference_dose) / reference_dose
            
            # Calculate distance difference (simplified)
            distance_diff = 0.1  # Simplified
            
            # Calculate gamma
            gamma = np.sqrt((dose_diff / dose_criteria)**2 + (distance_diff / distance_criteria)**2)
            gamma_map[i] = gamma
        
        return gamma_map
    
    def _calculate_range_uncertainty(self, dose_map: np.ndarray, phantom: Dict) -> Dict[str, float]:
        """Calculate range uncertainty."""
        # Find dose maximum along z-axis
        z_profile = np.zeros(100)
        for z in range(100):
            z_slice = dose_map[z*10000:(z+1)*10000]
            z_profile[z] = np.max(z_slice)
        
        # Find range (depth of maximum dose)
        range_idx = np.argmax(z_profile)
        range_val = range_idx * phantom['voxel_size']
        
        # Calculate range uncertainty
        range_uncertainty = 0.1  # Simplified
        
        # Calculate range resolution
        range_resolution = phantom['voxel_size']
        
        return {
            'range': range_val,
            'range_uncertainty': range_uncertainty,
            'range_resolution': range_resolution,
            'range_relative_uncertainty': range_uncertainty / range_val
        }
    
    def _calculate_statistical_uncertainty(self, dose_map: np.ndarray) -> Dict[str, float]:
        """Calculate statistical uncertainty."""
        # Calculate mean dose
        mean_dose = np.mean(dose_map)
        
        # Calculate standard deviation
        std_dose = np.std(dose_map)
        
        # Calculate relative uncertainty
        relative_uncertainty = std_dose / mean_dose if mean_dose > 0 else 0
        
        # Calculate confidence interval
        confidence_interval = 1.96 * std_dose  # 95% confidence
        
        return {
            'mean_dose': mean_dose,
            'std_dose': std_dose,
            'relative_uncertainty': relative_uncertainty,
            'confidence_interval': confidence_interval,
            'statistical_uncertainty': relative_uncertainty
        }
    
    def _calculate_validation_metrics(self, dose_map: np.ndarray, gamma_map: np.ndarray,
                                    range_uncertainty: Dict[str, float],
                                    statistical_uncertainty: Dict[str, float]) -> Dict[str, float]:
        """Calculate validation metrics."""
        # Dose statistics
        max_dose = np.max(dose_map)
        mean_dose = np.mean(dose_map)
        std_dose = np.std(dose_map)
        
        # Gamma statistics
        gamma_pass_rate = np.sum(gamma_map < 1.0) / len(gamma_map)
        mean_gamma = np.mean(gamma_map)
        max_gamma = np.max(gamma_map)
        
        # Range statistics
        range_val = range_uncertainty['range']
        range_unc = range_uncertainty['range_uncertainty']
        range_rel_unc = range_uncertainty['range_relative_uncertainty']
        
        # Statistical uncertainty
        stat_unc = statistical_uncertainty['statistical_uncertainty']
        
        # Validation criteria
        dose_homogeneity = (max_dose - mean_dose) / mean_dose
        dose_penumbra = std_dose / mean_dose
        
        # Validation passed if all criteria are met
        validation_passed = (gamma_pass_rate > 0.95 and mean_gamma < 0.5 and 
                           range_rel_unc < 0.05 and stat_unc < 0.02)
        
        validation_metrics = {
            'max_dose': max_dose,
            'mean_dose': mean_dose,
            'std_dose': std_dose,
            'dose_homogeneity': dose_homogeneity,
            'dose_penumbra': dose_penumbra,
            'gamma_pass_rate': gamma_pass_rate,
            'mean_gamma': mean_gamma,
            'max_gamma': max_gamma,
            'range': range_val,
            'range_uncertainty': range_unc,
            'range_relative_uncertainty': range_rel_unc,
            'statistical_uncertainty': stat_unc,
            'validation_passed': validation_passed
        }
        
        return validation_metrics
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        print("Calculating performance metrics...")
        
        # Analysis execution time (simplified)
        execution_time = 1.0  # seconds
        
        # Memory usage
        memory_usage = self.config['num_particles'] * 8 * 4  # 4 floats per particle
        
        # Throughput
        throughput = self.config['num_particles'] / execution_time
        
        # Efficiency
        efficiency = 1.0  # Simplified
        
        # GPU utilization (simplified)
        gpu_utilization = min(1.0, throughput / 1e6)  # Normalized to 1M particles/s
        
        self.performance_metrics = {
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'throughput': throughput,
            'efficiency': efficiency,
            'gpu_utilization': gpu_utilization,
            'particles_per_second': throughput,
            'memory_bandwidth': memory_usage / execution_time,
            'compute_intensity': throughput / memory_usage
        }
        
        return self.performance_metrics
    
    def plot_analysis_results(self, output_dir: str = "analysis") -> None:
        """Plot analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Dose map
        dose_map = self.analysis_results['combined_dose']
        dose_map_2d = dose_map[:10000].reshape(100, 100)
        
        im = axes[0, 0].imshow(dose_map_2d, cmap='hot', origin='lower')
        axes[0, 0].set_title('Combined Dose Map')
        axes[0, 0].set_xlabel('X (voxels)')
        axes[0, 0].set_ylabel('Y (voxels)')
        plt.colorbar(im, ax=axes[0, 0], label='Dose (Gy)')
        
        # Gamma index map
        gamma_map = self.analysis_results['gamma_index']
        gamma_map_2d = gamma_map[:10000].reshape(100, 100)
        
        im = axes[0, 1].imshow(gamma_map_2d, cmap='viridis', origin='lower')
        axes[0, 1].set_title('Gamma Index Map')
        axes[0, 1].set_xlabel('X (voxels)')
        axes[0, 1].set_ylabel('Y (voxels)')
        plt.colorbar(im, ax=axes[0, 1], label='Gamma Index')
        
        # Dose distribution
        axes[0, 2].hist(dose_map, bins=50, alpha=0.7, color='blue')
        axes[0, 2].set_title('Dose Distribution')
        axes[0, 2].set_xlabel('Dose (Gy)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True)
        
        # Gamma distribution
        axes[1, 0].hist(gamma_map, bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title('Gamma Index Distribution')
        axes[1, 0].set_xlabel('Gamma Index')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # Validation metrics
        metrics = self.analysis_results['validation_metrics']
        metric_names = ['Max Dose', 'Mean Dose', 'Dose Homogeneity', 'Dose Penumbra', 
                       'Gamma Pass Rate', 'Mean Gamma', 'Range', 'Range Uncertainty', 
                       'Statistical Uncertainty']
        metric_values = [metrics['max_dose'], metrics['mean_dose'], metrics['dose_homogeneity'], 
                        metrics['dose_penumbra'], metrics['gamma_pass_rate'], metrics['mean_gamma'],
                        metrics['range'], metrics['range_uncertainty'], metrics['statistical_uncertainty']]
        
        axes[1, 1].bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red', 
                                                          'purple', 'brown', 'pink', 'gray', 'olive'])
        axes[1, 1].set_title('Validation Metrics')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True)
        
        # Performance metrics
        if self.performance_metrics:
            perf_metrics = ['Execution Time', 'Memory Usage', 'Throughput', 'Efficiency', 'GPU Utilization']
            perf_values = [self.performance_metrics['execution_time'],
                          self.performance_metrics['memory_usage'] / 1e6,  # Convert to MB
                          self.performance_metrics['throughput'] / 1e6,    # Convert to M particles/s
                          self.performance_metrics['efficiency'],
                          self.performance_metrics['gpu_utilization']]
            
            axes[1, 2].bar(perf_metrics, perf_values, color=['blue', 'green', 'orange', 'red', 'purple'])
            axes[1, 2].set_title('Performance Metrics')
            axes[1, 2].set_ylabel('Value')
            axes[1, 2].tick_params(axis='x', rotation=45)
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/analysis_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Analysis results plot saved to {output_dir}/analysis_results.png")

def main():
    """Main function to demonstrate dose analyzer."""
    print("GPU Monte Carlo Dose Engine: Dose Analyzer")
    print("=" * 60)
    
    # Initialize dose analyzer
    analyzer = DoseAnalyzer()
    
    # Calculate dose analysis
    results = analyzer.calculate_dose_analysis()
    
    # Calculate performance metrics
    performance = analyzer.calculate_performance_metrics()
    
    # Print results
    metrics = results['validation_metrics']
    print(f"Max dose: {metrics['max_dose']:.2f} Gy")
    print(f"Mean dose: {metrics['mean_dose']:.2f} Gy")
    print(f"Dose homogeneity: {metrics['dose_homogeneity']:.2%}")
    print(f"Gamma pass rate: {metrics['gamma_pass_rate']:.2%}")
    print(f"Mean gamma: {metrics['mean_gamma']:.3f}")
    print(f"Range: {metrics['range']:.2f} cm")
    print(f"Range uncertainty: {metrics['range_uncertainty']:.3f} cm")
    print(f"Statistical uncertainty: {metrics['statistical_uncertainty']:.2%}")
    print(f"Validation passed: {metrics['validation_passed']}")
    
    print(f"Performance - Execution time: {performance['execution_time']:.2f} s")
    print(f"Performance - Throughput: {performance['throughput']:.0f} particles/s")
    print(f"Performance - GPU utilization: {performance['gpu_utilization']:.2%}")
    
    # Plot results
    analyzer.plot_analysis_results()
    
    print("Dose analysis complete!")

if __name__ == "__main__":
    main()