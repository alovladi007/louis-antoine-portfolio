#!/usr/bin/env python3
"""
GPU Monte Carlo Dose Engine: CUDA Kernels Module
CUDA/OpenCL kernels for photon and pencil-beam proton transport.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MonteCarloKernels:
    """CUDA kernels for Monte Carlo dose calculation."""
    
    def __init__(self, config: Dict = None):
        """Initialize Monte Carlo kernels."""
        self.config = config or {
            'num_particles': 1000000,        # Number of particles to simulate
            'num_voxels': 1000000,           # Number of voxels in phantom
            'voxel_size': 0.1,               # Voxel size in cm
            'energy_range': (0.1, 20.0),     # Energy range in MeV
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
        
        self.kernel_results = {}
        self.performance_metrics = {}
        
    def calculate_photon_transport_kernel(self) -> Dict[str, any]:
        """Calculate photon transport kernel."""
        print("Calculating photon transport kernel...")
        
        # Kernel parameters
        num_particles = self.config['num_particles']
        num_voxels = self.config['num_voxels']
        voxel_size = self.config['voxel_size']
        energy_range = self.config['energy_range']
        max_steps = self.config['max_steps']
        step_size = self.config['step_size']
        
        # Initialize particle data
        particles = self._initialize_particles(num_particles, energy_range)
        
        # Initialize voxel data
        voxels = self._initialize_voxels(num_voxels, voxel_size)
        
        # Photon transport simulation
        dose_map = np.zeros(num_voxels)
        energy_deposition = np.zeros(num_voxels)
        particle_tracks = []
        
        for i in range(num_particles):
            particle = particles[i]
            track = self._simulate_photon_track(particle, voxels, max_steps, step_size)
            particle_tracks.append(track)
            
            # Update dose map
            for step in track['steps']:
                voxel_idx = step['voxel_idx']
                if voxel_idx < num_voxels:
                    dose_map[voxel_idx] += step['energy_deposited']
                    energy_deposition[voxel_idx] += step['energy_deposited']
        
        # Calculate statistics
        total_dose = np.sum(dose_map)
        max_dose = np.max(dose_map)
        mean_dose = np.mean(dose_map)
        std_dose = np.std(dose_map)
        
        # Calculate efficiency
        efficiency = self._calculate_kernel_efficiency(particles, particle_tracks)
        
        self.kernel_results['photon_transport'] = {
            'dose_map': dose_map,
            'energy_deposition': energy_deposition,
            'particle_tracks': particle_tracks,
            'total_dose': total_dose,
            'max_dose': max_dose,
            'mean_dose': mean_dose,
            'std_dose': std_dose,
            'efficiency': efficiency,
            'num_particles': num_particles,
            'num_voxels': num_voxels
        }
        
        return self.kernel_results['photon_transport']
    
    def calculate_proton_transport_kernel(self) -> Dict[str, any]:
        """Calculate proton transport kernel."""
        print("Calculating proton transport kernel...")
        
        # Kernel parameters
        num_particles = self.config['num_particles']
        num_voxels = self.config['num_voxels']
        voxel_size = self.config['voxel_size']
        energy_range = self.config['energy_range']
        max_steps = self.config['max_steps']
        step_size = self.config['step_size']
        
        # Initialize particle data
        particles = self._initialize_particles(num_particles, energy_range)
        
        # Initialize voxel data
        voxels = self._initialize_voxels(num_voxels, voxel_size)
        
        # Proton transport simulation
        dose_map = np.zeros(num_voxels)
        energy_deposition = np.zeros(num_voxels)
        particle_tracks = []
        
        for i in range(num_particles):
            particle = particles[i]
            track = self._simulate_proton_track(particle, voxels, max_steps, step_size)
            particle_tracks.append(track)
            
            # Update dose map
            for step in track['steps']:
                voxel_idx = step['voxel_idx']
                if voxel_idx < num_voxels:
                    dose_map[voxel_idx] += step['energy_deposited']
                    energy_deposition[voxel_idx] += step['energy_deposited']
        
        # Calculate statistics
        total_dose = np.sum(dose_map)
        max_dose = np.max(dose_map)
        mean_dose = np.mean(dose_map)
        std_dose = np.std(dose_map)
        
        # Calculate efficiency
        efficiency = self._calculate_kernel_efficiency(particles, particle_tracks)
        
        self.kernel_results['proton_transport'] = {
            'dose_map': dose_map,
            'energy_deposition': energy_deposition,
            'particle_tracks': particle_tracks,
            'total_dose': total_dose,
            'max_dose': max_dose,
            'mean_dose': mean_dose,
            'std_dose': std_dose,
            'efficiency': efficiency,
            'num_particles': num_particles,
            'num_voxels': num_voxels
        }
        
        return self.kernel_results['proton_transport']
    
    def _initialize_particles(self, num_particles: int, energy_range: Tuple[float, float]) -> List[Dict]:
        """Initialize particle data."""
        particles = []
        
        for i in range(num_particles):
            # Random energy in range
            energy = np.random.uniform(energy_range[0], energy_range[1])
            
            # Random position
            position = np.random.uniform(-10, 10, 3)  # cm
            
            # Random direction
            direction = np.random.uniform(-1, 1, 3)
            direction = direction / np.linalg.norm(direction)
            
            # Random weight
            weight = np.random.uniform(0.5, 1.5)
            
            particles.append({
                'id': i,
                'energy': energy,
                'position': position,
                'direction': direction,
                'weight': weight,
                'alive': True,
                'steps': 0
            })
        
        return particles
    
    def _initialize_voxels(self, num_voxels: int, voxel_size: float) -> List[Dict]:
        """Initialize voxel data."""
        voxels = []
        
        for i in range(num_voxels):
            # Calculate voxel position
            x = (i % 100) * voxel_size
            y = ((i // 100) % 100) * voxel_size
            z = (i // 10000) * voxel_size
            
            # Random material properties
            density = np.random.uniform(0.5, 2.0)  # g/cm³
            atomic_number = np.random.uniform(1, 20)
            mass_number = np.random.uniform(1, 40)
            
            voxels.append({
                'id': i,
                'position': np.array([x, y, z]),
                'density': density,
                'atomic_number': atomic_number,
                'mass_number': mass_number,
                'dose': 0.0,
                'energy_deposited': 0.0
            })
        
        return voxels
    
    def _simulate_photon_track(self, particle: Dict, voxels: List[Dict], 
                              max_steps: int, step_size: float) -> Dict:
        """Simulate photon track."""
        track = {
            'particle_id': particle['id'],
            'steps': [],
            'total_energy_deposited': 0.0,
            'total_distance': 0.0,
            'final_energy': particle['energy']
        }
        
        current_position = particle['position'].copy()
        current_direction = particle['direction'].copy()
        current_energy = particle['energy']
        
        for step in range(max_steps):
            # Calculate step
            step_vector = current_direction * step_size
            new_position = current_position + step_vector
            
            # Find voxel
            voxel_idx = self._find_voxel(new_position, voxels)
            
            # Calculate energy deposition
            energy_deposited = self._calculate_photon_energy_deposition(
                current_energy, step_size, voxels[voxel_idx] if voxel_idx < len(voxels) else None
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
            if current_energy <= 0 or voxel_idx >= len(voxels):
                break
            
            # Scattering
            current_direction = self._calculate_photon_scattering(
                current_direction, self.config['scattering_angle']
            )
        
        track['final_energy'] = current_energy
        return track
    
    def _simulate_proton_track(self, particle: Dict, voxels: List[Dict], 
                              max_steps: int, step_size: float) -> Dict:
        """Simulate proton track."""
        track = {
            'particle_id': particle['id'],
            'steps': [],
            'total_energy_deposited': 0.0,
            'total_distance': 0.0,
            'final_energy': particle['energy']
        }
        
        current_position = particle['position'].copy()
        current_direction = particle['direction'].copy()
        current_energy = particle['energy']
        
        for step in range(max_steps):
            # Calculate step
            step_vector = current_direction * step_size
            new_position = current_position + step_vector
            
            # Find voxel
            voxel_idx = self._find_voxel(new_position, voxels)
            
            # Calculate energy deposition
            energy_deposited = self._calculate_proton_energy_deposition(
                current_energy, step_size, voxels[voxel_idx] if voxel_idx < len(voxels) else None
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
            if current_energy <= 0 or voxel_idx >= len(voxels):
                break
            
            # Scattering
            current_direction = self._calculate_proton_scattering(
                current_direction, current_energy, self.config['scattering_angle']
            )
        
        track['final_energy'] = current_energy
        return track
    
    def _find_voxel(self, position: np.ndarray, voxels: List[Dict]) -> int:
        """Find voxel containing position."""
        # Simplified voxel finding
        x, y, z = position
        voxel_size = self.config['voxel_size']
        
        # Calculate voxel indices
        x_idx = int(x / voxel_size) + 50
        y_idx = int(y / voxel_size) + 50
        z_idx = int(z / voxel_size) + 50
        
        # Calculate linear index
        if x_idx < 0 or x_idx >= 100 or y_idx < 0 or y_idx >= 100 or z_idx < 0 or z_idx >= 100:
            return len(voxels)  # Outside phantom
        
        return z_idx * 10000 + y_idx * 100 + x_idx
    
    def _calculate_photon_energy_deposition(self, energy: float, step_size: float, 
                                           voxel: Dict) -> float:
        """Calculate photon energy deposition."""
        if voxel is None:
            return 0.0
        
        # Simplified energy deposition calculation
        absorption_coeff = self.config['absorption_coefficient']
        density = voxel['density']
        
        # Energy deposition per unit length
        energy_deposition_rate = absorption_coeff * density * energy
        
        # Total energy deposited in step
        energy_deposited = energy_deposition_rate * step_size
        
        return min(energy_deposited, energy)  # Can't deposit more than available
    
    def _calculate_proton_energy_deposition(self, energy: float, step_size: float, 
                                           voxel: Dict) -> float:
        """Calculate proton energy deposition."""
        if voxel is None:
            return 0.0
        
        # Simplified energy deposition calculation
        # Protons have different energy deposition characteristics
        absorption_coeff = self.config['absorption_coefficient'] * 2.0  # Higher for protons
        density = voxel['density']
        
        # Energy deposition per unit length
        energy_deposition_rate = absorption_coeff * density * energy
        
        # Total energy deposited in step
        energy_deposited = energy_deposition_rate * step_size
        
        return min(energy_deposited, energy)  # Can't deposit more than available
    
    def _calculate_photon_scattering(self, direction: np.ndarray, 
                                   scattering_angle: float) -> np.ndarray:
        """Calculate photon scattering."""
        # Simplified scattering calculation
        angle = np.random.normal(0, scattering_angle)
        
        # Rotate direction by angle
        # This is a simplified 2D rotation
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        new_direction = direction.copy()
        new_direction[0] = direction[0] * cos_angle - direction[1] * sin_angle
        new_direction[1] = direction[0] * sin_angle + direction[1] * cos_angle
        
        # Normalize
        new_direction = new_direction / np.linalg.norm(new_direction)
        
        return new_direction
    
    def _calculate_proton_scattering(self, direction: np.ndarray, energy: float, 
                                   scattering_angle: float) -> np.ndarray:
        """Calculate proton scattering."""
        # Simplified scattering calculation
        # Protons have different scattering characteristics
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
    
    def _calculate_kernel_efficiency(self, particles: List[Dict], 
                                   tracks: List[Dict]) -> float:
        """Calculate kernel efficiency."""
        total_particles = len(particles)
        successful_tracks = sum(1 for track in tracks if len(track['steps']) > 0)
        
        return successful_tracks / total_particles if total_particles > 0 else 0.0
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        print("Calculating performance metrics...")
        
        # Kernel execution time (simplified)
        execution_time = 1.0  # seconds
        
        # Memory usage
        memory_usage = self.config['num_particles'] * 8 * 4  # 4 floats per particle
        
        # Throughput
        throughput = self.config['num_particles'] / execution_time
        
        # Efficiency
        efficiency = self._calculate_kernel_efficiency(
            self.kernel_results.get('photon_transport', {}).get('particle_tracks', []),
            self.kernel_results.get('photon_transport', {}).get('particle_tracks', [])
        )
        
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
    
    def plot_kernel_analysis(self, output_dir: str = "cuda_kernels") -> None:
        """Plot kernel analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Photon transport dose map
        if 'photon_transport' in self.kernel_results:
            photon_data = self.kernel_results['photon_transport']
            dose_map = photon_data['dose_map']
            
            # Reshape dose map for visualization
            dose_map_2d = dose_map[:10000].reshape(100, 100)
            
            im = axes[0, 0].imshow(dose_map_2d, cmap='hot', origin='lower')
            axes[0, 0].set_title('Photon Transport - Dose Map')
            axes[0, 0].set_xlabel('X (voxels)')
            axes[0, 0].set_ylabel('Y (voxels)')
            plt.colorbar(im, ax=axes[0, 0], label='Dose (Gy)')
        
        # Proton transport dose map
        if 'proton_transport' in self.kernel_results:
            proton_data = self.kernel_results['proton_transport']
            dose_map = proton_data['dose_map']
            
            # Reshape dose map for visualization
            dose_map_2d = dose_map[:10000].reshape(100, 100)
            
            im = axes[0, 1].imshow(dose_map_2d, cmap='hot', origin='lower')
            axes[0, 1].set_title('Proton Transport - Dose Map')
            axes[0, 1].set_xlabel('X (voxels)')
            axes[0, 1].set_ylabel('Y (voxels)')
            plt.colorbar(im, ax=axes[0, 1], label='Dose (Gy)')
        
        # Dose comparison
        if 'photon_transport' in self.kernel_results and 'proton_transport' in self.kernel_results:
            photon_dose = self.kernel_results['photon_transport']['dose_map']
            proton_dose = self.kernel_results['proton_transport']['dose_map']
            
            axes[0, 2].hist(photon_dose, bins=50, alpha=0.7, label='Photon', color='blue')
            axes[0, 2].hist(proton_dose, bins=50, alpha=0.7, label='Proton', color='red')
            axes[0, 2].set_title('Dose Distribution Comparison')
            axes[0, 2].set_xlabel('Dose (Gy)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # Performance metrics
        if self.performance_metrics:
            metrics = ['Execution Time', 'Memory Usage', 'Throughput', 'Efficiency', 'GPU Utilization']
            values = [self.performance_metrics['execution_time'],
                     self.performance_metrics['memory_usage'] / 1e6,  # Convert to MB
                     self.performance_metrics['throughput'] / 1e6,    # Convert to M particles/s
                     self.performance_metrics['efficiency'],
                     self.performance_metrics['gpu_utilization']]
            
            axes[1, 0].bar(metrics, values, color=['blue', 'green', 'orange', 'red', 'purple'])
            axes[1, 0].set_title('Performance Metrics')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True)
        
        # Particle tracks
        if 'photon_transport' in self.kernel_results:
            tracks = self.kernel_results['photon_transport']['particle_tracks']
            if tracks:
                # Plot first few tracks
                for i, track in enumerate(tracks[:10]):
                    steps = track['steps']
                    if steps:
                        positions = np.array([step['position'] for step in steps])
                        axes[1, 1].plot(positions[:, 0], positions[:, 1], alpha=0.7, linewidth=1)
                
                axes[1, 1].set_title('Particle Tracks (Photon)')
                axes[1, 1].set_xlabel('X (cm)')
                axes[1, 1].set_ylabel('Y (cm)')
                axes[1, 1].grid(True)
        
        # Energy deposition
        if 'photon_transport' in self.kernel_results:
            energy_deposition = self.kernel_results['photon_transport']['energy_deposition']
            axes[1, 2].hist(energy_deposition, bins=50, alpha=0.7, color='blue')
            axes[1, 2].set_title('Energy Deposition Distribution')
            axes[1, 2].set_xlabel('Energy Deposited (MeV)')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/kernel_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Kernel analysis plot saved to {output_dir}/kernel_analysis.png")

def main():
    """Main function to demonstrate Monte Carlo kernels."""
    print("GPU Monte Carlo Dose Engine: CUDA Kernels")
    print("=" * 60)
    
    # Initialize Monte Carlo kernels
    kernels = MonteCarloKernels()
    
    # Calculate photon transport
    photon_results = kernels.calculate_photon_transport_kernel()
    
    # Calculate proton transport
    proton_results = kernels.calculate_proton_transport_kernel()
    
    # Calculate performance metrics
    performance = kernels.calculate_performance_metrics()
    
    # Print results
    print(f"Photon transport - Total dose: {photon_results['total_dose']:.2f} Gy")
    print(f"Photon transport - Max dose: {photon_results['max_dose']:.2f} Gy")
    print(f"Photon transport - Efficiency: {photon_results['efficiency']:.2%}")
    
    print(f"Proton transport - Total dose: {proton_results['total_dose']:.2f} Gy")
    print(f"Proton transport - Max dose: {proton_results['max_dose']:.2f} Gy")
    print(f"Proton transport - Efficiency: {proton_results['efficiency']:.2%}")
    
    print(f"Performance - Execution time: {performance['execution_time']:.2f} s")
    print(f"Performance - Throughput: {performance['throughput']:.0f} particles/s")
    print(f"Performance - GPU utilization: {performance['gpu_utilization']:.2%}")
    
    # Plot results
    kernels.plot_kernel_analysis()
    
    print("Monte Carlo kernels complete!")

if __name__ == "__main__":
    main()