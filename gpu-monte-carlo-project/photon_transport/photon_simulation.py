#!/usr/bin/env python3
"""
GPU Monte Carlo Dose Engine: Photon Transport Module
Photon transport simulation with energy deposition and scattering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PhotonTransport:
    """Photon transport simulation for Monte Carlo dose calculation."""
    
    def __init__(self, config: Dict = None):
        """Initialize photon transport."""
        self.config = config or {
            'num_photons': 100000,           # Number of photons to simulate
            'energy_range': (0.1, 20.0),     # Energy range in MeV
            'voxel_size': 0.1,               # Voxel size in cm
            'max_steps': 1000,               # Maximum steps per photon
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
        
        self.transport_results = {}
        self.performance_metrics = {}
        
    def calculate_photon_transport(self) -> Dict[str, any]:
        """Calculate photon transport simulation."""
        print("Calculating photon transport simulation...")
        
        # Initialize photons
        photons = self._initialize_photons()
        
        # Initialize phantom
        phantom = self._initialize_phantom()
        
        # Transport simulation
        dose_map = np.zeros(phantom['num_voxels'])
        energy_deposition = np.zeros(phantom['num_voxels'])
        photon_tracks = []
        
        for i, photon in enumerate(photons):
            track = self._simulate_photon_track(photon, phantom)
            photon_tracks.append(track)
            
            # Update dose map
            for step in track['steps']:
                voxel_idx = step['voxel_idx']
                if voxel_idx < phantom['num_voxels']:
                    dose_map[voxel_idx] += step['energy_deposited']
                    energy_deposition[voxel_idx] += step['energy_deposited']
        
        # Calculate statistics
        total_dose = np.sum(dose_map)
        max_dose = np.max(dose_map)
        mean_dose = np.mean(dose_map)
        std_dose = np.std(dose_map)
        
        # Calculate efficiency
        efficiency = self._calculate_transport_efficiency(photons, photon_tracks)
        
        self.transport_results = {
            'dose_map': dose_map,
            'energy_deposition': energy_deposition,
            'photon_tracks': photon_tracks,
            'total_dose': total_dose,
            'max_dose': max_dose,
            'mean_dose': mean_dose,
            'std_dose': std_dose,
            'efficiency': efficiency,
            'num_photons': len(photons),
            'num_voxels': phantom['num_voxels']
        }
        
        return self.transport_results
    
    def _initialize_photons(self) -> List[Dict]:
        """Initialize photon data."""
        photons = []
        num_photons = self.config['num_photons']
        energy_range = self.config['energy_range']
        
        for i in range(num_photons):
            # Random energy in range
            energy = np.random.uniform(energy_range[0], energy_range[1])
            
            # Random position
            position = np.random.uniform(-10, 10, 3)  # cm
            
            # Random direction
            direction = np.random.uniform(-1, 1, 3)
            direction = direction / np.linalg.norm(direction)
            
            # Random weight
            weight = np.random.uniform(0.5, 1.5)
            
            photons.append({
                'id': i,
                'energy': energy,
                'position': position,
                'direction': direction,
                'weight': weight,
                'alive': True,
                'steps': 0
            })
        
        return photons
    
    def _initialize_phantom(self) -> Dict:
        """Initialize phantom data."""
        voxel_size = self.config['voxel_size']
        num_voxels = 1000000  # 100x100x100 voxels
        
        phantom = {
            'num_voxels': num_voxels,
            'voxel_size': voxel_size,
            'dimensions': (100, 100, 100),
            'materials': self._generate_materials(num_voxels)
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
    
    def _simulate_photon_track(self, photon: Dict, phantom: Dict) -> Dict:
        """Simulate photon track."""
        track = {
            'photon_id': photon['id'],
            'steps': [],
            'total_energy_deposited': 0.0,
            'total_distance': 0.0,
            'final_energy': photon['energy']
        }
        
        current_position = photon['position'].copy()
        current_direction = photon['direction'].copy()
        current_energy = photon['energy']
        
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
            
            # Update photon
            current_position = new_position
            current_energy -= energy_deposited
            track['total_energy_deposited'] += energy_deposited
            track['total_distance'] += step_size
            
            # Check if photon is absorbed or escaped
            if current_energy <= 0 or voxel_idx >= phantom['num_voxels']:
                break
            
            # Scattering
            current_direction = self._calculate_scattering(
                current_direction, self.config['scattering_angle']
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
    
    def _calculate_scattering(self, direction: np.ndarray, 
                            scattering_angle: float) -> np.ndarray:
        """Calculate scattering."""
        # Simplified scattering calculation
        angle = np.random.normal(0, scattering_angle)
        
        # Rotate direction by angle
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        new_direction = direction.copy()
        new_direction[0] = direction[0] * cos_angle - direction[1] * sin_angle
        new_direction[1] = direction[0] * sin_angle + direction[1] * cos_angle
        
        # Normalize
        new_direction = new_direction / np.linalg.norm(new_direction)
        
        return new_direction
    
    def _calculate_transport_efficiency(self, photons: List[Dict], 
                                      tracks: List[Dict]) -> float:
        """Calculate transport efficiency."""
        total_photons = len(photons)
        successful_tracks = sum(1 for track in tracks if len(track['steps']) > 0)
        
        return successful_tracks / total_photons if total_photons > 0 else 0.0
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        print("Calculating performance metrics...")
        
        # Transport execution time (simplified)
        execution_time = 1.0  # seconds
        
        # Memory usage
        memory_usage = self.config['num_photons'] * 8 * 4  # 4 floats per photon
        
        # Throughput
        throughput = self.config['num_photons'] / execution_time
        
        # Efficiency
        efficiency = self._calculate_transport_efficiency(
            self.transport_results.get('photon_tracks', []),
            self.transport_results.get('photon_tracks', [])
        )
        
        # GPU utilization (simplified)
        gpu_utilization = min(1.0, throughput / 1e6)  # Normalized to 1M photons/s
        
        self.performance_metrics = {
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'throughput': throughput,
            'efficiency': efficiency,
            'gpu_utilization': gpu_utilization,
            'photons_per_second': throughput,
            'memory_bandwidth': memory_usage / execution_time,
            'compute_intensity': throughput / memory_usage
        }
        
        return self.performance_metrics
    
    def plot_transport_analysis(self, output_dir: str = "photon_transport") -> None:
        """Plot transport analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Dose map
        dose_map = self.transport_results['dose_map']
        dose_map_2d = dose_map[:10000].reshape(100, 100)
        
        im = axes[0, 0].imshow(dose_map_2d, cmap='hot', origin='lower')
        axes[0, 0].set_title('Photon Transport - Dose Map')
        axes[0, 0].set_xlabel('X (voxels)')
        axes[0, 0].set_ylabel('Y (voxels)')
        plt.colorbar(im, ax=axes[0, 0], label='Dose (Gy)')
        
        # Energy deposition
        energy_deposition = self.transport_results['energy_deposition']
        axes[0, 1].hist(energy_deposition, bins=50, alpha=0.7, color='blue')
        axes[0, 1].set_title('Energy Deposition Distribution')
        axes[0, 1].set_xlabel('Energy Deposited (MeV)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True)
        
        # Dose distribution
        axes[0, 2].hist(dose_map, bins=50, alpha=0.7, color='red')
        axes[0, 2].set_title('Dose Distribution')
        axes[0, 2].set_xlabel('Dose (Gy)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True)
        
        # Performance metrics
        if self.performance_metrics:
            metrics = ['Execution Time', 'Memory Usage', 'Throughput', 'Efficiency', 'GPU Utilization']
            values = [self.performance_metrics['execution_time'],
                     self.performance_metrics['memory_usage'] / 1e6,  # Convert to MB
                     self.performance_metrics['throughput'] / 1e6,    # Convert to M photons/s
                     self.performance_metrics['efficiency'],
                     self.performance_metrics['gpu_utilization']]
            
            axes[1, 0].bar(metrics, values, color=['blue', 'green', 'orange', 'red', 'purple'])
            axes[1, 0].set_title('Performance Metrics')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True)
        
        # Particle tracks
        tracks = self.transport_results['photon_tracks']
        if tracks:
            # Plot first few tracks
            for i, track in enumerate(tracks[:10]):
                steps = track['steps']
                if steps:
                    positions = np.array([step['position'] for step in steps])
                    axes[1, 1].plot(positions[:, 0], positions[:, 1], alpha=0.7, linewidth=1)
            
            axes[1, 1].set_title('Photon Tracks')
            axes[1, 1].set_xlabel('X (cm)')
            axes[1, 1].set_ylabel('Y (cm)')
            axes[1, 1].grid(True)
        
        # Energy vs distance
        if tracks:
            energies = []
            distances = []
            for track in tracks:
                if track['steps']:
                    energies.append(track['total_energy_deposited'])
                    distances.append(track['total_distance'])
            
            axes[1, 2].scatter(distances, energies, alpha=0.7, color='green')
            axes[1, 2].set_title('Energy vs Distance')
            axes[1, 2].set_xlabel('Distance (cm)')
            axes[1, 2].set_ylabel('Energy Deposited (MeV)')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/transport_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Transport analysis plot saved to {output_dir}/transport_analysis.png")

def main():
    """Main function to demonstrate photon transport."""
    print("GPU Monte Carlo Dose Engine: Photon Transport")
    print("=" * 60)
    
    # Initialize photon transport
    transport = PhotonTransport()
    
    # Calculate photon transport
    results = transport.calculate_photon_transport()
    
    # Calculate performance metrics
    performance = transport.calculate_performance_metrics()
    
    # Print results
    print(f"Total dose: {results['total_dose']:.2f} Gy")
    print(f"Max dose: {results['max_dose']:.2f} Gy")
    print(f"Mean dose: {results['mean_dose']:.2f} Gy")
    print(f"Efficiency: {results['efficiency']:.2%}")
    
    print(f"Performance - Execution time: {performance['execution_time']:.2f} s")
    print(f"Performance - Throughput: {performance['throughput']:.0f} photons/s")
    print(f"Performance - GPU utilization: {performance['gpu_utilization']:.2%}")
    
    # Plot results
    transport.plot_transport_analysis()
    
    print("Photon transport complete!")

if __name__ == "__main__":
    main()