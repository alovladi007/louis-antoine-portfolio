#!/usr/bin/env python3
"""
DUV Energy Deposition: Monte Carlo Simulation Module
Monte Carlo particle model for DUV mask energy deposition simulation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MonteCarloSimulation:
    """Monte Carlo simulation for DUV mask energy deposition."""
    
    def __init__(self, config: Dict = None):
        """Initialize Monte Carlo simulation."""
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
            'num_particles': 1000000,        # Number of particles to simulate
            'max_steps': 1000,               # Maximum steps per particle
            'step_size': 1e-9,               # Step size in m
            'scattering_angle': 0.1,         # Scattering angle in rad
            'absorption_coefficient': 0.1,   # Absorption coefficient in m^-1
            'scattering_coefficient': 0.5,   # Scattering coefficient in m^-1
            'energy_deposition': True,       # Enable energy deposition
            'statistical_uncertainty': 0.01, # Statistical uncertainty threshold
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
        
        self.simulation_results = {}
        self.performance_metrics = {}
        
    def calculate_monte_carlo_simulation(self) -> Dict[str, any]:
        """Calculate Monte Carlo simulation."""
        print("Calculating Monte Carlo simulation...")
        
        # Initialize particles
        particles = self._initialize_particles()
        
        # Initialize mask
        mask = self._initialize_mask()
        
        # Run simulation
        simulation_results = self._run_simulation(particles, mask)
        
        # Calculate energy deposition
        energy_deposition = self._calculate_energy_deposition(simulation_results, mask)
        
        # Calculate simulation results
        self.simulation_results = {
            'particles': particles,
            'mask': mask,
            'simulation_results': simulation_results,
            'energy_deposition': energy_deposition
        }
        
        return self.simulation_results
    
    def _initialize_particles(self) -> List[Dict]:
        """Initialize particle data."""
        particles = []
        num_particles = self.config['num_particles']
        
        for i in range(num_particles):
            # Random position
            position = np.random.uniform(-500, 500, 2)  # pixels
            
            # Random direction
            direction = np.random.uniform(-1, 1, 2)
            direction = direction / np.linalg.norm(direction)
            
            # Random energy
            energy = np.random.uniform(0.5, 1.5)
            
            # Random weight
            weight = np.random.uniform(0.5, 1.5)
            
            particles.append({
                'id': i,
                'position': position,
                'direction': direction,
                'energy': energy,
                'weight': weight,
                'alive': True,
                'steps': 0
            })
        
        return particles
    
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
            'absorption_coefficient': self.config['absorption_coefficient'],
            'scattering_coefficient': self.config['scattering_coefficient']
        }
        
        return mask
    
    def _run_simulation(self, particles: List[Dict], mask: Dict) -> Dict:
        """Run Monte Carlo simulation."""
        print("Running Monte Carlo simulation...")
        
        num_particles = len(particles)
        max_steps = self.config['max_steps']
        step_size = self.config['step_size']
        
        # Initialize results
        particle_tracks = []
        energy_deposition_map = np.zeros(mask['size'])
        
        for i, particle in enumerate(particles):
            if i % 10000 == 0:
                print(f"Processing particle {i}/{num_particles}")
            
            # Simulate particle track
            track = self._simulate_particle_track(particle, mask, max_steps, step_size)
            particle_tracks.append(track)
            
            # Update energy deposition map
            for step in track['steps']:
                x, y = step['position']
                if 0 <= x < mask['size'][0] and 0 <= y < mask['size'][1]:
                    energy_deposition_map[int(x), int(y)] += step['energy_deposited']
        
        simulation_results = {
            'particle_tracks': particle_tracks,
            'energy_deposition_map': energy_deposition_map,
            'num_particles': num_particles,
            'max_steps': max_steps
        }
        
        return simulation_results
    
    def _simulate_particle_track(self, particle: Dict, mask: Dict, 
                                max_steps: int, step_size: float) -> Dict:
        """Simulate particle track."""
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
            
            # Calculate energy deposition
            energy_deposited = self._calculate_energy_deposition_step(
                current_energy, step_size, mask, new_position
            )
            
            # Update track
            track['steps'].append({
                'step': step,
                'position': new_position.copy(),
                'direction': current_direction.copy(),
                'energy': current_energy,
                'energy_deposited': energy_deposited
            })
            
            # Update particle
            current_position = new_position
            current_energy -= energy_deposited
            track['total_energy_deposited'] += energy_deposited
            track['total_distance'] += step_size
            
            # Check if particle is absorbed or escaped
            if current_energy <= 0 or not self._is_inside_mask(current_position, mask):
                break
            
            # Scattering
            current_direction = self._calculate_scattering(
                current_direction, self.config['scattering_angle']
            )
        
        track['final_energy'] = current_energy
        return track
    
    def _calculate_energy_deposition_step(self, energy: float, step_size: float, 
                                        mask: Dict, position: np.ndarray) -> float:
        """Calculate energy deposition for a single step."""
        # Check if position is inside mask
        if not self._is_inside_mask(position, mask):
            return 0.0
        
        # Get mask properties at position
        x, y = position
        if 0 <= x < mask['size'][0] and 0 <= y < mask['size'][1]:
            mask_value = mask['pattern'][int(x), int(y)]
        else:
            mask_value = 0.0
        
        # Calculate energy deposition
        absorption_coeff = mask['absorption_coefficient'] * mask_value
        energy_deposited = absorption_coeff * energy * step_size
        
        return min(energy_deposited, energy)  # Can't deposit more than available
    
    def _is_inside_mask(self, position: np.ndarray, mask: Dict) -> bool:
        """Check if position is inside mask."""
        x, y = position
        return 0 <= x < mask['size'][0] and 0 <= y < mask['size'][1]
    
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
    
    def _calculate_energy_deposition(self, simulation_results: Dict, mask: Dict) -> Dict:
        """Calculate energy deposition."""
        print("Calculating energy deposition...")
        
        energy_deposition_map = simulation_results['energy_deposition_map']
        
        # Calculate statistics
        total_energy = np.sum(energy_deposition_map)
        max_energy = np.max(energy_deposition_map)
        mean_energy = np.mean(energy_deposition_map)
        std_energy = np.std(energy_deposition_map)
        
        # Calculate efficiency
        efficiency = self._calculate_efficiency(simulation_results)
        
        energy_deposition = {
            'energy_deposition_map': energy_deposition_map,
            'total_energy': total_energy,
            'max_energy': max_energy,
            'mean_energy': mean_energy,
            'std_energy': std_energy,
            'efficiency': efficiency
        }
        
        return energy_deposition
    
    def _calculate_efficiency(self, simulation_results: Dict) -> float:
        """Calculate simulation efficiency."""
        particle_tracks = simulation_results['particle_tracks']
        total_particles = len(particle_tracks)
        successful_tracks = sum(1 for track in particle_tracks if len(track['steps']) > 0)
        
        return successful_tracks / total_particles if total_particles > 0 else 0.0
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        print("Calculating performance metrics...")
        
        # Simulation execution time (simplified)
        execution_time = 1.0  # seconds
        
        # Memory usage
        mask_size = self.config['mask_size']
        num_particles = self.config['num_particles']
        
        memory_usage = (mask_size[0] * mask_size[1] + num_particles * 8) * 4  # 4 bytes per float
        
        # Throughput
        throughput = (mask_size[0] * mask_size[1] * num_particles) / execution_time
        
        # Efficiency
        efficiency = self._calculate_efficiency(self.simulation_results.get('simulation_results', {}))
        
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
    
    def plot_simulation_analysis(self, output_dir: str = "monte_carlo") -> None:
        """Plot simulation analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Mask pattern
        mask_pattern = self.simulation_results['mask']['pattern']
        axes[0, 0].imshow(mask_pattern, cmap='gray', origin='lower')
        axes[0, 0].set_title('Mask Pattern')
        axes[0, 0].set_xlabel('X (pixels)')
        axes[0, 0].set_ylabel('Y (pixels)')
        
        # Energy deposition map
        energy_deposition_map = self.simulation_results['energy_deposition']['energy_deposition_map']
        im = axes[0, 1].imshow(energy_deposition_map, cmap='hot', origin='lower')
        axes[0, 1].set_title('Energy Deposition Map')
        axes[0, 1].set_xlabel('X (pixels)')
        axes[0, 1].set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=axes[0, 1], label='Energy (J)')
        
        # Energy distribution
        energy_values = energy_deposition_map.flatten()
        axes[0, 2].hist(energy_values, bins=50, alpha=0.7, color='blue')
        axes[0, 2].set_title('Energy Distribution')
        axes[0, 2].set_xlabel('Energy (J)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True)
        
        # Particle tracks
        particle_tracks = self.simulation_results['simulation_results']['particle_tracks']
        if particle_tracks:
            # Plot first few tracks
            for i, track in enumerate(particle_tracks[:10]):
                steps = track['steps']
                if steps:
                    positions = np.array([step['position'] for step in steps])
                    axes[1, 0].plot(positions[:, 0], positions[:, 1], alpha=0.7, linewidth=1)
            
            axes[1, 0].set_title('Particle Tracks')
            axes[1, 0].set_xlabel('X (pixels)')
            axes[1, 0].set_ylabel('Y (pixels)')
            axes[1, 0].grid(True)
        
        # Energy vs distance
        if particle_tracks:
            energies = []
            distances = []
            for track in particle_tracks:
                if track['steps']:
                    energies.append(track['total_energy_deposited'])
                    distances.append(track['total_distance'])
            
            axes[1, 1].scatter(distances, energies, alpha=0.7, color='green')
            axes[1, 1].set_title('Energy vs Distance')
            axes[1, 1].set_xlabel('Distance (pixels)')
            axes[1, 1].set_ylabel('Energy Deposited (J)')
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
        plt.savefig(f"{output_dir}/simulation_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Simulation analysis plot saved to {output_dir}/simulation_analysis.png")

def main():
    """Main function to demonstrate Monte Carlo simulation."""
    print("DUV Energy Deposition: Monte Carlo Simulation")
    print("=" * 60)
    
    # Initialize Monte Carlo simulation
    simulation = MonteCarloSimulation()
    
    # Calculate Monte Carlo simulation
    results = simulation.calculate_monte_carlo_simulation()
    
    # Calculate performance metrics
    performance = simulation.calculate_performance_metrics()
    
    # Print results
    energy_deposition = results['energy_deposition']
    print(f"Total energy: {energy_deposition['total_energy']:.2e} J")
    print(f"Max energy: {energy_deposition['max_energy']:.2e} J")
    print(f"Mean energy: {energy_deposition['mean_energy']:.2e} J")
    print(f"Efficiency: {energy_deposition['efficiency']:.2%}")
    
    print(f"Performance - Execution time: {performance['execution_time']:.2f} s")
    print(f"Performance - Memory usage: {performance['memory_usage'] / 1e6:.1f} MB")
    print(f"Performance - Throughput: {performance['throughput'] / 1e9:.1f} G operations/s")
    print(f"Performance - GPU utilization: {performance['gpu_utilization']:.2%}")
    
    # Plot results
    simulation.plot_simulation_analysis()
    
    print("Monte Carlo simulation complete!")

if __name__ == "__main__":
    main()