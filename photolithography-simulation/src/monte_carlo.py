"""
Monte Carlo Simulation Module

This module implements Monte Carlo methods for simulating stochastic
processes in photolithography including resist development, line edge
roughness, and process variations.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import ndimage, stats
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


@dataclass
class ProcessParameters:
    """Parameters for Monte Carlo process simulation"""
    temperature: float = 293.15  # K
    exposure_dose: float = 30.0  # mJ/cm²
    development_time: float = 60.0  # seconds
    resist_thickness: float = 100.0  # nm
    acid_diffusion_length: float = 10.0  # nm
    quencher_concentration: float = 0.1  # mol/L
    dissolution_rate_max: float = 100.0  # nm/min
    dissolution_rate_min: float = 0.1  # nm/min
    protection_ratio: float = 0.7  # Initial protection ratio
    
    # Stochastic parameters
    shot_noise: bool = True
    acid_diffusion: bool = True
    development_noise: bool = True
    
    # Monte Carlo parameters
    n_photons: int = 10000
    n_acid_molecules: int = 5000
    grid_size: Tuple[int, int] = (256, 256)
    pixel_size: float = 2.0  # nm


class MonteCarloSimulator:
    """
    Monte Carlo simulator for photolithography processes
    """
    
    def __init__(self, params: Optional[ProcessParameters] = None):
        """
        Initialize Monte Carlo simulator
        
        Args:
            params: Process parameters
        """
        self.params = params or ProcessParameters()
        self.rng = np.random.RandomState(42)
        
    def simulate_exposure(self, aerial_image: np.ndarray,
                         include_shot_noise: bool = True) -> np.ndarray:
        """
        Simulate exposure with photon shot noise
        
        Args:
            aerial_image: Normalized aerial image intensity
            include_shot_noise: Whether to include shot noise
            
        Returns:
            Exposed resist with shot noise
        """
        if not include_shot_noise:
            return aerial_image * self.params.exposure_dose
        
        # Calculate expected number of photons per pixel
        photon_density = aerial_image * self.params.n_photons
        
        # Apply Poisson statistics for shot noise
        photon_counts = self.rng.poisson(photon_density)
        
        # Normalize back to dose
        exposed_resist = photon_counts / self.params.n_photons * self.params.exposure_dose
        
        return exposed_resist
    
    def simulate_acid_generation(self, exposed_resist: np.ndarray) -> np.ndarray:
        """
        Simulate photo-acid generation
        
        Args:
            exposed_resist: Exposed resist pattern
            
        Returns:
            Acid concentration distribution
        """
        # Quantum efficiency (simplified model)
        quantum_efficiency = 0.3
        
        # Generate acid based on exposure
        acid_concentration = exposed_resist * quantum_efficiency
        
        # Add stochastic variation
        noise = self.rng.normal(0, 0.05, acid_concentration.shape)
        acid_concentration = np.maximum(0, acid_concentration + noise)
        
        return acid_concentration
    
    def simulate_acid_diffusion(self, acid_concentration: np.ndarray,
                               diffusion_steps: int = 100) -> np.ndarray:
        """
        Simulate acid diffusion during post-exposure bake (PEB)
        
        Args:
            acid_concentration: Initial acid distribution
            diffusion_steps: Number of diffusion steps
            
        Returns:
            Diffused acid concentration
        """
        # Diffusion coefficient (simplified)
        D = self.params.acid_diffusion_length**2 / (2 * self.params.development_time)
        dt = 0.1  # Time step
        dx = self.params.pixel_size
        
        # Stability criterion
        alpha = D * dt / dx**2
        if alpha > 0.25:
            dt = 0.25 * dx**2 / D
            alpha = 0.25
        
        acid = acid_concentration.copy()
        
        for _ in range(diffusion_steps):
            # Apply diffusion equation with random walk
            laplacian = ndimage.laplace(acid)
            acid += alpha * laplacian
            
            # Add random walk component
            if self.params.acid_diffusion:
                random_walk = self.rng.normal(0, np.sqrt(2 * D * dt) / dx, acid.shape)
                acid += random_walk * acid
            
            # Account for quencher
            quenching = self.params.quencher_concentration * acid * dt
            acid = np.maximum(0, acid - quenching)
        
        return acid
    
    def simulate_deprotection(self, acid_concentration: np.ndarray) -> np.ndarray:
        """
        Simulate polymer deprotection reaction
        
        Args:
            acid_concentration: Acid concentration after PEB
            
        Returns:
            Deprotection level (0 = fully protected, 1 = fully deprotected)
        """
        # Arrhenius kinetics
        k_0 = 1e13  # Pre-exponential factor (1/s)
        E_a = 50000  # Activation energy (J/mol)
        R = 8.314  # Gas constant (J/mol/K)
        
        # Rate constant
        k = k_0 * np.exp(-E_a / (R * self.params.temperature))
        
        # Deprotection reaction (simplified)
        reaction_probability = 1 - np.exp(-k * acid_concentration * self.params.development_time)
        
        # Stochastic deprotection
        if self.params.development_noise:
            threshold = self.rng.random(acid_concentration.shape)
            deprotection = (reaction_probability > threshold).astype(float)
        else:
            deprotection = reaction_probability
        
        return deprotection
    
    def simulate_development(self, deprotection: np.ndarray) -> np.ndarray:
        """
        Simulate resist development
        
        Args:
            deprotection: Deprotection level
            
        Returns:
            Developed resist profile
        """
        # Dissolution rate based on deprotection
        R_max = self.params.dissolution_rate_max
        R_min = self.params.dissolution_rate_min
        
        dissolution_rate = R_min + (R_max - R_min) * deprotection
        
        # Development time
        dissolved_thickness = dissolution_rate * self.params.development_time / 60  # Convert to nm
        
        # Remaining resist
        remaining_resist = np.maximum(0, self.params.resist_thickness - dissolved_thickness)
        
        # Binary pattern (simplified)
        threshold = self.params.resist_thickness * 0.5
        pattern = (remaining_resist > threshold).astype(float) * 255
        
        return pattern
    
    def simulate_full_process(self, mask: np.ndarray,
                            aerial_image: np.ndarray,
                            n_simulations: int = 10) -> Dict[str, Any]:
        """
        Run full Monte Carlo simulation of lithography process
        
        Args:
            mask: Input mask pattern
            aerial_image: Aerial image from optical simulation
            n_simulations: Number of Monte Carlo runs
            
        Returns:
            Dictionary with simulation results
        """
        results = {
            'patterns': [],
            'line_edge_roughness': [],
            'critical_dimensions': [],
            'pattern_fidelity': []
        }
        
        for i in tqdm(range(n_simulations), desc="Monte Carlo simulations"):
            # Set random seed for reproducibility
            self.rng = np.random.RandomState(42 + i)
            
            # Exposure with shot noise
            exposed = self.simulate_exposure(aerial_image, self.params.shot_noise)
            
            # Acid generation
            acid = self.simulate_acid_generation(exposed)
            
            # Acid diffusion
            if self.params.acid_diffusion:
                acid = self.simulate_acid_diffusion(acid)
            
            # Deprotection
            deprotection = self.simulate_deprotection(acid)
            
            # Development
            pattern = self.simulate_development(deprotection)
            
            # Store results
            results['patterns'].append(pattern)
            
            # Calculate metrics
            ler = self.calculate_line_edge_roughness(pattern)
            cd = self.measure_critical_dimension(pattern)
            fidelity = self.calculate_pattern_fidelity(mask, pattern)
            
            results['line_edge_roughness'].append(ler)
            results['critical_dimensions'].append(cd)
            results['pattern_fidelity'].append(fidelity)
        
        # Calculate statistics
        results['statistics'] = self.calculate_statistics(results)
        
        return results
    
    def calculate_line_edge_roughness(self, pattern: np.ndarray) -> float:
        """
        Calculate line edge roughness (LER)
        
        Args:
            pattern: Developed resist pattern
            
        Returns:
            LER value in nm
        """
        # Find edges
        edges = ndimage.sobel(pattern)
        edge_points = np.where(edges > 0)
        
        if len(edge_points[0]) < 10:
            return 0.0
        
        # Find vertical edges (simplified)
        vertical_edges = []
        for x in range(pattern.shape[1]):
            edge_y = np.where(edges[:, x] > 0)[0]
            if len(edge_y) > 0:
                vertical_edges.append(edge_y[0])
        
        if len(vertical_edges) < 10:
            return 0.0
        
        # Calculate standard deviation
        ler = np.std(vertical_edges) * self.params.pixel_size
        
        return ler
    
    def measure_critical_dimension(self, pattern: np.ndarray) -> float:
        """
        Measure critical dimension (CD)
        
        Args:
            pattern: Developed resist pattern
            
        Returns:
            CD in nm
        """
        # Find horizontal line (simplified)
        middle_row = pattern.shape[0] // 2
        line = pattern[middle_row, :]
        
        # Find edges
        edges = np.where(np.diff(line) != 0)[0]
        
        if len(edges) >= 2:
            cd = (edges[1] - edges[0]) * self.params.pixel_size
            return cd
        
        return 0.0
    
    def calculate_pattern_fidelity(self, target: np.ndarray, 
                                  actual: np.ndarray) -> float:
        """
        Calculate pattern fidelity
        
        Args:
            target: Target pattern
            actual: Actual pattern
            
        Returns:
            Fidelity score (0-1)
        """
        # Normalize patterns
        target_norm = (target > 0).astype(float)
        actual_norm = (actual > 0).astype(float)
        
        # Calculate overlap
        intersection = np.sum(target_norm * actual_norm)
        union = np.sum(np.maximum(target_norm, actual_norm))
        
        if union > 0:
            fidelity = intersection / union
        else:
            fidelity = 0.0
        
        return fidelity
    
    def calculate_statistics(self, results: Dict[str, List]) -> Dict[str, Any]:
        """
        Calculate statistics from Monte Carlo results
        
        Args:
            results: Monte Carlo simulation results
            
        Returns:
            Statistical summary
        """
        stats_dict = {}
        
        for metric in ['line_edge_roughness', 'critical_dimensions', 'pattern_fidelity']:
            values = np.array(results[metric])
            stats_dict[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return stats_dict
    
    def simulate_process_window(self, mask: np.ndarray,
                              focus_range: np.ndarray,
                              dose_range: np.ndarray,
                              n_simulations: int = 5) -> np.ndarray:
        """
        Simulate process window using Monte Carlo
        
        Args:
            mask: Input mask pattern
            focus_range: Array of focus values
            dose_range: Array of dose values
            n_simulations: Number of MC runs per condition
            
        Returns:
            CD values for each focus/dose combination
        """
        cd_matrix = np.zeros((len(focus_range), len(dose_range)))
        
        for i, focus in enumerate(focus_range):
            for j, dose in enumerate(dose_range):
                # Update parameters
                self.params.exposure_dose = dose
                
                # Simulate aerial image (simplified - would need optical model)
                aerial_image = self.simulate_defocused_image(mask, focus)
                
                # Run MC simulations
                cds = []
                for _ in range(n_simulations):
                    exposed = self.simulate_exposure(aerial_image)
                    acid = self.simulate_acid_generation(exposed)
                    acid = self.simulate_acid_diffusion(acid)
                    deprotection = self.simulate_deprotection(acid)
                    pattern = self.simulate_development(deprotection)
                    cd = self.measure_critical_dimension(pattern)
                    cds.append(cd)
                
                cd_matrix[i, j] = np.mean(cds)
        
        return cd_matrix
    
    def simulate_defocused_image(self, mask: np.ndarray, defocus: float) -> np.ndarray:
        """
        Simulate defocused aerial image (simplified)
        
        Args:
            mask: Input mask
            defocus: Defocus amount in nm
            
        Returns:
            Defocused aerial image
        """
        # Apply Gaussian blur to simulate defocus
        sigma = abs(defocus) / (2 * self.params.pixel_size)
        blurred = ndimage.gaussian_filter(mask.astype(float), sigma)
        
        # Normalize
        if np.max(blurred) > 0:
            blurred = blurred / np.max(blurred)
        
        return blurred
    
    def parallel_simulate(self, mask: np.ndarray, aerial_image: np.ndarray,
                         n_simulations: int = 100, n_processes: int = None) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations in parallel
        
        Args:
            mask: Input mask pattern
            aerial_image: Aerial image
            n_simulations: Number of simulations
            n_processes: Number of parallel processes
            
        Returns:
            Combined results
        """
        if n_processes is None:
            n_processes = mp.cpu_count()
        
        # Create worker function
        def worker(seed, mask, aerial_image, params):
            simulator = MonteCarloSimulator(params)
            simulator.rng = np.random.RandomState(seed)
            
            exposed = simulator.simulate_exposure(aerial_image)
            acid = simulator.simulate_acid_generation(exposed)
            acid = simulator.simulate_acid_diffusion(acid)
            deprotection = simulator.simulate_deprotection(acid)
            pattern = simulator.simulate_development(deprotection)
            
            ler = simulator.calculate_line_edge_roughness(pattern)
            cd = simulator.measure_critical_dimension(pattern)
            fidelity = simulator.calculate_pattern_fidelity(mask, pattern)
            
            return pattern, ler, cd, fidelity
        
        # Run parallel simulations
        with mp.Pool(n_processes) as pool:
            worker_partial = partial(worker, 
                                    mask=mask,
                                    aerial_image=aerial_image,
                                    params=self.params)
            
            results = pool.map(worker_partial, range(n_simulations))
        
        # Unpack results
        patterns = [r[0] for r in results]
        lers = [r[1] for r in results]
        cds = [r[2] for r in results]
        fidelities = [r[3] for r in results]
        
        return {
            'patterns': patterns,
            'line_edge_roughness': lers,
            'critical_dimensions': cds,
            'pattern_fidelity': fidelities,
            'statistics': {
                'line_edge_roughness': {
                    'mean': np.mean(lers),
                    'std': np.std(lers)
                },
                'critical_dimensions': {
                    'mean': np.mean(cds),
                    'std': np.std(cds)
                },
                'pattern_fidelity': {
                    'mean': np.mean(fidelities),
                    'std': np.std(fidelities)
                }
            }
        }
    
    def visualize_results(self, results: Dict[str, Any]) -> None:
        """
        Visualize Monte Carlo simulation results
        
        Args:
            results: Simulation results dictionary
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Show sample patterns
        n_samples = min(3, len(results['patterns']))
        for i in range(n_samples):
            axes[0, i].imshow(results['patterns'][i], cmap='gray')
            axes[0, i].set_title(f'MC Run {i+1}')
            axes[0, i].axis('off')
        
        # LER distribution
        axes[1, 0].hist(results['line_edge_roughness'], bins=20, edgecolor='black')
        axes[1, 0].set_xlabel('Line Edge Roughness (nm)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('LER Distribution')
        
        # CD distribution
        axes[1, 1].hist(results['critical_dimensions'], bins=20, edgecolor='black')
        axes[1, 1].set_xlabel('Critical Dimension (nm)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('CD Distribution')
        
        # Fidelity distribution
        axes[1, 2].hist(results['pattern_fidelity'], bins=20, edgecolor='black')
        axes[1, 2].set_xlabel('Pattern Fidelity')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Fidelity Distribution')
        
        plt.suptitle('Monte Carlo Simulation Results')
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create simulator
    params = ProcessParameters(
        exposure_dose=30.0,
        acid_diffusion_length=10.0,
        shot_noise=True,
        acid_diffusion=True,
        development_noise=True
    )
    
    simulator = MonteCarloSimulator(params)
    
    # Create test pattern
    mask = np.zeros((256, 256))
    mask[100:150, 50:200] = 255  # Horizontal line
    mask[50:200, 120:140] = 255  # Vertical line
    
    # Create aerial image (simplified)
    aerial_image = ndimage.gaussian_filter(mask / 255, sigma=2)
    
    print("Running Monte Carlo simulations...")
    results = simulator.simulate_full_process(mask, aerial_image, n_simulations=20)
    
    # Print statistics
    print("\nSimulation Statistics:")
    for metric, stats in results['statistics'].items():
        print(f"\n{metric}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.3f}")
    
    # Visualize results
    simulator.visualize_results(results)