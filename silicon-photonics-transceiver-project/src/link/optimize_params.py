"""Parameter optimization for minimum transmit power"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from typing import Dict, List, Tuple
import os

from sim_link_nrz_pam4 import SimulationConfig, TransceiverSimulator


class ParameterOptimizer:
    """Optimize transceiver parameters for minimum power consumption"""
    
    def __init__(self, target_ber: float = 1e-12):
        self.target_ber = target_ber
        self.results_dir = '../../results'
        os.makedirs(f'{self.results_dir}/figs', exist_ok=True)
        os.makedirs(f'{self.results_dir}/tables', exist_ok=True)
        
    def grid_search(self, param_ranges: Dict[str, List]) -> pd.DataFrame:
        """Perform grid search over parameter space
        
        Args:
            param_ranges: Dictionary of parameter ranges to search
            
        Returns:
            DataFrame with optimization results
        """
        results = []
        
        # Default ranges if not specified
        if not param_ranges:
            param_ranges = {
                'tx_power_dbm': [-5, -2, 0, 2, 5],
                'vpp': [1.5, 2.0, 2.5, 3.0],
                'vpi': [2.5, 3.0, 3.5, 4.0],
                'extinction_ratio_db': [6, 8, 10, 12]
            }
        
        # Generate all combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        total_combinations = np.prod([len(v) for v in param_values])
        print(f"Testing {total_combinations} parameter combinations...")
        
        for i, combination in enumerate(product(*param_values)):
            params = dict(zip(param_names, combination))
            
            # Run simulation with these parameters
            config = SimulationConfig(
                bitrate_gbps=25.0,
                modulation='NRZ',
                fiber_length_km=2.0,
                smoke_test=True  # Fast mode for optimization
            )
            
            # Apply parameter overrides
            if 'tx_power_dbm' in params:
                config.tx_power_dbm = params['tx_power_dbm']
            if 'vpp' in params:
                config.vpp = params['vpp']
            
            simulator = TransceiverSimulator(config)
            
            # Override component parameters
            if 'vpi' in params:
                simulator.mzm_config.vpi = params['vpi']
                simulator.mzm.vpi = params['vpi']
            if 'extinction_ratio_db' in params:
                simulator.mzm_config.extinction_ratio_db = params['extinction_ratio_db']
            
            # Run simulation
            sim_results = simulator.simulate_link()
            
            # Calculate power consumption (simplified model)
            power_consumption_mw = self._calculate_power(params, sim_results)
            
            # Store results
            result = params.copy()
            result['ber'] = sim_results['ber']
            result['eye_height'] = sim_results['eye_metrics']['eye_height']
            result['power_consumption_mw'] = power_consumption_mw
            result['meets_target'] = sim_results['ber'] <= self.target_ber
            
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{total_combinations}")
        
        df = pd.DataFrame(results)
        return df
    
    def _calculate_power(self, params: Dict, sim_results: Dict) -> float:
        """Calculate total power consumption
        
        Args:
            params: System parameters
            sim_results: Simulation results
            
        Returns:
            Power consumption in mW
        """
        # Simplified power model
        tx_power_mw = 10**(params.get('tx_power_dbm', 0) / 10)
        
        # Driver power (proportional to Vpp^2)
        vpp = params.get('vpp', 2.0)
        driver_power_mw = 20 * (vpp / 2.0)**2  # 20mW at 2V nominal
        
        # Bias power
        bias_power_mw = 10  # Fixed bias circuits
        
        # TIA power
        tia_power_mw = 30  # Fixed TIA power
        
        total_power_mw = tx_power_mw + driver_power_mw + bias_power_mw + tia_power_mw
        
        return total_power_mw
    
    def find_pareto_frontier(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find Pareto-optimal solutions
        
        Args:
            df: DataFrame with optimization results
            
        Returns:
            DataFrame with Pareto-optimal solutions
        """
        # Filter solutions that meet BER target
        valid_df = df[df['meets_target']].copy()
        
        if len(valid_df) == 0:
            print("Warning: No solutions meet BER target!")
            return pd.DataFrame()
        
        # Sort by power consumption
        valid_df = valid_df.sort_values('power_consumption_mw')
        
        # Find Pareto frontier (minimize power, maximize margin)
        pareto = []
        best_ber = 1.0
        
        for _, row in valid_df.iterrows():
            if row['ber'] < best_ber:
                pareto.append(row)
                best_ber = row['ber']
        
        pareto_df = pd.DataFrame(pareto)
        return pareto_df
    
    def plot_pareto(self, df: pd.DataFrame, pareto_df: pd.DataFrame):
        """Plot Pareto frontier
        
        Args:
            df: All results
            pareto_df: Pareto-optimal results
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Power vs BER
        valid = df[df['meets_target']]
        invalid = df[~df['meets_target']]
        
        ax1.scatter(invalid['power_consumption_mw'], invalid['ber'], 
                   alpha=0.3, color='red', label='Fails target')
        ax1.scatter(valid['power_consumption_mw'], valid['ber'], 
                   alpha=0.5, color='blue', label='Meets target')
        ax1.scatter(pareto_df['power_consumption_mw'], pareto_df['ber'], 
                   s=100, color='green', marker='*', label='Pareto optimal')
        
        ax1.set_xlabel('Power Consumption (mW)')
        ax1.set_ylabel('BER')
        ax1.set_yscale('log')
        ax1.axhline(y=self.target_ber, color='r', linestyle='--', label='Target BER')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Power vs BER Trade-off')
        
        # Plot 2: Design space
        ax2.scatter(df['vpp'], df['tx_power_dbm'], 
                   c=df['power_consumption_mw'], cmap='viridis', alpha=0.6)
        ax2.scatter(pareto_df['vpp'], pareto_df['tx_power_dbm'], 
                   s=100, color='red', marker='*', label='Pareto optimal')
        
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Power (mW)')
        
        ax2.set_xlabel('Drive Voltage Vpp (V)')
        ax2.set_ylabel('Tx Power (dBm)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Design Space Exploration')
        
        plt.suptitle('Transceiver Parameter Optimization')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/figs/pareto_tx_power_vs_margin.png', dpi=150)
        plt.close()
    
    def run_optimization(self) -> pd.DataFrame:
        """Run complete optimization
        
        Returns:
            DataFrame with optimization results
        """
        print("Starting parameter optimization...")
        
        # Define search space
        param_ranges = {
            'tx_power_dbm': np.linspace(-5, 5, 5),
            'vpp': np.linspace(1.5, 3.0, 4),
            'vpi': [2.5, 3.0, 3.5],
            'extinction_ratio_db': [6, 8, 10]
        }
        
        # Grid search
        results_df = self.grid_search(param_ranges)
        
        # Find Pareto frontier
        pareto_df = self.find_pareto_frontier(results_df)
        
        # Save results
        results_df.to_csv(f'{self.results_dir}/tables/optimization_results.csv', index=False)
        pareto_df.to_csv(f'{self.results_dir}/tables/pareto_optimal.csv', index=False)
        
        # Plot
        self.plot_pareto(results_df, pareto_df)
        
        # Print summary
        print("\nOptimization Results:")
        print("=" * 50)
        if len(pareto_df) > 0:
            best = pareto_df.iloc[0]
            print(f"Minimum power solution:")
            print(f"  Power consumption: {best['power_consumption_mw']:.1f} mW")
            print(f"  Tx power: {best['tx_power_dbm']:.1f} dBm")
            print(f"  Drive voltage: {best['vpp']:.1f} V")
            print(f"  Vπ: {best['vpi']:.1f} V")
            print(f"  Extinction ratio: {best['extinction_ratio_db']:.1f} dB")
            print(f"  Achieved BER: {best['ber']:.2e}")
        else:
            print("No solutions found that meet target BER!")
        
        return pareto_df


def main():
    """Main entry point"""
    optimizer = ParameterOptimizer(target_ber=1e-12)
    results = optimizer.run_optimization()
    
    print(f"\nOptimization complete!")
    print(f"Results saved to results/tables/")
    print(f"Pareto plot saved to results/figs/pareto_tx_power_vs_margin.png")


if __name__ == '__main__':
    main()