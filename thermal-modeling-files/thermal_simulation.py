#!/usr/bin/env python3
"""
Main thermal simulation script for InGaN/GaN HEMT
Performs steady-state and transient thermal analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import argparse
import sys
import os

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from device_structures.hemt_structure import HEMTStructure, AdvancedHEMTStructure
from simulations.python.thermal_models import (
    HeatEquationSolver, ThermalResistanceNetwork, 
    ElectroThermalModel, create_example_power_distribution
)


class HEMTThermalSimulator:
    """Complete thermal simulation framework for InGaN/GaN HEMTs"""
    
    def __init__(self, device_structure, mesh_resolution='medium'):
        """
        Initialize thermal simulator
        
        Args:
            device_structure: HEMTStructure object
            mesh_resolution: 'coarse', 'medium', 'fine', or 'ultra'
        """
        self.device = device_structure
        
        # Set mesh parameters based on resolution
        mesh_params = {
            'coarse': (30, 15, 30),
            'medium': (50, 25, 50),
            'fine': (100, 50, 100),
            'ultra': (200, 100, 200)
        }
        
        nx, ny, nz = mesh_params.get(mesh_resolution, mesh_params['medium'])
        
        # Initialize solvers
        self.heat_solver = HeatEquationSolver(device_structure, nx=nx, ny=ny, nz=nz)
        self.trn_solver = ThermalResistanceNetwork(device_structure)
        self.electro_thermal = ElectroThermalModel(device_structure, nx=nx, ny=ny, nz=nz)
        
        # Storage for results
        self.results = {}
    
    def run_dc_thermal_analysis(self, V_ds, I_ds, save_results=True):
        """
        Run steady-state DC thermal analysis
        
        Args:
            V_ds: Drain-source voltage (V)
            I_ds: Drain current (A)
            save_results: Whether to save results to file
        
        Returns:
            Dictionary with thermal analysis results
        """
        print(f"\nRunning DC thermal analysis: Vds={V_ds}V, Ids={I_ds}A")
        print("=" * 60)
        
        # Total power dissipation
        P_total = V_ds * I_ds
        print(f"Total power dissipation: {P_total:.3f} W")
        
        # Quick thermal resistance calculation
        T_j_trn = self.trn_solver.calculate_junction_temperature(P_total)
        print(f"\nThermal resistance network:")
        print(f"  Junction temperature: {T_j_trn-273.15:.1f} °C")
        print(f"  Temperature rise: {T_j_trn-self.trn_solver.T_ambient:.1f} K")
        print(f"  Total thermal resistance: {(T_j_trn-self.trn_solver.T_ambient)/P_total:.2f} K/W")
        
        # Detailed heat equation solution
        print("\nSolving 3D heat equation...")
        power_func = create_example_power_distribution(self.device, V_ds, I_ds)
        T_field = self.heat_solver.solve_steady_state(power_func)
        
        # Extract key temperatures
        z_2deg = self.device.get_2deg_position() * 1e-9
        z_idx = np.argmin(np.abs(self.heat_solver.z - z_2deg))
        
        # Temperature at different locations
        T_max = np.max(T_field)
        T_junction = T_field[self.heat_solver.nx//2, self.heat_solver.ny//2, z_idx]
        T_substrate = T_field[self.heat_solver.nx//2, self.heat_solver.ny//2, -1]
        
        # Temperature profile along channel
        gate_start_idx = int(1.5e-6 / self.heat_solver.dx)
        gate_end_idx = int((1.5e-6 + 0.25e-6) / self.heat_solver.dx)
        T_channel = T_field[:, self.heat_solver.ny//2, z_idx]
        
        print(f"\n3D thermal simulation results:")
        print(f"  Maximum temperature: {T_max-273.15:.1f} °C")
        print(f"  Junction temperature: {T_junction-273.15:.1f} °C")
        print(f"  Substrate temperature: {T_substrate-273.15:.1f} °C")
        print(f"  Peak channel temperature: {np.max(T_channel)-273.15:.1f} °C")
        
        # Store results
        results = {
            'V_ds': V_ds,
            'I_ds': I_ds,
            'P_total': P_total,
            'T_j_trn': T_j_trn,
            'T_max': T_max,
            'T_junction': T_junction,
            'T_substrate': T_substrate,
            'T_field': T_field,
            'T_channel': T_channel,
            'thermal_resistance': (T_junction - self.heat_solver.T_ambient) / P_total
        }
        
        self.results['dc_analysis'] = results
        
        if save_results:
            self._save_results('dc_thermal_results.npz', results)
        
        return results
    
    def run_transient_thermal_analysis(self, V_ds, I_ds, t_pulse, t_period, 
                                     n_cycles=5, save_results=True):
        """
        Run transient thermal analysis for pulsed operation
        
        Args:
            V_ds: Drain-source voltage (V)
            I_ds: Drain current (A)
            t_pulse: Pulse width (s)
            t_period: Period (s)
            n_cycles: Number of cycles to simulate
            save_results: Whether to save results
        
        Returns:
            Dictionary with transient analysis results
        """
        print(f"\nRunning transient thermal analysis:")
        print(f"  Vds={V_ds}V, Ids={I_ds}A")
        print(f"  Pulse: {t_pulse*1e6:.1f}μs, Period: {t_period*1e3:.1f}ms")
        print(f"  Duty cycle: {100*t_pulse/t_period:.1f}%")
        print("=" * 60)
        
        # Time points
        dt = min(t_pulse/20, t_period/100)
        t_total = n_cycles * t_period
        time_points = np.arange(0, t_total, dt)
        
        # Initialize temperature
        T = np.ones_like(self.heat_solver.X) * self.heat_solver.T_ambient
        T_junction_history = []
        T_max_history = []
        
        # Power function
        power_func = create_example_power_distribution(self.device, V_ds, I_ds)
        
        # Get 2DEG position
        z_2deg = self.device.get_2deg_position() * 1e-9
        z_idx = np.argmin(np.abs(self.heat_solver.z - z_2deg))
        
        print("Simulating transient response...")
        for i, t in enumerate(time_points):
            # Check if power is on or off
            cycle_time = t % t_period
            power_on = cycle_time < t_pulse
            
            if power_on:
                # Solve with heat source
                T = self._solve_transient_step(T, dt, power_func)
            else:
                # Solve without heat source (cooling)
                T = self._solve_transient_step(T, dt, lambda x,y,z: 0)
            
            # Record temperatures
            T_junction = T[self.heat_solver.nx//2, self.heat_solver.ny//2, z_idx]
            T_junction_history.append(T_junction)
            T_max_history.append(np.max(T))
            
            # Progress update
            if (i+1) % (len(time_points)//10) == 0:
                print(f"  Progress: {100*(i+1)/len(time_points):.0f}%")
        
        # Convert to arrays
        T_junction_history = np.array(T_junction_history)
        T_max_history = np.array(T_max_history)
        
        # Calculate steady-state values
        T_j_ss_max = np.max(T_junction_history[-len(time_points)//n_cycles:])
        T_j_ss_min = np.min(T_junction_history[-len(time_points)//n_cycles:])
        T_j_ss_avg = np.mean(T_junction_history[-len(time_points)//n_cycles:])
        
        print(f"\nTransient analysis results:")
        print(f"  Steady-state max temperature: {T_j_ss_max-273.15:.1f} °C")
        print(f"  Steady-state min temperature: {T_j_ss_min-273.15:.1f} °C")
        print(f"  Steady-state avg temperature: {T_j_ss_avg-273.15:.1f} °C")
        print(f"  Temperature swing: {T_j_ss_max-T_j_ss_min:.1f} K")
        
        # Store results
        results = {
            'V_ds': V_ds,
            'I_ds': I_ds,
            't_pulse': t_pulse,
            't_period': t_period,
            'duty_cycle': t_pulse/t_period,
            'time': time_points,
            'T_junction': T_junction_history,
            'T_max': T_max_history,
            'T_j_ss_max': T_j_ss_max,
            'T_j_ss_min': T_j_ss_min,
            'T_j_ss_avg': T_j_ss_avg,
            'T_final': T
        }
        
        self.results['transient_analysis'] = results
        
        if save_results:
            self._save_results('transient_thermal_results.npz', results)
        
        return results
    
    def run_self_consistent_analysis(self, V_gs, V_ds, max_iterations=10):
        """
        Run self-consistent electro-thermal analysis
        
        Args:
            V_gs: Gate-source voltage (V)
            V_ds: Drain-source voltage (V)
            max_iterations: Maximum iterations for convergence
        
        Returns:
            Converged temperature field
        """
        print(f"\nRunning self-consistent electro-thermal analysis:")
        print(f"  Vgs={V_gs}V, Vds={V_ds}V")
        print("=" * 60)
        
        T_field = self.electro_thermal.self_consistent_solve(
            V_gs, V_ds, max_iterations=max_iterations
        )
        
        # Extract results
        z_2deg = self.device.get_2deg_position() * 1e-9
        z_idx = np.argmin(np.abs(self.electro_thermal.z - z_2deg))
        
        T_max = np.max(T_field)
        T_junction = T_field[self.electro_thermal.nx//2, self.electro_thermal.ny//2, z_idx]
        
        print(f"\nConverged results:")
        print(f"  Maximum temperature: {T_max-273.15:.1f} °C")
        print(f"  Junction temperature: {T_junction-273.15:.1f} °C")
        
        results = {
            'V_gs': V_gs,
            'V_ds': V_ds,
            'T_field': T_field,
            'T_max': T_max,
            'T_junction': T_junction
        }
        
        self.results['self_consistent'] = results
        return T_field
    
    def analyze_thermal_impedance(self, frequencies):
        """
        Analyze frequency-dependent thermal impedance
        
        Args:
            frequencies: Array of frequencies (Hz)
        
        Returns:
            Complex thermal impedance
        """
        print("\nAnalyzing thermal impedance spectrum...")
        
        Z_th = self.trn_solver.calculate_thermal_impedance(frequencies)
        
        results = {
            'frequencies': frequencies,
            'Z_th': Z_th,
            'magnitude': np.abs(Z_th),
            'phase': np.angle(Z_th, deg=True)
        }
        
        self.results['thermal_impedance'] = results
        return results
    
    def _solve_transient_step(self, T_prev, dt, power_func):
        """Solve one time step of transient heat equation"""
        # Build system matrix for transient
        A = self.heat_solver.build_system_matrix(dt=dt)
        
        # Right-hand side includes previous temperature
        n_total = self.heat_solver.nx * self.heat_solver.ny * self.heat_solver.nz
        b = np.zeros(n_total)
        
        # Add previous temperature contribution
        T_prev_flat = T_prev.flatten()
        for i in range(n_total):
            # Get material properties at this point
            ix = i % self.heat_solver.nx
            iy = (i // self.heat_solver.nx) % self.heat_solver.ny
            iz = i // (self.heat_solver.nx * self.heat_solver.ny)
            
            rho_c = (self.heat_solver.density[ix, iy, iz] * 
                    self.heat_solver.specific_heat[ix, iy, iz])
            b[i] = -rho_c * T_prev_flat[i] / dt
        
        # Add heat source
        b = self.heat_solver.add_heat_source(b, power_func)
        
        # Apply boundary conditions
        A, b = self.heat_solver.apply_boundary_conditions(A, b)
        
        # Solve
        from scipy.sparse.linalg import spsolve
        T_flat = spsolve(A, b)
        
        return T_flat.reshape(T_prev.shape)
    
    def _save_results(self, filename, results):
        """Save results to file"""
        # Convert arrays to save format
        save_dict = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                save_dict[key] = value
            else:
                save_dict[key] = np.array(value)
        
        filepath = os.path.join('../../results', filename)
        np.savez_compressed(filepath, **save_dict)
        print(f"Results saved to: {filepath}")


def plot_thermal_results(simulator, results_type='dc'):
    """
    Plot thermal simulation results
    
    Args:
        simulator: HEMTThermalSimulator object with results
        results_type: 'dc', 'transient', or 'impedance'
    """
    if results_type == 'dc' and 'dc_analysis' in simulator.results:
        plot_dc_thermal_results(simulator)
    elif results_type == 'transient' and 'transient_analysis' in simulator.results:
        plot_transient_thermal_results(simulator)
    elif results_type == 'impedance' and 'thermal_impedance' in simulator.results:
        plot_thermal_impedance(simulator)
    else:
        print(f"No {results_type} results available to plot")


def plot_dc_thermal_results(simulator):
    """Plot DC thermal analysis results"""
    results = simulator.results['dc_analysis']
    solver = simulator.heat_solver
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Temperature distribution along channel
    ax = axes[0, 0]
    x_um = solver.x * 1e6
    T_channel = results['T_channel'] - 273.15  # Convert to °C
    ax.plot(x_um, T_channel, 'b-', linewidth=2)
    ax.axvspan(1.5, 1.75, alpha=0.3, color='red', label='Gate')
    ax.set_xlabel('Position along channel (μm)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Channel Temperature Profile')
    ax.grid(True)
    ax.legend()
    
    # 2. Cross-section temperature map
    ax = axes[0, 1]
    T_xz = results['T_field'][:, solver.ny//2, :].T - 273.15
    extent = [0, solver.x[-1]*1e6, solver.z[-1]*1e9, 0]
    im = ax.imshow(T_xz, extent=extent, aspect='auto', cmap='hot')
    ax.set_xlabel('X position (μm)')
    ax.set_ylabel('Z position (nm)')
    ax.set_title('Temperature Cross-Section (Y=Ly/2)')
    plt.colorbar(im, ax=ax, label='Temperature (°C)')
    
    # 3. Top view temperature map at 2DEG
    ax = axes[1, 0]
    z_2deg = simulator.device.get_2deg_position() * 1e-9
    z_idx = np.argmin(np.abs(solver.z - z_2deg))
    T_xy = results['T_field'][:, :, z_idx].T - 273.15
    extent = [0, solver.x[-1]*1e6, 0, solver.y[-1]*1e6]
    im = ax.imshow(T_xy, extent=extent, aspect='auto', cmap='hot', origin='lower')
    ax.set_xlabel('X position (μm)')
    ax.set_ylabel('Y position (μm)')
    ax.set_title(f'Temperature at 2DEG (Z={z_2deg*1e9:.1f}nm)')
    plt.colorbar(im, ax=ax, label='Temperature (°C)')
    
    # 4. Summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""Thermal Analysis Summary
    
Operating Conditions:
  Vds = {results['V_ds']:.1f} V
  Ids = {results['I_ds']*1e3:.1f} mA
  Power = {results['P_total']:.3f} W

Temperature Results:
  Max Temperature: {results['T_max']-273.15:.1f} °C
  Junction Temp: {results['T_junction']-273.15:.1f} °C
  Substrate Temp: {results['T_substrate']-273.15:.1f} °C
  
Thermal Resistance:
  Rth = {results['thermal_resistance']:.2f} K/W"""
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('InGaN/GaN HEMT DC Thermal Analysis', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_transient_thermal_results(simulator):
    """Plot transient thermal analysis results"""
    results = simulator.results['transient_analysis']
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 1. Junction temperature vs time
    ax = axes[0]
    time_ms = results['time'] * 1e3
    T_j = results['T_junction'] - 273.15
    ax.plot(time_ms, T_j, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Junction Temperature (°C)')
    ax.set_title('Transient Junction Temperature')
    ax.grid(True)
    
    # Add duty cycle indication
    t_period_ms = results['t_period'] * 1e3
    t_pulse_ms = results['t_pulse'] * 1e3
    for i in range(int(time_ms[-1] / t_period_ms)):
        ax.axvspan(i*t_period_ms, i*t_period_ms + t_pulse_ms, 
                  alpha=0.2, color='red')
    
    # 2. Temperature swing analysis
    ax = axes[1]
    # Extract last cycle
    n_points_per_cycle = int(results['t_period'] / (results['time'][1] - results['time'][0]))
    last_cycle_idx = -n_points_per_cycle
    time_cycle = (results['time'][last_cycle_idx:] - results['time'][last_cycle_idx]) * 1e6
    T_cycle = results['T_junction'][last_cycle_idx:] - 273.15
    
    ax.plot(time_cycle, T_cycle, 'g-', linewidth=2)
    ax.axhline(results['T_j_ss_max']-273.15, color='r', linestyle='--', label='Max')
    ax.axhline(results['T_j_ss_min']-273.15, color='b', linestyle='--', label='Min')
    ax.axhline(results['T_j_ss_avg']-273.15, color='k', linestyle='--', label='Avg')
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Junction Temperature (°C)')
    ax.set_title('Steady-State Temperature Cycle')
    ax.grid(True)
    ax.legend()
    
    # Add text summary
    duty_cycle = results['duty_cycle'] * 100
    temp_swing = results['T_j_ss_max'] - results['T_j_ss_min']
    ax.text(0.6, 0.2, f'Duty Cycle: {duty_cycle:.1f}%\nTemp Swing: {temp_swing:.1f}K', 
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))
    
    plt.suptitle('InGaN/GaN HEMT Transient Thermal Analysis', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_thermal_impedance(simulator):
    """Plot thermal impedance spectrum"""
    results = simulator.results['thermal_impedance']
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 1. Magnitude
    ax = axes[0]
    ax.loglog(results['frequencies'], results['magnitude'], 'b-', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|Z_th| (K/W)')
    ax.set_title('Thermal Impedance Magnitude')
    ax.grid(True, which='both', alpha=0.3)
    
    # 2. Phase
    ax = axes[1]
    ax.semilogx(results['frequencies'], results['phase'], 'r-', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase (degrees)')
    ax.set_title('Thermal Impedance Phase')
    ax.grid(True)
    ax.set_ylim(-90, 0)
    
    plt.suptitle('InGaN/GaN HEMT Thermal Impedance Spectrum', fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run thermal simulations"""
    parser = argparse.ArgumentParser(description='InGaN/GaN HEMT Thermal Simulation')
    parser.add_argument('--substrate', type=str, default='SiC',
                       choices=['SiC', 'Si', 'Sapphire', 'GaN'],
                       help='Substrate material')
    parser.add_argument('--resolution', type=str, default='medium',
                       choices=['coarse', 'medium', 'fine', 'ultra'],
                       help='Mesh resolution')
    parser.add_argument('--vds', type=float, default=20.0,
                       help='Drain-source voltage (V)')
    parser.add_argument('--ids', type=float, default=0.1,
                       help='Drain current (A)')
    parser.add_argument('--analysis', type=str, default='all',
                       choices=['dc', 'transient', 'impedance', 'self-consistent', 'all'],
                       help='Type of analysis to run')
    
    args = parser.parse_args()
    
    # Create device structure
    print(f"Creating InGaN/GaN HEMT structure on {args.substrate} substrate...")
    device = AdvancedHEMTStructure(substrate_type=args.substrate)
    device.add_back_barrier()  # Add back barrier for better confinement
    device.print_structure()
    
    # Initialize simulator
    print(f"\nInitializing thermal simulator with {args.resolution} resolution...")
    simulator = HEMTThermalSimulator(device, mesh_resolution=args.resolution)
    
    # Run analyses based on user selection
    if args.analysis in ['dc', 'all']:
        results_dc = simulator.run_dc_thermal_analysis(args.vds, args.ids)
        plot_thermal_results(simulator, 'dc')
    
    if args.analysis in ['transient', 'all']:
        # Transient analysis with 10μs pulse, 1ms period (1% duty cycle)
        results_transient = simulator.run_transient_thermal_analysis(
            args.vds, args.ids, t_pulse=10e-6, t_period=1e-3, n_cycles=5
        )
        plot_thermal_results(simulator, 'transient')
    
    if args.analysis in ['impedance', 'all']:
        # Thermal impedance from 1 Hz to 1 MHz
        frequencies = np.logspace(0, 6, 100)
        results_impedance = simulator.analyze_thermal_impedance(frequencies)
        plot_thermal_results(simulator, 'impedance')
    
    if args.analysis in ['self-consistent', 'all']:
        # Self-consistent electro-thermal analysis
        V_gs = -2.0  # Gate voltage
        T_field = simulator.run_self_consistent_analysis(V_gs, args.vds)
    
    print("\nSimulation completed successfully!")
    
    # Show all plots
    plt.show()


if __name__ == "__main__":
    main()