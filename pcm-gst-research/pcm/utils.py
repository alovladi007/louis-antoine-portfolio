"""
Utilities Module
================
Helper functions for PCM simulations.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import zipfile
from datetime import datetime


def load_params(filepath: str = "params.json") -> Dict:
    """Load parameters from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_params(params: Dict, filepath: str = "params.json"):
    """Save parameters to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=2)


def save_figure(fig: plt.Figure, name: str, outdir: str = "data"):
    """Save figure to file."""
    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, f"{name}.png")
    fig.savefig(filepath, dpi=180, bbox_inches='tight')
    print(f"Saved figure: {filepath}")
    return filepath


def save_dataframe(df: pd.DataFrame, name: str, outdir: str = "data"):
    """Save DataFrame to CSV."""
    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, f"{name}.csv")
    df.to_csv(filepath, index=False)
    print(f"Saved data: {filepath}")
    return filepath


def create_project_bundle(project_dir: str = ".", 
                         output_name: str = "pcm_gst_bundle.zip"):
    """Create ZIP bundle of all project files."""
    with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(project_dir):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') 
                      and d != '__pycache__']
            for file in files:
                if not file.startswith('.') and not file.endswith('.pyc'):
                    filepath = os.path.join(root, file)
                    arcname = os.path.relpath(filepath, project_dir)
                    zf.write(filepath, arcname)
    print(f"Created bundle: {output_name}")
    return output_name


def plot_pulse_results(data: Dict, title_prefix: str = "Pulse") -> Dict[str, plt.Figure]:
    """Create standard plots for pulse simulation results."""
    figs = {}
    t_ns = data["t_s"] * 1e9
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Current vs time
    ax = axes[0, 0]
    ax.plot(t_ns, data["I_A"] * 1e3, 'b-', linewidth=2)
    ax.set_xlabel("Time (ns)", fontsize=11)
    ax.set_ylabel("Current (mA)", fontsize=11)
    ax.set_title(f"{title_prefix}: Current vs Time", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Temperature vs time
    ax = axes[0, 1]
    ax.plot(t_ns, data["T_K"], 'r-', linewidth=2)
    ax.set_xlabel("Time (ns)", fontsize=11)
    ax.set_ylabel("Temperature (K)", fontsize=11)
    ax.set_title(f"{title_prefix}: Temperature vs Time", fontsize=12)
    ax.grid(True, alpha=0.3)
    # Add melting temperature line
    if "Tm_K" in data:
        ax.axhline(data.get("Tm_K", 900), color='orange', linestyle='--', 
                   alpha=0.5, label='Melting')
    ax.axhline(430, color='green', linestyle='--', alpha=0.5, label='Crystallization')
    ax.legend(loc='best', fontsize=9)
    
    # Crystalline fraction vs time
    ax = axes[1, 0]
    ax.plot(t_ns, data["X"], 'g-', linewidth=2)
    ax.set_xlabel("Time (ns)", fontsize=11)
    ax.set_ylabel("Crystalline fraction X", fontsize=11)
    ax.set_title(f"{title_prefix}: Phase Evolution", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    
    # Resistance vs time
    ax = axes[1, 1]
    ax.semilogy(t_ns, np.maximum(data["R_ohm"], 1e0), 'purple', linewidth=2)
    ax.set_xlabel("Time (ns)", fontsize=11)
    ax.set_ylabel("Device Resistance (Ω)", fontsize=11)
    ax.set_title(f"{title_prefix}: Resistance vs Time", fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.suptitle(f"{title_prefix} Operation Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    
    figs["combined"] = fig
    
    return figs


def plot_resistance_vs_crystallinity(X_array: np.ndarray = None, 
                                    T: float = 300.0) -> plt.Figure:
    """Plot resistance vs crystalline fraction."""
    if X_array is None:
        X_array = np.linspace(0, 1, 100)
    
    # Calculate resistance using percolation model
    rho_a = 1.0  # Amorphous resistivity
    rho_c = 1e-3  # Crystalline resistivity
    
    # Percolation mixing
    sigma = X_array / rho_c + (1 - X_array) / rho_a
    R = 1 / sigma  # Normalized resistance
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(X_array, R, 'b-', linewidth=3)
    ax.set_xlabel("Crystalline Fraction X", fontsize=12)
    ax.set_ylabel("Device Resistance (Ω)", fontsize=12)
    ax.set_title(f"Resistance vs Crystalline Fraction @ {T:.0f} K", fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0, 1])
    
    # Add annotations
    ax.annotate(f'Amorphous\nR = {R[0]:.2e} Ω', 
                xy=(0, R[0]), xytext=(0.1, R[0]*2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='left')
    ax.annotate(f'Crystalline\nR = {R[-1]:.2e} Ω', 
                xy=(1, R[-1]), xytext=(0.9, R[-1]/2),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, ha='right')
    
    return fig


def generate_summary_report(results: Dict) -> str:
    """Generate a text summary report of simulation results."""
    report = []
    report.append("=" * 60)
    report.append("PCM GST SIMULATION SUMMARY REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    if "device_params" in results:
        report.append("DEVICE PARAMETERS:")
        report.append("-" * 30)
        params = results["device_params"]
        report.append(f"  Active Layer: {params.get('thickness_m', 50e-9)*1e9:.1f} nm GST")
        report.append(f"  Device Area: {params.get('area_m2', (50e-9)**2)*1e12:.1f} nm²")
        report.append(f"  Thermal Capacitance: {params.get('Cth_J_per_K', 1e-11):.2e} J/K")
        report.append(f"  Thermal Resistance: {params.get('Rth_K_per_W', 5e5):.2e} K/W")
        report.append("")
    
    if "reset_set_cycle" in results:
        cycle = results["reset_set_cycle"]
        report.append("PROGRAMMING OPERATIONS:")
        report.append("-" * 30)
        if "reset" in cycle:
            reset = cycle["reset"]
            report.append(f"  RESET Pulse:")
            report.append(f"    Duration: {(reset['t_s'][-1] - reset['t_s'][0])*1e9:.1f} ns")
            report.append(f"    Max Temperature: {np.max(reset['T_K']):.0f} K")
            report.append(f"    Final X: {reset['X'][-1]:.3f}")
            report.append(f"    Final R: {reset['R_ohm'][-1]:.2e} Ω")
        
        if "set" in cycle:
            set_data = cycle["set"]
            report.append(f"  SET Pulse:")
            report.append(f"    Duration: {(set_data['t_s'][-1] - set_data['t_s'][0])*1e9:.1f} ns")
            report.append(f"    Max Temperature: {np.max(set_data['T_K']):.0f} K")
            report.append(f"    Final X: {set_data['X'][-1]:.3f}")
            report.append(f"    Final R: {set_data['R_ohm'][-1]:.2e} Ω")
        
        if "resistance_ratio" in cycle:
            report.append(f"  Resistance Ratio: {cycle['resistance_ratio']:.0f}×")
        report.append("")
    
    if "retention" in results:
        ret = results["retention"]
        report.append("RETENTION ANALYSIS:")
        report.append("-" * 30)
        for T_C, t50_y in zip(ret["temp_C"], ret["t50_years"]):
            if t50_y < 1e10:
                report.append(f"  {T_C:3.0f}°C: {t50_y:.2e} years")
        report.append("")
    
    if "endurance" in results:
        end = results["endurance"]
        report.append("ENDURANCE ANALYSIS:")
        report.append("-" * 30)
        report.append(f"  Weibull Shape (β): {end['beta']:.1f}")
        report.append(f"  Characteristic Life (η): {end['eta']:.2e} cycles")
        if "percentiles" in end:
            report.append("  Percentiles:")
            for p, val in end["percentiles"].items():
                report.append(f"    {p:2d}%: {val:.2e} cycles")
        report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)


def compare_materials(materials: List[Dict]) -> plt.Figure:
    """Compare different PCM materials."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Placeholder for material comparison plots
    # This would be expanded with actual material data
    
    ax = axes[0, 0]
    ax.text(0.5, 0.5, "Crystallization Temperature", 
            ha='center', va='center', fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    ax = axes[0, 1]
    ax.text(0.5, 0.5, "Resistance Ratio", 
            ha='center', va='center', fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    ax = axes[1, 0]
    ax.text(0.5, 0.5, "Retention @ 85°C", 
            ha='center', va='center', fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    ax = axes[1, 1]
    ax.text(0.5, 0.5, "Endurance", 
            ha='center', va='center', fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.suptitle("PCM Material Comparison", fontsize=16)
    plt.tight_layout()
    
    return fig