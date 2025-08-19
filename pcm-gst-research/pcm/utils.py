"""
Utility Functions for PCM Simulations
=====================================
Helper functions for data I/O, plotting, and analysis.
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
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return filepath


def save_dataframe(df: pd.DataFrame, name: str, outdir: str = "data"):
    """Save DataFrame to CSV."""
    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, f"{name}.csv")
    df.to_csv(filepath, index=False)
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
    return output_name


def plot_pulse_results(data: Dict, title_prefix: str = "Pulse") -> Dict[str, plt.Figure]:
    """Create standard plots for pulse simulation results."""
    figs = {}
    t_ns = data["t_s"] * 1e9
    
    # Current vs time
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_ns, data["I_A"] * 1e3, linewidth=2)
    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel("Current (mA)", fontsize=12)
    ax.set_title(f"{title_prefix}: Current vs Time", fontsize=14)
    ax.grid(True, alpha=0.3)
    figs["current"] = fig
    
    # Temperature vs time
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_ns, data["T_K"], linewidth=2, color='red')
    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel("Temperature (K)", fontsize=12)
    ax.set_title(f"{title_prefix}: Temperature vs Time", fontsize=14)
    ax.grid(True, alpha=0.3)
    # Add melting temperature line
    if "Tm_K" in data:
        ax.axhline(data["Tm_K"], color='k', linestyle='--', alpha=0.3, label='Tm')
        ax.legend()
    figs["temperature"] = fig
    
    # Crystalline fraction vs time
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_ns, data["X"], linewidth=2, color='green')
    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel("Crystalline fraction X", fontsize=12)
    ax.set_title(f"{title_prefix}: Phase Evolution", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    figs["phase"] = fig
    
    # Resistance vs time (log scale)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(t_ns, np.maximum(data["R_ohm"], 1e0), linewidth=2, color='purple')
    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel("Device Resistance (Ω)", fontsize=12)
    ax.set_title(f"{title_prefix}: Resistance vs Time", fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    figs["resistance"] = fig
    
    return figs


def plot_combined_pulse_results(reset_data: Dict, set_data: Dict) -> plt.Figure:
    """Create combined plot showing RESET and SET operations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # RESET data
    t_reset_ns = reset_data["t_s"] * 1e9
    axes[0, 0].plot(t_reset_ns, reset_data["T_K"], 'r-', label='RESET')
    axes[0, 0].set_ylabel("Temperature (K)")
    axes[0, 0].set_title("Temperature Evolution")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].plot(t_reset_ns, reset_data["X"], 'b-', label='RESET')
    axes[0, 1].set_ylabel("Crystalline Fraction")
    axes[0, 1].set_title("Phase State")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # SET data
    t_set_ns = set_data["t_s"] * 1e9
    axes[1, 0].plot(t_set_ns, set_data["T_K"], 'r--', label='SET')
    axes[1, 0].set_xlabel("Time (ns)")
    axes[1, 0].set_ylabel("Temperature (K)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].plot(t_set_ns, set_data["X"], 'b--', label='SET')
    axes[1, 1].set_xlabel("Time (ns)")
    axes[1, 1].set_ylabel("Crystalline Fraction")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    fig.suptitle("PCM Operation: RESET vs SET", fontsize=16)
    plt.tight_layout()
    
    return fig


def generate_switching_map(device, voltages: np.ndarray, durations: np.ndarray,
                          initial_X: float = 0.0) -> Dict:
    """
    Generate 2D switching map for voltage-duration parameter space.
    
    Args:
        device: PCMDevice instance
        voltages: Array of voltages to test
        durations: Array of pulse durations to test
        initial_X: Initial crystalline fraction
        
    Returns:
        Dictionary with switching map data
    """
    final_X = np.zeros((len(durations), len(voltages)))
    peak_T = np.zeros((len(durations), len(voltages)))
    
    for i, duration in enumerate(durations):
        for j, voltage in enumerate(voltages):
            t, V = device.create_set_pulse(duration, voltage)
            result = device.simulate_voltage_pulse(V, t, X0=initial_X)
            final_X[i, j] = result['X'][-1]
            peak_T[i, j] = result['T_K'].max()
    
    return {
        "voltages": voltages,
        "durations": durations,
        "final_X": final_X,
        "peak_T": peak_T
    }


def plot_switching_map(data: Dict) -> plt.Figure:
    """Plot 2D switching map."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Final crystalline fraction
    im1 = ax1.imshow(data["final_X"], aspect='auto', origin='lower',
                     extent=[data["voltages"].min(), data["voltages"].max(),
                            data["durations"].min()*1e9, data["durations"].max()*1e9],
                     cmap='viridis', vmin=0, vmax=1)
    ax1.set_xlabel("Voltage (V)", fontsize=12)
    ax1.set_ylabel("Duration (ns)", fontsize=12)
    ax1.set_title("Final Crystalline Fraction", fontsize=14)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("X", fontsize=12)
    
    # Peak temperature
    im2 = ax2.imshow(data["peak_T"], aspect='auto', origin='lower',
                     extent=[data["voltages"].min(), data["voltages"].max(),
                            data["durations"].min()*1e9, data["durations"].max()*1e9],
                     cmap='hot')
    ax2.set_xlabel("Voltage (V)", fontsize=12)
    ax2.set_ylabel("Duration (ns)", fontsize=12)
    ax2.set_title("Peak Temperature", fontsize=14)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("T (K)", fontsize=12)
    
    fig.suptitle("Switching Parameter Map", fontsize=16)
    plt.tight_layout()
    
    return fig


def analyze_variability(mc_results: pd.DataFrame) -> Dict:
    """Analyze Monte Carlo variability results."""
    analysis = {
        "mean_X": mc_results['final_X'].mean(),
        "std_X": mc_results['final_X'].std(),
        "cv_X": mc_results['final_X'].std() / mc_results['final_X'].mean(),
        "mean_T": mc_results['peak_T'].mean(),
        "std_T": mc_results['peak_T'].std(),
        "mean_R": mc_results['final_R'].mean(),
        "std_R": mc_results['final_R'].std(),
        "success_rate": (mc_results['final_X'] > 0.9).mean()
    }
    
    return analysis


def create_latex_report(results: Dict, output_file: str = "report.tex"):
    """Generate LaTeX report from simulation results."""
    latex_content = r"""
\documentclass[11pt]{article}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{hyperref}

\title{Phase Change Memory (PCM) Simulation Report\\
\large Ge-Sb-Te Material System}
\author{PCM Research Framework}
\date{\today}

\begin{document}
\maketitle

\section{Introduction}
This report presents comprehensive simulation results for Phase Change Memory devices 
based on Ge-Sb-Te (GST) chalcogenide materials.

\section{Material Properties}
\begin{itemize}
    \item Active layer thickness: """ + f"{results.get('thickness_nm', 50):.0f}" + r""" nm
    \item Resistance ratio: """ + f"{results.get('R_ratio', 1000):.0f}" + r"""x
    \item Melting temperature: """ + f"{results.get('Tm_K', 900):.0f}" + r""" K
    \item Crystallization onset: """ + f"{results.get('Tc_K', 430):.0f}" + r""" K
\end{itemize}

\section{JMAK Kinetics}
The Johnson-Mehl-Avrami-Kolmogorov equation describes the crystallization kinetics:
\begin{equation}
    X(t) = 1 - \exp[-(kt)^n]
\end{equation}
where $k = k_0 \exp(-E_a/k_B T)$.

\section{Device Performance}
\subsection{RESET Operation}
Successfully achieved amorphization with X = """ + f"{results.get('reset_final_X', 0.0):.3f}" + r"""

\subsection{SET Operation}
Successfully achieved crystallization with X = """ + f"{results.get('set_final_X', 1.0):.3f}" + r"""

\section{Reliability}
\subsection{Retention}
Data retention follows Arrhenius behavior with activation energy 
$E_a$ = """ + f"{results.get('Ea_eV', 2.1):.2f}" + r""" eV.

\subsection{Endurance}
Cycling endurance follows Weibull distribution with characteristic life 
$\eta$ = """ + f"{results.get('eta', 1e6):.0e}" + r""" cycles.

\end{document}
"""
    
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    return output_file