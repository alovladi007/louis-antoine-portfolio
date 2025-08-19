#!/usr/bin/env python3
"""
PCM GST Research - Complete Simulation Script
==============================================
Run all PCM simulations and generate plots/reports.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import PCM modules
from pcm.kinetics import JMAKModel
from pcm.device import PCMDevice
from pcm.iv import ThresholdSwitching
from pcm.reliability import Reliability
from pcm.utils import (
    save_figure, save_dataframe, plot_pulse_results,
    plot_resistance_vs_crystallinity, generate_summary_report,
    save_params
)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def main():
    """Run complete PCM simulation suite."""
    print("=" * 60)
    print("PCM GST RESEARCH - SIMULATION SUITE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create output directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Store all results
    results = {}
    
    # ========== 1. JMAK Crystallization Kinetics ==========
    print("1. Running JMAK crystallization simulations...")
    jmak = JMAKModel()
    
    # Plot crystallization kinetics
    fig_jmak, df_jmak = jmak.plot_kinetics()
    save_figure(fig_jmak, "01_jmak_kinetics", "data")
    save_dataframe(df_jmak, "01_jmak_data", "data")
    plt.close(fig_jmak)
    
    # Calculate crystallization times
    temps = [450, 500, 550, 600, 650]
    t50_values = [jmak.time_to_crystallize(T, 0.5) for T in temps]
    df_t50 = pd.DataFrame({'Temperature_K': temps, 't50_s': t50_values})
    save_dataframe(df_t50, "01_crystallization_times", "data")
    
    print("   ✓ JMAK kinetics completed")
    
    # ========== 2. Device Electro-Thermal Model ==========
    print("\n2. Running device simulations...")
    device = PCMDevice()
    
    # Save device parameters
    save_params(device.params, "data/device_params.json")
    results["device_params"] = device.params
    
    # Resistance vs crystallinity
    fig_resist = plot_resistance_vs_crystallinity()
    save_figure(fig_resist, "02_resistance_vs_X", "data")
    plt.close(fig_resist)
    
    print("   ✓ Device model initialized")
    
    # ========== 3. RESET/SET Pulse Operations ==========
    print("\n3. Simulating RESET/SET pulses...")
    
    # Run complete cycle
    cycle_results = device.simulate_reset_set_cycle(X0=1.0)
    results["reset_set_cycle"] = cycle_results
    
    # Plot RESET pulse results
    figs_reset = plot_pulse_results(cycle_results["reset"], "RESET")
    save_figure(figs_reset["combined"], "03_RESET_pulse", "data")
    plt.close(figs_reset["combined"])
    
    # Plot SET pulse results
    figs_set = plot_pulse_results(cycle_results["set"], "SET")
    save_figure(figs_set["combined"], "03_SET_pulse", "data")
    plt.close(figs_set["combined"])
    
    # Save pulse data
    df_reset = pd.DataFrame(cycle_results["reset"])
    save_dataframe(df_reset, "03_RESET_data", "data")
    
    df_set = pd.DataFrame(cycle_results["set"])
    save_dataframe(df_set, "03_SET_data", "data")
    
    print(f"   ✓ RESET: X={cycle_results['reset']['X'][0]:.3f} → {cycle_results['reset']['X'][-1]:.3f}")
    print(f"   ✓ SET: X={cycle_results['set']['X'][0]:.3f} → {cycle_results['set']['X'][-1]:.3f}")
    print(f"   ✓ Resistance ratio: {cycle_results['resistance_ratio']:.0f}×")
    
    # ========== 4. Threshold Switching I-V ==========
    print("\n4. Generating I-V characteristics...")
    ts = ThresholdSwitching()
    
    # Generate I-V for different states
    X_states = [0.0, 0.5, 1.0]
    for X in X_states:
        fig_iv = ts.plot_iv(X_state=X)
        save_figure(fig_iv, f"04_IV_X{X:.1f}", "data")
        plt.close(fig_iv)
    
    # Comparison plot
    fig_iv_comp = ts.plot_iv_comparison(np.array(X_states))
    save_figure(fig_iv_comp, "04_IV_comparison", "data")
    plt.close(fig_iv_comp)
    
    print("   ✓ I-V characteristics completed")
    
    # ========== 5. Reliability Analysis ==========
    print("\n5. Running reliability analysis...")
    reliability = Reliability()
    
    # Retention analysis
    retention_data = reliability.retention_arrhenius()
    results["retention"] = retention_data
    
    fig_retention = reliability.plot_retention(retention_data)
    save_figure(fig_retention, "05_retention_arrhenius", "data")
    plt.close(fig_retention)
    
    df_retention = pd.DataFrame(retention_data)
    save_dataframe(df_retention, "05_retention_data", "data")
    
    # Endurance analysis
    endurance_data = reliability.endurance_weibull(n_samples=5000)
    results["endurance"] = endurance_data
    
    fig_endurance = reliability.plot_endurance(endurance_data)
    save_figure(fig_endurance, "05_endurance_weibull", "data")
    plt.close(fig_endurance)
    
    # Save endurance statistics
    df_endurance_stats = pd.DataFrame({
        'Metric': ['Beta (shape)', 'Eta (scale)', 'Mean cycles', 'Median cycles'],
        'Value': [
            endurance_data['beta'],
            endurance_data['eta'],
            np.mean(endurance_data['lifetimes']),
            np.median(endurance_data['lifetimes'])
        ]
    })
    save_dataframe(df_endurance_stats, "05_endurance_stats", "data")
    
    print("   ✓ Retention analysis completed")
    print("   ✓ Endurance analysis completed")
    
    # ========== 6. Monte Carlo Variability Analysis ==========
    print("\n6. Running Monte Carlo simulations...")
    mc_retention = reliability.monte_carlo_retention(n_devices=1000, T_C=85.0)
    
    # Plot distribution
    fig_mc, ax = plt.subplots(figsize=(10, 6))
    ax.hist(mc_retention['t50_years'], bins=50, density=True, 
            alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel("Retention Time (years)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(f"Monte Carlo Retention Distribution @ 85°C\n"
                f"Mean: {mc_retention['mean_t50_years']:.2e} years", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    save_figure(fig_mc, "06_monte_carlo_retention", "data")
    plt.close(fig_mc)
    
    print(f"   ✓ Monte Carlo: {len(mc_retention['t50_values'])} devices simulated")
    
    # ========== 7. Generate Summary Report ==========
    print("\n7. Generating summary report...")
    report = generate_summary_report(results)
    
    # Save report
    with open("reports/simulation_summary.txt", "w") as f:
        f.write(report)
    
    print("   ✓ Report generated")
    
    # ========== 8. Create Combined Summary Figure ==========
    print("\n8. Creating summary figure...")
    fig_summary = plt.figure(figsize=(16, 12))
    
    # JMAK kinetics
    ax1 = plt.subplot(3, 3, 1)
    temps = [400, 450, 500, 550, 600, 650]
    t = np.logspace(-9, -3, 100)
    for T in temps:
        X = jmak.simulate_isothermal(T, t)
        ax1.semilogx(t, X, label=f"{T}K")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("X")
    ax1.set_title("JMAK Kinetics")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Resistance vs X
    ax2 = plt.subplot(3, 3, 2)
    X = np.linspace(0, 1, 100)
    R = device.mix_resistivity(X, 300*np.ones_like(X))
    ax2.semilogy(X, R)
    ax2.set_xlabel("Crystalline Fraction")
    ax2.set_ylabel("Resistance (Ω)")
    ax2.set_title("R vs X")
    ax2.grid(True, alpha=0.3)
    
    # RESET pulse
    ax3 = plt.subplot(3, 3, 3)
    reset = cycle_results["reset"]
    ax3.plot(reset["t_s"]*1e9, reset["I_A"]*1e3)
    ax3.set_xlabel("Time (ns)")
    ax3.set_ylabel("Current (mA)")
    ax3.set_title("RESET: Current")
    ax3.grid(True, alpha=0.3)
    
    # RESET temperature
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(reset["t_s"]*1e9, reset["T_K"])
    ax4.set_xlabel("Time (ns)")
    ax4.set_ylabel("Temperature (K)")
    ax4.set_title("RESET: Temperature")
    ax4.axhline(900, color='r', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    # RESET phase
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(reset["t_s"]*1e9, reset["X"])
    ax5.set_xlabel("Time (ns)")
    ax5.set_ylabel("Crystalline Fraction")
    ax5.set_title("RESET: Phase Evolution")
    ax5.grid(True, alpha=0.3)
    
    # RESET resistance
    ax6 = plt.subplot(3, 3, 6)
    ax6.semilogy(reset["t_s"]*1e9, reset["R_ohm"])
    ax6.set_xlabel("Time (ns)")
    ax6.set_ylabel("Resistance (Ω)")
    ax6.set_title("RESET: Resistance")
    ax6.grid(True, alpha=0.3)
    
    # I-V characteristics
    ax7 = plt.subplot(3, 3, 7)
    iv_data = ts.sweep_iv(X_state=0.0)
    ax7.plot(iv_data["V_up"], iv_data["I_up"]*1e3, 'b-', label='Up')
    ax7.plot(iv_data["V_down"], iv_data["I_down"]*1e3, 'r-', label='Down')
    ax7.set_xlabel("Voltage (V)")
    ax7.set_ylabel("Current (mA)")
    ax7.set_title("Threshold Switching I-V")
    ax7.set_yscale('log')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Retention
    ax8 = plt.subplot(3, 3, 8)
    valid = retention_data["t50_s"] < 1e15
    if np.any(valid):
        ax8.semilogy(1000*retention_data["inv_T"][valid], 
                    retention_data["t50_s"][valid], 'bo-')
    ax8.set_xlabel("1000/T (1/K)")
    ax8.set_ylabel("t50 (s)")
    ax8.set_title("Retention")
    ax8.grid(True, alpha=0.3)
    
    # Endurance
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(endurance_data["cycles"], endurance_data["cdf"])
    ax9.set_xlabel("Cycles")
    ax9.set_ylabel("CDF")
    ax9.set_title("Endurance (Weibull)")
    ax9.set_xscale('log')
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle("PCM GST Simulation Summary", fontsize=16, y=1.02)
    plt.tight_layout()
    save_figure(fig_summary, "00_summary", "data")
    plt.close(fig_summary)
    
    print("   ✓ Summary figure created")
    
    # ========== Complete ==========
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE!")
    print("=" * 60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved in:")
    print(f"  - Figures: data/*.png")
    print(f"  - Data: data/*.csv")
    print(f"  - Report: reports/simulation_summary.txt")
    
    return results


if __name__ == "__main__":
    results = main()