"""
Reliability Analysis Module
===========================
Retention and endurance models for PCM devices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from scipy import stats
from .kinetics import JMAKModel


class Reliability:
    """Retention and endurance models for PCM."""
    
    def __init__(self, params: Optional[Dict] = None):
        """Initialize reliability models."""
        if params is None:
            params = self.default_params()
        self.params = params
        self.jmak = JMAKModel(params)
    
    @staticmethod
    def default_params() -> Dict:
        """Default reliability parameters."""
        return {
            # JMAK parameters
            "k0_cryst": 1e16,
            "Ea_cryst_eV": 1.8,
            "avrami_n": 3.0,
            "Tc_K": 430.0,
            "kb_eV": 8.617333262e-5,
            
            # Retention
            "retention_temps_C": [55, 85, 125, 150],
            "target_X": 0.5,
            
            # Endurance Weibull
            "weibull_shape": 3.0,
            "weibull_scale": 1e6
        }
    
    def calculate_t50(self, T_K: float, target_X: float = 0.5) -> float:
        """
        Calculate time to reach target crystalline fraction.
        
        Args:
            T_K: Temperature (K)
            target_X: Target crystalline fraction
            
        Returns:
            Time to reach target_X (s)
        """
        if T_K < self.params["Tc_K"]:
            return np.inf
        
        k = self.jmak.rate_constant(T_K)
        n = self.params["avrami_n"]
        
        if k <= 0 or target_X <= 0 or target_X >= 1:
            return np.inf
        
        # Solve: target_X = 1 - exp(-(k*t)^n)
        # t = ((-ln(1-target_X))^(1/n)) / k
        t = ((-np.log(1 - target_X))**(1/n)) / k
        
        return t
    
    def retention_arrhenius(self, temps_C: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate retention times vs temperature.
        
        Args:
            temps_C: Temperatures in Celsius
            
        Returns:
            Dictionary with retention data
        """
        if temps_C is None:
            temps_C = np.array(self.params["retention_temps_C"])
        
        temps_K = temps_C + 273.15
        target_X = self.params["target_X"]
        
        t50_vals = np.array([self.calculate_t50(T, target_X) for T in temps_K])
        
        # Convert to years for readability
        t50_years = t50_vals / (365.25 * 24 * 3600)
        
        return {
            "temp_C": temps_C,
            "temp_K": temps_K,
            "t50_s": t50_vals,
            "t50_years": t50_years,
            "inv_T": 1.0 / temps_K
        }
    
    def plot_retention(self, data: Dict = None) -> plt.Figure:
        """Plot Arrhenius retention curve."""
        if data is None:
            data = self.retention_arrhenius()
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        valid = data["t50_s"] < 1e15
        if np.any(valid):
            ax.semilogy(1000 * data["inv_T"][valid], data["t50_s"][valid], 
                       'bo-', markersize=8, linewidth=2)
            
            # Fit line for extrapolation
            if np.sum(valid) > 1:
                log_t50 = np.log10(data["t50_s"][valid])
                coeffs = np.polyfit(data["inv_T"][valid], log_t50, 1)
                fit_line = np.poly1d(coeffs)
                x_fit = np.linspace(data["inv_T"][valid].min(),
                                  data["inv_T"][valid].max(), 100)
                ax.semilogy(1000 * x_fit, 10**fit_line(x_fit), 
                           'r--', alpha=0.5, label='Arrhenius fit')
                
                # Calculate activation energy from slope
                Ea_fit = -coeffs[0] * self.params["kb_eV"] * np.log(10)
                ax.text(0.95, 0.95, f'Ea = {Ea_fit:.2f} eV', 
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel("1000/T (1/K)", fontsize=12)
        ax.set_ylabel("t₅₀ (s)", fontsize=12)
        ax.set_title("Retention: Arrhenius Plot\n(Time to 50% crystallization)", fontsize=14)
        ax.grid(True, alpha=0.3, which='both')
        
        # Add temperature labels
        for T_C, inv_T, t50 in zip(data["temp_C"], data["inv_T"], data["t50_s"]):
            if t50 < 1e15:
                ax.annotate(f"{T_C}°C", (1000*inv_T, t50), 
                          textcoords="offset points", xytext=(0,10),
                          ha='center', fontsize=10)
        
        # Add 10-year line
        ten_years_s = 10 * 365.25 * 24 * 3600
        ax.axhline(ten_years_s, color='g', linestyle='--', alpha=0.5, 
                   label='10-year target')
        
        ax.legend(loc='best')
        
        return fig
    
    def endurance_weibull(self, n_samples: int = 5000) -> Dict:
        """
        Generate Weibull distribution for endurance.
        
        Args:
            n_samples: Number of devices to simulate
            
        Returns:
            Dictionary with endurance data
        """
        beta = self.params["weibull_shape"]
        eta = self.params["weibull_scale"]
        
        # Generate random lifetimes
        rng = np.random.default_rng(42)
        u = rng.random(n_samples)
        lifetimes = eta * (-np.log(1 - u))**(1.0/beta)
        
        # Calculate CDF
        x = np.linspace(0, np.percentile(lifetimes, 99.9), 1000)
        cdf = 1 - np.exp(-(x/eta)**beta)
        
        # Calculate percentiles
        percentiles = [1, 10, 50, 90, 99]
        percentile_values = np.percentile(lifetimes, percentiles)
        
        return {
            "lifetimes": lifetimes,
            "cycles": x,
            "cdf": cdf,
            "beta": beta,
            "eta": eta,
            "percentiles": dict(zip(percentiles, percentile_values))
        }
    
    def plot_endurance(self, data: Dict = None) -> plt.Figure:
        """Plot Weibull endurance CDF."""
        if data is None:
            data = self.endurance_weibull()
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # CDF plot
        ax1.plot(data["cycles"], data["cdf"], 'b-', linewidth=2)
        
        # Add characteristic life marker
        ax1.axvline(data["eta"], color='r', linestyle='--', alpha=0.5)
        ax1.text(data["eta"], 0.63, f'η={data["eta"]:.0e} cycles',
                rotation=90, ha='right', va='bottom')
        
        # Add percentile markers
        for p, val in data["percentiles"].items():
            ax1.axvline(val, color='g', linestyle=':', alpha=0.3)
            ax1.text(val, 0.05, f'{p}%', rotation=90, fontsize=8)
        
        ax1.set_xlabel("Cycles", fontsize=12)
        ax1.set_ylabel("CDF", fontsize=12)
        ax1.set_title(f"Endurance: Weibull CDF (simulated devices)\nβ={data['beta']:.1f}, η={data['eta']:.0e}", 
                      fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_xlim([1e4, 2e6])
        
        # Histogram
        ax2.hist(data["lifetimes"], bins=50, density=True, alpha=0.7, 
                color='blue', edgecolor='black')
        
        # Overlay theoretical PDF
        x_pdf = np.linspace(0, np.max(data["lifetimes"]), 1000)
        pdf = (data["beta"]/data["eta"]) * (x_pdf/data["eta"])**(data["beta"]-1) * \
              np.exp(-(x_pdf/data["eta"])**data["beta"])
        ax2.plot(x_pdf, pdf, 'r-', linewidth=2, label='Theoretical PDF')
        
        ax2.set_xlabel("Cycles", fontsize=12)
        ax2.set_ylabel("Probability Density", fontsize=12)
        ax2.set_title("Endurance Distribution", fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def monte_carlo_retention(self, n_devices: int = 1000, 
                            T_C: float = 85.0) -> Dict:
        """
        Monte Carlo simulation of retention with variability.
        
        Args:
            n_devices: Number of devices to simulate
            T_C: Temperature in Celsius
            
        Returns:
            Dictionary with Monte Carlo results
        """
        T_K = T_C + 273.15
        rng = np.random.default_rng(42)
        
        # Add variability to activation energy (±10%)
        Ea_mean = self.params["Ea_cryst_eV"]
        Ea_std = 0.1 * Ea_mean
        Ea_values = rng.normal(Ea_mean, Ea_std, n_devices)
        
        # Calculate t50 for each device
        t50_values = []
        for Ea in Ea_values:
            # Temporarily update params
            old_Ea = self.params["Ea_cryst_eV"]
            self.params["Ea_cryst_eV"] = Ea
            t50 = self.calculate_t50(T_K, self.params["target_X"])
            t50_values.append(t50)
            self.params["Ea_cryst_eV"] = old_Ea
        
        t50_values = np.array(t50_values)
        t50_years = t50_values / (365.25 * 24 * 3600)
        
        return {
            "T_C": T_C,
            "Ea_values": Ea_values,
            "t50_values": t50_values,
            "t50_years": t50_years,
            "mean_t50_years": np.mean(t50_years),
            "std_t50_years": np.std(t50_years),
            "min_t50_years": np.min(t50_years),
            "max_t50_years": np.max(t50_years)
        }