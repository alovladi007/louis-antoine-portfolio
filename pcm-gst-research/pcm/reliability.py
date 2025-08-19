"""
Reliability Models for PCM
==========================
Retention (Arrhenius) and endurance (Weibull) models.
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
            "k0_cryst": 1e13,
            "Ea_cryst_eV": 2.1,
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
        
        return {
            "temp_C": temps_C,
            "temp_K": temps_K,
            "t50_s": t50_vals,
            "inv_T": 1.0 / temps_K
        }
    
    def plot_retention(self, data: Dict) -> plt.Figure:
        """Plot Arrhenius retention curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        valid = data["t50_s"] < 1e10
        if np.any(valid):
            ax.semilogy(data["inv_T"][valid] * 1000, data["t50_s"][valid], 
                       'bo-', markersize=8, linewidth=2)
            
            # Fit line for extrapolation
            if np.sum(valid) > 1:
                log_t50 = np.log10(data["t50_s"][valid])
                coeffs = np.polyfit(data["inv_T"][valid] * 1000, log_t50, 1)
                fit_line = np.poly1d(coeffs)
                x_fit = np.linspace(data["inv_T"][valid].min() * 1000,
                                  data["inv_T"][valid].max() * 1000, 100)
                ax.semilogy(x_fit, 10**fit_line(x_fit), 'r--', 
                           alpha=0.5, label='Arrhenius fit')
        
        ax.set_xlabel("1000/T (1/K)", fontsize=12)
        ax.set_ylabel("t₅₀ (s)", fontsize=12)
        ax.set_title("Retention: Arrhenius Plot", fontsize=14)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend()
        
        # Add temperature labels
        for T_C, inv_T, t50 in zip(data["temp_C"], data["inv_T"], data["t50_s"]):
            if t50 < 1e10:
                ax.annotate(f"{T_C}°C", (inv_T * 1000, t50), 
                          textcoords="offset points", xytext=(5, 5),
                          fontsize=10)
        
        # Add 10-year line
        t_10yr = 10 * 365.25 * 24 * 3600
        ax.axhline(t_10yr, color='g', linestyle=':', alpha=0.5)
        ax.text(ax.get_xlim()[0], t_10yr, '10 years', 
                va='bottom', fontsize=10, color='g')
        
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
        x = np.logspace(np.log10(lifetimes.min()), 
                        np.log10(lifetimes.max()), 400)
        cdf = 1 - np.exp(-(x/eta)**beta)
        
        # Calculate percentiles
        percentiles = [1, 10, 50, 63.2, 90, 99]
        percentile_values = {}
        for p in percentiles:
            idx = np.argmin(np.abs(cdf - p/100))
            percentile_values[f"P{p}"] = x[idx]
        
        return {
            "lifetimes": lifetimes,
            "cycles": x,
            "cdf": cdf,
            "beta": beta,
            "eta": eta,
            "percentiles": percentile_values
        }
    
    def plot_endurance(self, data: Dict) -> plt.Figure:
        """Plot Weibull endurance CDF."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # CDF plot
        ax1.semilogx(data["cycles"], data["cdf"] * 100, 'b-', linewidth=2)
        
        # Add characteristic life marker
        ax1.axvline(data["eta"], color='r', linestyle='--', alpha=0.5)
        ax1.text(data["eta"], 63.2, f"η={data['eta']:.0e} cycles",
                rotation=90, ha='right', fontsize=10)
        
        # Add percentile markers
        for p_name, p_val in data["percentiles"].items():
            p_num = float(p_name[1:])
            if p_num in [10, 50, 90]:
                ax1.axvline(p_val, color='gray', linestyle=':', alpha=0.3)
                ax1.text(p_val, 5, f"{p_num}%", rotation=90, 
                        ha='center', fontsize=9)
        
        ax1.set_xlabel("Cycles", fontsize=12)
        ax1.set_ylabel("CDF (%)", fontsize=12)
        ax1.set_title(f"Endurance: Weibull CDF (β={data['beta']:.1f})", fontsize=14)
        ax1.grid(True, alpha=0.3, which='both')
        ax1.set_ylim(0, 100)
        
        # Weibull plot (linearized)
        ax2.loglog(data["cycles"], -np.log(1 - data["cdf"]), 'b-', linewidth=2)
        ax2.set_xlabel("Cycles", fontsize=12)
        ax2.set_ylabel("ln(ln(1/(1-F)))", fontsize=12)
        ax2.set_title("Weibull Plot (Linearized)", fontsize=14)
        ax2.grid(True, alpha=0.3, which='both')
        
        return fig
    
    def monte_carlo_retention(self, T_K: float, n_samples: int = 1000,
                             variation: float = 0.1) -> Dict:
        """
        Monte Carlo simulation of retention with parameter variations.
        
        Args:
            T_K: Temperature (K)
            n_samples: Number of samples
            variation: Fractional variation in parameters
            
        Returns:
            Dictionary with Monte Carlo results
        """
        rng = np.random.default_rng(42)
        
        # Generate parameter variations
        Ea_samples = self.params["Ea_cryst_eV"] * (1 + variation * rng.normal(0, 1, n_samples))
        k0_samples = self.params["k0_cryst"] * (1 + variation * rng.normal(0, 1, n_samples))
        
        t50_samples = []
        for Ea, k0 in zip(Ea_samples, k0_samples):
            if T_K < self.params["Tc_K"]:
                t50_samples.append(np.inf)
            else:
                k = k0 * np.exp(-Ea / (self.params["kb_eV"] * T_K))
                n = self.params["avrami_n"]
                target_X = self.params["target_X"]
                if k > 0:
                    t = ((-np.log(1 - target_X))**(1/n)) / k
                    t50_samples.append(t)
                else:
                    t50_samples.append(np.inf)
        
        t50_samples = np.array(t50_samples)
        valid = t50_samples < 1e10
        
        return {
            "T_K": T_K,
            "t50_samples": t50_samples[valid],
            "mean": np.mean(t50_samples[valid]) if np.any(valid) else np.inf,
            "std": np.std(t50_samples[valid]) if np.any(valid) else 0,
            "median": np.median(t50_samples[valid]) if np.any(valid) else np.inf
        }