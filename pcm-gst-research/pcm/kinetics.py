"""
JMAK Crystallization Kinetics Module
====================================
Johnson-Mehl-Avrami-Kolmogorov model for phase transformation kinetics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Tuple, Dict, Optional
import json


class JMAKModel:
    """Johnson-Mehl-Avrami-Kolmogorov crystallization kinetics model."""
    
    def __init__(self, params: Optional[Dict] = None):
        """Initialize JMAK model with parameters."""
        if params is None:
            params = self.default_params()
        self.params = params
        
    @staticmethod
    def default_params() -> Dict:
        """Default GST-like kinetic parameters."""
        return {
            "k0_cryst": 1e13,        # Prefactor (1/s)
            "Ea_cryst_eV": 2.1,      # Activation energy (eV)
            "avrami_n": 3.0,         # Avrami exponent
            "Tc_K": 430.0,           # Crystallization onset (K)
            "kb_eV": 8.617333262e-5  # Boltzmann constant (eV/K)
        }
    
    def rate_constant(self, T: float) -> float:
        """Calculate crystallization rate constant at temperature T."""
        if T < self.params["Tc_K"]:
            return 0.0
        k0 = self.params["k0_cryst"]
        Ea = self.params["Ea_cryst_eV"]
        kb = self.params["kb_eV"]
        return k0 * np.exp(-Ea / (kb * T))
    
    def simulate_isothermal(self, T: float, t_array: np.ndarray, 
                           X0: float = 0.0) -> np.ndarray:
        """
        Simulate isothermal crystallization at temperature T.
        
        Args:
            T: Temperature (K)
            t_array: Time array (s)
            X0: Initial crystalline fraction
            
        Returns:
            Array of crystalline fractions
        """
        if T < self.params["Tc_K"]:
            return np.full_like(t_array, X0)
        
        k = self.rate_constant(T)
        n = self.params["avrami_n"]
        
        # JMAK equation: X = 1 - exp(-(kt)^n)
        # Starting from X0: X = 1 - (1-X0)*exp(-(kt)^n)
        X = 1.0 - (1.0 - X0) * np.exp(-(k * t_array)**n)
        return np.clip(X, 0.0, 1.0)
    
    def incremental_update(self, X_old: float, T: float, dt: float) -> float:
        """
        Update crystalline fraction incrementally.
        
        Args:
            X_old: Previous crystalline fraction
            T: Temperature (K)
            dt: Time step (s)
            
        Returns:
            Updated crystalline fraction
        """
        if T < self.params["Tc_K"] or dt <= 0:
            return X_old
        
        k = self.rate_constant(T)
        n = self.params["avrami_n"]
        
        # Incremental JMAK: X_new = 1 - (1-X_old) * exp(-(k*dt)^n)
        inc = 1.0 - np.exp(-(k * dt)**n)
        X_new = X_old + (1.0 - X_old) * inc
        return float(np.clip(X_new, 0.0, 1.0))
    
    def plot_kinetics(self, temps_K: np.ndarray, 
                      t_array: np.ndarray) -> Tuple[plt.Figure, pd.DataFrame]:
        """Plot crystallization kinetics at multiple temperatures."""
        fig, ax = plt.subplots(figsize=(8, 6))
        data_list = []
        
        for T in temps_K:
            X = self.simulate_isothermal(T, t_array)
            ax.semilogx(t_array, X, label=f"{T:.0f} K", linewidth=2)
            
            # Store data
            for t, x in zip(t_array, X):
                data_list.append({"T_K": T, "t_s": t, "X": x})
        
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Crystalline fraction X", fontsize=12)
        ax.set_title("JMAK Crystallization Kinetics (GST)", fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t_array.min(), t_array.max())
        ax.set_ylim(0, 1)
        
        df = pd.DataFrame(data_list)
        return fig, df