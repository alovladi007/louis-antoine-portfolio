"""
I-V Characteristics and Threshold Switching Module
==================================================
Quasi-static I-V sweep with threshold switching hysteresis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional


class ThresholdSwitching:
    """Threshold switching I-V characteristics."""
    
    def __init__(self, params: Optional[Dict] = None):
        """Initialize with device parameters."""
        if params is None:
            params = self.default_params()
        self.params = params
    
    @staticmethod
    def default_params() -> Dict:
        """Default threshold switching parameters."""
        return {
            "Vth0_V": 1.2,
            "Ihold_A": 50e-6,
            "Ron_ohm": 1e3,
            "Roff_base_ohm": 20e6,
            "series_R_ohm": 500.0
        }
    
    def sweep_iv(self, X_state: float = 0.0, Vmax: float = 3.0,
                 steps: int = 600) -> Dict:
        """
        Simulate quasi-static I-V sweep with hysteresis.
        
        Args:
            X_state: Crystalline fraction
            Vmax: Maximum voltage
            steps: Number of voltage steps
            
        Returns:
            Dictionary with I-V data
        """
        # Parameters
        Vth0 = self.params["Vth0_V"]
        Ihold = self.params["Ihold_A"]
        Ron = self.params["Ron_ohm"]
        Rseries = self.params["series_R_ohm"]
        Roff_base = self.params["Roff_base_ohm"]
        
        # Adjust parameters based on crystalline state
        Vth = Vth0 * (0.5 + 0.5 * (1.0 - X_state))
        Roff = Roff_base * (1.0 - 0.9 * X_state)  # Lower Roff for crystalline
        
        # Up-sweep
        V_up = np.linspace(0, Vmax, steps)
        I_up = np.zeros_like(V_up)
        on = False
        
        for i, v in enumerate(V_up):
            if not on:
                i_trial = v / (Roff + Rseries)
                v_device = i_trial * Roff
                if v_device >= Vth:
                    on = True
                    I_up[i] = v / (Ron + Rseries)
                else:
                    I_up[i] = i_trial
            else:
                I_up[i] = v / (Ron + Rseries)
        
        # Down-sweep
        V_down = np.linspace(Vmax, 0, steps)
        I_down = np.zeros_like(V_down)
        on = True if I_up[-1] > Ihold else False
        
        for i, v in enumerate(V_down):
            if on:
                i_now = v / (Ron + Rseries)
                if i_now < Ihold:
                    on = False
                    I_down[i] = v / (Roff + Rseries)
                else:
                    I_down[i] = i_now
            else:
                I_down[i] = v / (Roff + Rseries)
        
        return {
            "V_up": V_up,
            "I_up": I_up,
            "V_down": V_down,
            "I_down": I_down,
            "Vth": Vth,
            "Ihold": Ihold,
            "Ron": Ron,
            "Roff": Roff,
            "X_state": X_state
        }
    
    def plot_iv(self, data: Dict) -> plt.Figure:
        """Plot I-V characteristics with hysteresis."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(data["V_up"], data["I_up"] * 1e3, 'b-', 
                label="Up-sweep", linewidth=2)
        ax.plot(data["V_down"], data["I_down"] * 1e3, 'r-', 
                label="Down-sweep", linewidth=2)
        
        ax.set_xlabel("Voltage (V)", fontsize=12)
        ax.set_ylabel("Current (mA)", fontsize=12)
        ax.set_title(f"Threshold Switching I-V (X={data['X_state']:.2f})", 
                     fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Annotate threshold
        ax.axvline(data["Vth"], color='k', linestyle='--', alpha=0.5)
        ax.text(data["Vth"], ax.get_ylim()[1]*0.9, 
                f"Vth={data['Vth']:.2f}V",
                ha='center', fontsize=10)
        
        # Annotate hold current
        ax.axhline(data["Ihold"] * 1e3, color='k', linestyle='--', alpha=0.5)
        ax.text(ax.get_xlim()[1]*0.9, data["Ihold"] * 1e3,
                f"Ihold={data['Ihold']*1e6:.0f}µA",
                va='bottom', ha='right', fontsize=10)
        
        return fig
    
    def sweep_multiple_states(self, X_states: np.ndarray = None,
                             Vmax: float = 3.0) -> Dict:
        """
        Sweep I-V for multiple crystalline states.
        
        Args:
            X_states: Array of crystalline fractions
            Vmax: Maximum voltage
            
        Returns:
            Dictionary with all sweep data
        """
        if X_states is None:
            X_states = np.array([0.0, 0.3, 0.7, 1.0])
        
        results = {}
        for X in X_states:
            results[f"X_{X:.1f}"] = self.sweep_iv(X, Vmax)
        
        return results
    
    def plot_multiple_iv(self, results: Dict) -> plt.Figure:
        """Plot multiple I-V curves on same axes."""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        
        for (key, data), color in zip(results.items(), colors):
            X = data["X_state"]
            ax.plot(data["V_up"], data["I_up"] * 1e3,
                   color=color, linestyle='-', alpha=0.7,
                   label=f"X={X:.1f} (up)")
            ax.plot(data["V_down"], data["I_down"] * 1e3,
                   color=color, linestyle='--', alpha=0.7,
                   label=f"X={X:.1f} (down)")
        
        ax.set_xlabel("Voltage (V)", fontsize=12)
        ax.set_ylabel("Current (mA)", fontsize=12)
        ax.set_title("Threshold Switching I-V for Different States", fontsize=14)
        ax.legend(loc='best', ncol=2, fontsize=9)
        ax.grid(True, alpha=0.3)
        
        return fig