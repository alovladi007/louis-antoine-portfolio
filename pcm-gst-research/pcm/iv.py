"""
Threshold Switching I-V Module
===============================
I-V characteristics with electronic threshold switching.
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
            "Vth0_V": 1.2,           # Base threshold voltage
            "Ihold_A": 50e-6,        # Hold current
            "Ron_ohm": 1e3,          # ON state resistance
            "Roff_base_ohm": 20e6,   # Base OFF state resistance
            "series_R_ohm": 500.0    # Series resistance
        }
    
    def sweep_iv(self, X_state: float = 0.0, Vmax: float = 3.0,
                 steps: int = 600) -> Dict:
        """
        Simulate quasi-static I-V sweep with hysteresis.
        
        Args:
            X_state: Crystalline fraction (0=amorphous, 1=crystalline)
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
        on = True
        
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
            "Roff": Roff,
            "X_state": X_state,
            "Ihold": Ihold
        }
    
    def plot_iv(self, data: Dict = None, X_state: float = 0.0) -> plt.Figure:
        """Plot I-V characteristics with hysteresis."""
        if data is None:
            data = self.sweep_iv(X_state=X_state)
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot up and down sweeps
        ax.plot(data["V_up"], data["I_up"] * 1e3, 'b-', 
                label="Up-sweep", linewidth=2)
        ax.plot(data["V_down"], data["I_down"] * 1e3, 'r-', 
                label="Down-sweep", linewidth=2)
        
        # Add threshold and hold indicators
        ax.axvline(data["Vth"], color='k', linestyle='--', 
                   alpha=0.5, label=f'Vth={data["Vth"]:.2f}V')
        ax.axhline(data["Ihold"] * 1e3, color='g', linestyle='--', 
                   alpha=0.5, label=f'Ihold={data["Ihold"]*1e6:.1f}μA')
        
        ax.set_xlabel("Voltage (V)", fontsize=12)
        ax.set_ylabel("Current (mA)", fontsize=12)
        ax.set_title(f"Threshold Switching I-V (GST, simplified)\nCrystalline Fraction X={data['X_state']:.2f}", 
                     fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Use log scale for better visualization
        ax.set_yscale('log')
        ax.set_ylim([1e-6, 10])
        
        return fig
    
    def plot_iv_comparison(self, X_values: np.ndarray = None) -> plt.Figure:
        """Plot I-V curves for different crystalline fractions."""
        if X_values is None:
            X_values = np.array([0.0, 0.5, 1.0])
        
        fig, axes = plt.subplots(1, len(X_values), figsize=(15, 5))
        
        for idx, X in enumerate(X_values):
            ax = axes[idx] if len(X_values) > 1 else axes
            data = self.sweep_iv(X_state=X)
            
            ax.plot(data["V_up"], data["I_up"] * 1e3, 'b-', 
                    label="Up-sweep", linewidth=2)
            ax.plot(data["V_down"], data["I_down"] * 1e3, 'r-', 
                    label="Down-sweep", linewidth=2)
            
            ax.set_xlabel("Voltage (V)", fontsize=11)
            ax.set_ylabel("Current (mA)" if idx == 0 else "", fontsize=11)
            ax.set_title(f"X = {X:.1f}", fontsize=12)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.set_ylim([1e-6, 10])
        
        fig.suptitle("Threshold Switching I-V for Different Phase States", 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        
        return fig