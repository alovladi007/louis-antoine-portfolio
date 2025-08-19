"""
Electro-thermal PCM Device Model
================================
Coupled electrical and thermal dynamics for PCM device simulation.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from .kinetics import JMAKModel


class PCMDevice:
    """Electro-thermal PCM device model."""
    
    def __init__(self, params: Optional[Dict] = None):
        """Initialize device with parameters."""
        if params is None:
            params = self.default_params()
        self.params = params
        self.jmak = JMAKModel(params)
        
    @staticmethod
    def default_params() -> Dict:
        """Default device parameters."""
        return {
            # Material
            "thickness_m": 50e-9,
            "area_m2": (50e-9)**2,
            "rho_amor_ohm_m": 1.0,
            "rho_cryst_ohm_m": 1e-3,
            "alpha_TCR": 0.001,
            "Tm_K": 900.0,
            "Tamb_K": 300.0,
            
            # Thermal
            "Cth_J_per_K": 1e-11,    # Thermal capacitance
            "Rth_K_per_W": 5e5,      # Thermal resistance
            
            # Electrical
            "series_R_ohm": 500.0,    # Series resistance
            
            # Threshold switching
            "Vth0_V": 1.2,
            "Ihold_A": 50e-6,
            "Ron_ohm": 1e3,
            
            # JMAK parameters
            "k0_cryst": 1e13,
            "Ea_cryst_eV": 2.1,
            "avrami_n": 3.0,
            "Tc_K": 430.0,
            "kb_eV": 8.617333262e-5,
            
            # Simulation
            "dt_s": 1e-9
        }
    
    def mix_resistivity(self, X: Union[float, np.ndarray], 
                       T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate device resistance from crystalline fraction and temperature.
        
        Uses percolation-based conductivity mixing model.
        """
        rho_a = self.params["rho_amor_ohm_m"]
        rho_c0 = self.params["rho_cryst_ohm_m"]
        alpha = self.params["alpha_TCR"]
        Tamb = self.params["Tamb_K"]
        
        # Temperature coefficient for crystalline
        rho_c = rho_c0 * (1.0 + alpha * (T - Tamb))
        
        # Percolation-based mixing of conductivities
        # Use parallel model for simplicity
        sigma = X / rho_c + (1 - X) / rho_a
        sigma = np.maximum(sigma, 1e-12)
        
        # Device resistance
        L = self.params["thickness_m"]
        A = self.params["area_m2"]
        R = L / (A * sigma)
        
        return R
    
    def simulate_voltage_pulse(self, voltage_waveform: np.ndarray,
                              t_grid: np.ndarray, X0: float = 0.0,
                              enable_ts: bool = True) -> Dict:
        """
        Simulate voltage pulse with electro-thermal dynamics.
        
        Args:
            voltage_waveform: Applied voltage vs time (V)
            t_grid: Time array (s)
            X0: Initial crystalline fraction
            enable_ts: Enable threshold switching
            
        Returns:
            Dictionary with simulation results
        """
        # Extract parameters
        Cth = self.params["Cth_J_per_K"]
        Rth = self.params["Rth_K_per_W"]
        Rseries = self.params["series_R_ohm"]
        Tamb = self.params["Tamb_K"]
        Tm = self.params["Tm_K"]
        Tc = self.params["Tc_K"]
        
        # Threshold switching params
        Vth0 = self.params["Vth0_V"]
        Ihold = self.params["Ihold_A"]
        Ron = self.params["Ron_ohm"]
        
        # Initialize arrays
        n = len(t_grid)
        T = np.full(n, Tamb)
        X = np.zeros(n)
        X[0] = X0
        I = np.zeros(n)
        Vdev = np.zeros(n)
        P = np.zeros(n)
        R_eff = np.zeros(n)
        
        # State flags
        melted = False
        ts_on = False
        
        for i in range(1, n):
            dt = t_grid[i] - t_grid[i-1]
            
            # Check if we melted in previous step
            if T[i-1] >= Tm:
                melted = True
            
            # Device resistance from phase state
            Rdev_base = self.mix_resistivity(X[i-1], T[i-1])
            
            # Threshold voltage scales with amorphous content
            if enable_ts:
                Vth = Vth0 * (0.5 + 0.5 * (1.0 - X[i-1]))
            
            # Electrical calculation with TS
            Vtot = voltage_waveform[i]
            
            if enable_ts and ts_on:
                R_current = min(Ron, Rdev_base)
            else:
                R_current = Rdev_base
            
            # Calculate current and device voltage
            I[i] = Vtot / (R_current + Rseries)
            Vdev[i] = I[i] * R_current
            
            # Check threshold switching conditions
            if enable_ts:
                if not ts_on and Vdev[i] >= Vth and X[i-1] < 0.5:
                    # Switch ON
                    ts_on = True
                    R_current = Ron
                    I[i] = Vtot / (R_current + Rseries)
                    Vdev[i] = I[i] * R_current
                elif ts_on and I[i] < Ihold:
                    # Switch OFF
                    ts_on = False
                    R_current = Rdev_base
                    I[i] = Vtot / (R_current + Rseries)
                    Vdev[i] = I[i] * R_current
            
            R_eff[i] = R_current
            
            # Power dissipation
            P[i] = I[i]**2 * R_current
            
            # Thermal dynamics
            dT = (P[i] - (T[i-1] - Tamb) / Rth) * dt / Cth
            T[i] = T[i-1] + dT
            
            # Phase evolution
            if melted and T[i] < Tm:
                # Quenched from melt - stay amorphous
                X[i] = 0.0
                melted = False  # Reset melted flag after quenching
            elif T[i] >= Tm:
                # Melting - force amorphous
                X[i] = 0.0
                melted = True
            elif T[i] > Tc and not melted:
                # Crystallization via JMAK
                X[i] = self.jmak.incremental_update(X[i-1], T[i], dt)
            else:
                # No phase change
                X[i] = X[i-1]
        
        # Final device resistance array
        R_final = self.mix_resistivity(X, T)
        
        return {
            "t_s": t_grid,
            "I_A": I,
            "Vdev_V": Vdev,
            "T_K": T,
            "X": X,
            "R_ohm": R_final,
            "R_eff_ohm": R_eff,
            "P_W": P
        }
    
    def create_reset_pulse(self, duration_s: float = 100e-9,
                          voltage_V: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """Create RESET pulse (melt-quench)."""
        dt = self.params["dt_s"]
        t = np.arange(0, duration_s + 100e-9, dt)
        V = np.zeros_like(t)
        V[:int(duration_s/dt)] = voltage_V
        return t, V
    
    def create_set_pulse(self, duration_s: float = 500e-9,
                        voltage_V: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """Create SET pulse (crystallization)."""
        dt = self.params["dt_s"]
        t = np.arange(0, duration_s + 200e-9, dt)
        V = np.zeros_like(t)
        V[:int(duration_s/dt)] = voltage_V
        return t, V
    
    def simulate_current_pulse(self, current_waveform: np.ndarray,
                               t_grid: np.ndarray, X0: float = 0.0) -> Dict:
        """
        Simulate current-driven pulse.
        
        Args:
            current_waveform: Applied current vs time (A)
            t_grid: Time array (s)
            X0: Initial crystalline fraction
            
        Returns:
            Dictionary with simulation results
        """
        # Extract parameters
        Cth = self.params["Cth_J_per_K"]
        Rth = self.params["Rth_K_per_W"]
        Tamb = self.params["Tamb_K"]
        Tm = self.params["Tm_K"]
        Tc = self.params["Tc_K"]
        
        # Initialize arrays
        n = len(t_grid)
        T = np.full(n, Tamb)
        X = np.zeros(n)
        X[0] = X0
        I = current_waveform.copy()
        Vdev = np.zeros(n)
        P = np.zeros(n)
        R = np.zeros(n)
        
        # State flags
        melted = False
        
        for i in range(1, n):
            dt = t_grid[i] - t_grid[i-1]
            
            # Check if we melted
            if T[i-1] >= Tm:
                melted = True
            
            # Device resistance
            R[i] = self.mix_resistivity(X[i-1], T[i-1])
            
            # Voltage and power
            Vdev[i] = I[i] * R[i]
            P[i] = I[i]**2 * R[i]
            
            # Thermal dynamics
            dT = (P[i] - (T[i-1] - Tamb) / Rth) * dt / Cth
            T[i] = T[i-1] + dT
            
            # Phase evolution
            if melted and T[i] < Tm:
                X[i] = 0.0
                melted = False
            elif T[i] >= Tm:
                X[i] = 0.0
                melted = True
            elif T[i] > Tc and not melted:
                X[i] = self.jmak.incremental_update(X[i-1], T[i], dt)
            else:
                X[i] = X[i-1]
        
        # Update final resistance
        R_final = self.mix_resistivity(X, T)
        
        return {
            "t_s": t_grid,
            "I_A": I,
            "Vdev_V": Vdev,
            "T_K": T,
            "X": X,
            "R_ohm": R_final,
            "P_W": P
        }