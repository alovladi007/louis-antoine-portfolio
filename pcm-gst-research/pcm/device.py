"""
PCM Device Model
================
Electro-thermal device simulation with phase change dynamics.
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
            
            # Thermal - tuned for realistic nanosecond heating
            "Cth_J_per_K": 5e-12,    # Small thermal mass for fast response
            "Rth_K_per_W": 2.5e4,    # High thermal resistance
            
            # Electrical
            "series_R_ohm": 100.0,    # Series resistance
            
            # Threshold switching
            "Vth0_V": 1.2,
            "Ihold_A": 50e-6,
            "Ron_ohm": 1e3,
            
            # JMAK parameters
            "k0_cryst": 1e16,
            "Ea_cryst_eV": 1.8,
            "avrami_n": 3.0,
            "Tc_K": 430.0,
            "kb_eV": 8.617333262e-5,
            
            # Pulse parameters
            "reset_V_V": 2.5,
            "set_V_V": 0.9,
            "reset_duration_s": 80e-9,
            "set_duration_s": 200e-9,
            
            # Simulation
            "dt_s": 1e-9
        }
    
    def mix_resistivity(self, X: Union[float, np.ndarray], 
                       T: Union[float, np.ndarray], p: float = 1.0) -> Union[float, np.ndarray]:
        """
        Calculate device resistance from crystalline fraction and temperature.
        
        Uses percolation-based parallel conductivity mixing model.
        """
        rho_a = self.params["rho_amor_ohm_m"]
        rho_c0 = self.params["rho_cryst_ohm_m"]
        alpha = self.params["alpha_TCR"]
        Tamb = self.params["Tamb_K"]
        
        # Temperature coefficient for crystalline
        rho_c = rho_c0 * (1.0 + alpha * (T - Tamb))
        
        # Percolation mixing of conductivities
        sigma = (X**p) / rho_c + ((1 - X)**p) / rho_a
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
                if not ts_on and Vdev[i] >= Vth and X[i-1] < 0.2:
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
            
            # Phase evolution with sticky amorphization
            if melted or T[i] >= Tm:
                # Melt-quench to amorphous
                melted = True
                X[i] = 0.0
            elif T[i] > Tc:
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
    
    def create_reset_pulse(self, duration_s: float = None,
                          voltage_V: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create RESET pulse (melt-quench to amorphous)."""
        if duration_s is None:
            duration_s = self.params["reset_duration_s"]
        if voltage_V is None:
            voltage_V = self.params["reset_V_V"]
            
        dt = self.params["dt_s"]
        t = np.arange(0, duration_s + 80e-9, dt)
        V = np.zeros_like(t)
        pulse_samples = int(duration_s/dt)
        V[:pulse_samples] = voltage_V
        return t, V
    
    def create_set_pulse(self, duration_s: float = None,
                        voltage_V: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create SET pulse (crystallization)."""
        if duration_s is None:
            duration_s = self.params["set_duration_s"]
        if voltage_V is None:
            voltage_V = self.params["set_V_V"]
            
        dt = self.params["dt_s"]
        t = np.arange(0, duration_s + 100e-9, dt)
        V = np.zeros_like(t)
        pulse_samples = int(duration_s/dt)
        V[:pulse_samples] = voltage_V
        return t, V
    
    def simulate_reset_set_cycle(self, X0: float = 1.0) -> Dict:
        """Simulate complete RESET-SET cycle."""
        # RESET pulse
        t_reset, V_reset = self.create_reset_pulse()
        result_reset = self.simulate_voltage_pulse(V_reset, t_reset, X0=X0)
        
        # SET pulse starting from amorphous state
        t_set, V_set = self.create_set_pulse()
        result_set = self.simulate_voltage_pulse(V_set, t_set, 
                                                 X0=result_reset["X"][-1])
        
        return {
            "reset": result_reset,
            "set": result_set,
            "resistance_ratio": result_reset["R_ohm"][-1] / result_set["R_ohm"][-1]
        }