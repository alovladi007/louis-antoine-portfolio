"""
Interleaved Boost Converter Simulation Kernel
Implements multi-phase boost converter with current ripple cancellation
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Tuple, Optional
import numba as nb


class InterleavedBoostSimulator:
    """
    Interleaved boost converter simulator with phase shedding capability
    
    State equations for N-phase interleaved boost:
    For each phase k:
        L * di_k/dt = v_in - v_out * (1 - d_k) - i_k * R_L
        
    Output capacitor:
        C * dv_out/dt = sum(i_k * (1 - d_k)) - v_out / R_load
    
    Where d_k is the duty cycle for phase k, phase-shifted by 360°/N
    """
    
    def __init__(
        self,
        vin: float,
        vout_ref: float,
        power: float,
        phases: int = 3,
        fsw: float = 100e3,
        L: float = 100e-6,
        C: float = 1000e-6,
        rload: Optional[float] = None,
        rl: float = 0.01,  # Inductor ESR
        rc: float = 0.001   # Capacitor ESR
    ):
        self.vin = vin
        self.vout_ref = vout_ref
        self.power = power
        self.phases = phases
        self.fsw = fsw
        self.L = L
        self.C = C
        self.rload = rload if rload else vout_ref**2 / power
        self.rl = rl
        self.rc = rc
        
        # Control parameters
        self.kp = 0.1
        self.ki = 100
        self.current_limit = power / vin * 1.5  # 150% overcurrent
        
        # Phase shift for interleaving
        self.phase_shift = 2 * np.pi / phases
        
        # State variables: [i_L1, i_L2, ..., i_Ln, v_C, integral_error]
        self.state_dim = phases + 2
        
    def set_controller(self, kp: float, ki: float, current_limit: float):
        """Configure PI controller parameters"""
        self.kp = kp
        self.ki = ki
        self.current_limit = current_limit
        
    @nb.jit(nopython=True)
    def calculate_duty_cycles(
        self,
        t: float,
        vout: float,
        vout_ref: float,
        integral_error: float,
        phases: int,
        fsw: float,
        kp: float,
        ki: float
    ) -> np.ndarray:
        """
        Calculate duty cycles for each phase with PI control
        Implements phase shifting for ripple cancellation
        """
        # PI controller
        error = vout_ref - vout
        duty_base = kp * error + ki * integral_error
        
        # Clamp duty cycle
        duty_base = np.clip(duty_base, 0.1, 0.9)
        
        # Generate phase-shifted PWM carriers
        duties = np.zeros(phases)
        carrier_period = 1.0 / fsw
        
        for k in range(phases):
            # Phase shift for each converter
            phase_offset = k * 2 * np.pi / phases
            carrier = (t % carrier_period) * fsw
            
            # Triangle carrier with phase shift
            carrier_shifted = (carrier + k / phases) % 1.0
            
            # PWM comparator
            if carrier_shifted < duty_base:
                duties[k] = 1.0
            else:
                duties[k] = 0.0
                
        return duties
    
    def state_equations(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Differential equations for the interleaved boost converter
        
        State vector x = [i_L1, i_L2, ..., i_Ln, v_C, integral_error]
        """
        # Extract states
        iL = x[:self.phases]
        vC = x[self.phases]
        integral_error = x[self.phases + 1]
        
        # Calculate duty cycles
        duties = self.calculate_duty_cycles(
            t, vC, self.vout_ref, integral_error,
            self.phases, self.fsw, self.kp, self.ki
        )
        
        # State derivatives
        dx = np.zeros_like(x)
        
        # Inductor currents
        for k in range(self.phases):
            # di_Lk/dt = (v_in - v_out*(1-d_k) - i_Lk*R_L) / L
            dx[k] = (self.vin - vC * (1 - duties[k]) - iL[k] * self.rl) / self.L
            
            # Current limiting
            if iL[k] > self.current_limit:
                dx[k] = -iL[k] / (self.L / self.rl)  # Fast decay
        
        # Output capacitor voltage
        # dv_C/dt = (sum(i_Lk*(1-d_k)) - v_C/R_load) / C
        iout = np.sum(iL * (1 - duties))
        dx[self.phases] = (iout - vC / self.rload) / self.C
        
        # Integral of voltage error for PI control
        dx[self.phases + 1] = self.vout_ref - vC
        
        return dx
    
    def calculate_ripple_current(self, vin: float, vout: float, L: float, fsw: float, D: float) -> float:
        """
        Calculate peak-to-peak inductor current ripple
        ΔI_L = (V_in * D * T_s) / L = (V_in * D) / (L * f_sw)
        """
        return (vin * D * (1 - D)) / (L * fsw)
    
    def calculate_phase_shedding_threshold(self, power: float, efficiency: float = 0.95) -> float:
        """
        Determine power threshold for phase shedding to improve light-load efficiency
        Typically shed phases below 30% load per phase
        """
        power_per_phase = power / self.phases
        return 0.3 * power_per_phase * efficiency
    
    async def simulate(self, timespan: float, dt: float) -> Dict[str, np.ndarray]:
        """Run time-domain simulation"""
        # Initial conditions
        x0 = np.zeros(self.state_dim)
        x0[:self.phases] = self.power / (self.vin * self.phases)  # Initial inductor currents
        x0[self.phases] = self.vout_ref  # Initial output voltage
        
        # Time vector
        t_eval = np.arange(0, timespan, dt)
        
        # Solve ODEs
        sol = solve_ivp(
            self.state_equations,
            [0, timespan],
            x0,
            t_eval=t_eval,
            method='DOP853',  # High-order Runge-Kutta
            rtol=1e-6,
            atol=1e-9
        )
        
        # Extract results
        iL_phases = sol.y[:self.phases, :].T
        vout = sol.y[self.phases, :]
        
        # Calculate additional metrics
        time = sol.t
        iin_total = np.sum(iL_phases, axis=1)
        
        # Calculate duty cycles at each time point
        duty_cycles = np.zeros((len(time), self.phases))
        for i, t in enumerate(time):
            duty_cycles[i] = self.calculate_duty_cycles(
                t, vout[i], self.vout_ref, sol.y[self.phases + 1, i],
                self.phases, self.fsw, self.kp, self.ki
            )
        
        # Power calculations
        pin = self.vin * iin_total
        pout = vout**2 / self.rload
        efficiency = pout / (pin + 1e-10)  # Avoid division by zero
        
        # Ripple calculations
        ripple_current = np.zeros(self.phases)
        for k in range(self.phases):
            ripple_current[k] = self.calculate_ripple_current(
                self.vin, vout.mean(), self.L, self.fsw, duty_cycles[:, k].mean()
            )
        
        return {
            "time": time,
            "i_phases": iL_phases,
            "i_total": iin_total,
            "v_out": vout,
            "v_dc": np.full_like(time, self.vin),
            "duty_cycles": duty_cycles,
            "efficiency": efficiency,
            "i_ripple": ripple_current,
            "power_in": pin,
            "power_out": pout
        }
    
    def calculate_thd(self, signal: np.ndarray, fundamental_freq: float = 50.0) -> float:
        """
        Calculate Total Harmonic Distortion
        THD = sqrt(sum(V_n^2 for n>1)) / V_1
        """
        # FFT of the signal
        fft = np.fft.rfft(signal)
        magnitudes = np.abs(fft)
        
        # Find fundamental frequency index
        freq_resolution = 1.0 / (len(signal) * (1.0 / self.fsw))
        fundamental_idx = int(fundamental_freq / freq_resolution)
        
        if fundamental_idx >= len(magnitudes):
            return 0.0
        
        # Calculate THD
        fundamental = magnitudes[fundamental_idx]
        harmonics = magnitudes[fundamental_idx + 1:]
        
        if fundamental > 0:
            thd = np.sqrt(np.sum(harmonics**2)) / fundamental
            return min(thd, 1.0)  # Cap at 100%
        
        return 0.0
    
    def design_inductor(self, ripple_factor: float = 0.3) -> Dict[str, float]:
        """
        Design inductor for specified current ripple
        L = (V_in * D * (1-D)) / (ΔI * f_sw)
        where ΔI = ripple_factor * I_avg
        """
        iavg = self.power / self.vin
        delta_i = ripple_factor * iavg / self.phases
        
        # Worst case duty cycle for ripple (D = 0.5)
        D = 0.5
        L_required = (self.vin * D * (1 - D)) / (delta_i * self.fsw)
        
        # With interleaving, effective ripple frequency is N * fsw
        L_effective = L_required / self.phases
        
        return {
            "inductance": L_required,
            "peak_current": iavg / self.phases + delta_i / 2,
            "rms_current": iavg / self.phases,
            "ripple_current": delta_i,
            "effective_frequency": self.phases * self.fsw
        }