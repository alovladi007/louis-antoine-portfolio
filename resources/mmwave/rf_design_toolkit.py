#!/usr/bin/env python3
"""
mmWave RF Frontend Design Toolkit
Author: Louis Antoine
Date: December 2024
Description: Complete Python toolkit for 28GHz RF frontend design and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize
from scipy.constants import c, pi
import pandas as pd

class RFDesignToolkit:
    """Main toolkit for RF frontend design calculations"""
    
    def __init__(self, frequency=28e9):
        self.freq = frequency
        self.wavelength = c / frequency
        self.Z0 = 50  # Standard impedance
        
    def calculate_path_loss(self, distance, n=2):
        """
        Calculate free space path loss
        Args:
            distance: Distance in meters
            n: Path loss exponent (2 for free space)
        Returns:
            Path loss in dB
        """
        path_loss = 20 * np.log10(4 * pi * distance * self.freq / c)
        return path_loss
    
    def design_microstrip(self, er=3.5, h=0.5e-3, target_z0=50):
        """
        Design microstrip transmission line
        Args:
            er: Relative permittivity
            h: Substrate height in meters
            target_z0: Target characteristic impedance
        Returns:
            Width of microstrip line
        """
        # Initial guess
        w = h
        
        def impedance_error(w):
            w_h = w / h
            if w_h <= 1:
                er_eff = (er + 1)/2 + (er - 1)/2 * (1/np.sqrt(1 + 12/w_h) + 0.04*(1 - w_h)**2)
                Z0 = 60/np.sqrt(er_eff) * np.log(8/w_h + w_h/4)
            else:
                er_eff = (er + 1)/2 + (er - 1)/2 / np.sqrt(1 + 12/w_h)
                Z0 = 120*pi / (np.sqrt(er_eff) * (w_h + 1.393 + 0.667*np.log(w_h + 1.444)))
            return abs(Z0 - target_z0)
        
        result = optimize.minimize_scalar(impedance_error, bounds=(0.1*h, 10*h), method='bounded')
        return result.x
    
    def calculate_noise_figure(self, stages):
        """
        Calculate cascaded noise figure (Friis formula)
        Args:
            stages: List of tuples (gain_dB, nf_dB) for each stage
        Returns:
            Total noise figure in dB
        """
        total_nf_linear = 10**(stages[0][1]/10)
        cumulative_gain = 10**(stages[0][0]/10)
        
        for i in range(1, len(stages)):
            gain = 10**(stages[i][0]/10)
            nf = 10**(stages[i][1]/10)
            total_nf_linear += (nf - 1) / cumulative_gain
            cumulative_gain *= gain
        
        return 10 * np.log10(total_nf_linear)
    
    def smith_chart_plot(self, z_load):
        """
        Plot impedance on Smith chart
        Args:
            z_load: Complex load impedance
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw Smith chart circles
        theta = np.linspace(0, 2*pi, 100)
        
        # Constant resistance circles
        for r in [0, 0.2, 0.5, 1, 2, 5]:
            if r == 0:
                ax.plot([-1, 1], [0, 0], 'k-', linewidth=0.5)
            else:
                center = r / (r + 1)
                radius = 1 / (r + 1)
                x = center + radius * np.cos(theta)
                y = radius * np.sin(theta)
                ax.plot(x, y, 'k-', linewidth=0.5)
        
        # Constant reactance circles
        for x in [-2, -1, -0.5, 0.5, 1, 2]:
            center_y = 1/x
            radius = abs(1/x)
            y = center_y + radius * np.cos(theta)
            x_circle = radius * np.sin(theta)
            ax.plot(x_circle, y, 'k-', linewidth=0.5)
        
        # Plot load impedance
        gamma = (z_load - self.Z0) / (z_load + self.Z0)
        ax.plot(gamma.real, gamma.imag, 'ro', markersize=10, label=f'Z_L = {z_load:.1f}Ω')
        
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Smith Chart - Impedance Plot')
        ax.legend()
        
        return fig

class PowerAmplifierDesign:
    """GaN Power Amplifier Design Tools"""
    
    def __init__(self, vds=28, freq=28e9):
        self.vds = vds
        self.freq = freq
        self.vknee = 3  # Knee voltage for GaN
        
    def calculate_load_impedance(self, pout_dbm):
        """
        Calculate optimal load impedance for given output power
        """
        pout_w = 10**(pout_dbm/10) / 1000
        r_opt = (self.vds - self.vknee)**2 / (2 * pout_w)
        return r_opt
    
    def load_pull_simulation(self, center_z, vswr_max=3):
        """
        Simulate load-pull contours
        Returns power and efficiency contours
        """
        # Create impedance grid
        gamma_mag = np.linspace(0, (vswr_max-1)/(vswr_max+1), 20)
        gamma_phase = np.linspace(0, 2*pi, 36)
        
        power_contours = []
        eff_contours = []
        
        for mag in gamma_mag:
            for phase in gamma_phase:
                gamma = mag * np.exp(1j * phase)
                z_load = 50 * (1 + gamma) / (1 - gamma)
                
                # Simple PA model
                rl = z_load.real
                pout = (self.vds - self.vknee)**2 / (2 * rl) if rl > 0 else 0
                pout_dbm = 10 * np.log10(pout * 1000) if pout > 0 else -30
                
                # Efficiency model
                idc = (self.vds - self.vknee) / (2 * rl) if rl > 0 else 0
                pdc = self.vds * idc
                eff = (pout / pdc * 100) if pdc > 0 else 0
                
                power_contours.append([gamma.real, gamma.imag, pout_dbm])
                eff_contours.append([gamma.real, gamma.imag, eff])
        
        return np.array(power_contours), np.array(eff_contours)
    
    def design_bias_network(self, idc_ma=100):
        """
        Design bias network components
        """
        # Quarter-wave bias line
        lambda_4 = c / (4 * self.freq * np.sqrt(2.2))  # Rogers substrate
        
        # RF choke inductor
        l_choke = 50 / (2 * pi * self.freq) * 10  # 10x reactance
        
        # Bypass capacitor
        c_bypass = 1 / (2 * pi * self.freq * 5)  # 5 ohm reactance
        
        return {
            'bias_line_length_mm': lambda_4 * 1000,
            'choke_inductance_nH': l_choke * 1e9,
            'bypass_capacitance_pF': c_bypass * 1e12,
            'gate_resistor_ohm': 100,
            'drain_current_mA': idc_ma
        }

class BeamformingArray:
    """Phased Array Beamforming Calculations"""
    
    def __init__(self, freq=28e9, nx=8, ny=8):
        self.freq = freq
        self.wavelength = c / freq
        self.nx = nx
        self.ny = ny
        self.d = 0.5 * self.wavelength  # Half-wavelength spacing
        
    def calculate_array_factor(self, theta, phi, theta_steer=0, phi_steer=0):
        """
        Calculate array factor for given angles
        """
        k = 2 * pi / self.wavelength
        
        # Create array grid
        af = 0
        for m in range(self.nx):
            for n in range(self.ny):
                # Position
                x = m * self.d
                y = n * self.d
                
                # Phase for steering
                phase_steer = -k * (x * np.sin(theta_steer) * np.cos(phi_steer) +
                                   y * np.sin(theta_steer) * np.sin(phi_steer))
                
                # Array factor contribution
                phase = k * (x * np.sin(theta) * np.cos(phi) +
                           y * np.sin(theta) * np.sin(phi))
                
                af += np.exp(1j * (phase + phase_steer))
        
        return np.abs(af) / (self.nx * self.ny)
    
    def plot_pattern(self, theta_steer=0, phi_steer=0):
        """
        Plot 2D radiation pattern
        """
        theta = np.linspace(-pi/2, pi/2, 361)
        pattern = []
        
        for t in theta:
            af = self.calculate_array_factor(t, phi_steer, theta_steer, phi_steer)
            pattern.append(20 * np.log10(af + 1e-10))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Rectangular plot
        ax1.plot(np.degrees(theta), pattern)
        ax1.set_xlabel('Angle (degrees)')
        ax1.set_ylabel('Gain (dB)')
        ax1.set_title(f'Array Pattern - Steering: {np.degrees(theta_steer):.1f}°')
        ax1.grid(True)
        ax1.set_xlim(-90, 90)
        ax1.set_ylim(-40, 0)
        ax1.axvline(x=np.degrees(theta_steer), color='r', linestyle='--', label='Steering angle')
        ax1.legend()
        
        # Polar plot
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(theta, np.array(pattern) + 40)
        ax2.set_theta_zero_location('N')
        ax2.set_title('Polar Pattern')
        
        return fig
    
    def calculate_directivity(self):
        """
        Calculate array directivity
        """
        # Approximate for uniform array
        d0 = 10 * np.log10(self.nx * self.ny)
        return d0

class DigitalPredistortion:
    """Digital Predistortion Implementation"""
    
    def __init__(self, order=7, memory=4):
        self.order = order
        self.memory = memory
        self.coefficients = None
        
    def create_basis_matrix(self, x):
        """
        Create memory polynomial basis functions
        """
        n = len(x)
        num_coeffs = self.order * self.memory
        basis = np.zeros((n, num_coeffs), dtype=complex)
        
        col = 0
        for k in range(self.order):
            for m in range(self.memory):
                if m <= n-1:
                    x_delayed = np.roll(x, m)
                    if m > 0:
                        x_delayed[:m] = 0
                    basis[:, col] = x_delayed * np.abs(x_delayed)**(2*k)
                col += 1
        
        return basis
    
    def train(self, x_ideal, y_measured):
        """
        Train DPD coefficients using indirect learning
        """
        # Normalize signals
        x_norm = x_ideal / np.max(np.abs(x_ideal))
        y_norm = y_measured / np.max(np.abs(y_measured))
        
        # Create basis matrix from PA output
        basis = self.create_basis_matrix(y_norm)
        
        # Least squares solution
        self.coefficients = np.linalg.lstsq(basis, x_norm, rcond=None)[0]
        
        return self.coefficients
    
    def predistort(self, x):
        """
        Apply predistortion to signal
        """
        if self.coefficients is None:
            raise ValueError("DPD not trained. Call train() first.")
        
        basis = self.create_basis_matrix(x)
        y_dpd = basis @ self.coefficients
        
        # Prevent saturation
        max_val = np.max(np.abs(y_dpd))
        if max_val > 1.0:
            y_dpd = y_dpd / max_val
        
        return y_dpd
    
    def calculate_metrics(self, x_ideal, y_actual):
        """
        Calculate EVM and ACLR
        """
        # Normalize
        x_ideal = x_ideal / np.max(np.abs(x_ideal))
        y_actual = y_actual / np.max(np.abs(y_actual))
        
        # EVM
        evm = np.sqrt(np.mean(np.abs(y_actual - x_ideal)**2))
        evm_percent = evm * 100
        
        # ACLR (simplified)
        f, psd = signal.periodogram(y_actual, fs=1.0)
        in_band = np.sum(psd[len(f)//4:3*len(f)//4])
        out_band = np.sum(psd[:len(f)//4]) + np.sum(psd[3*len(f)//4:])
        aclr_db = 10 * np.log10(in_band / (out_band + 1e-10))
        
        return {'EVM': evm_percent, 'ACLR': aclr_db}

def example_usage():
    """Example usage of the toolkit"""
    
    print("=" * 60)
    print("mmWave RF Frontend Design Toolkit - Example Usage")
    print("=" * 60)
    
    # Initialize toolkit
    rf = RFDesignToolkit(frequency=28e9)
    
    # Path loss calculation
    distance = 100  # meters
    pl = rf.calculate_path_loss(distance)
    print(f"\n1. Path Loss at {distance}m: {pl:.1f} dB")
    
    # Microstrip design
    w = rf.design_microstrip(er=3.5, h=0.5e-3, target_z0=50)
    print(f"\n2. Microstrip width for 50Ω: {w*1000:.3f} mm")
    
    # Noise figure cascade
    stages = [(20, 2.5), (15, 4.0), (-3, 3.0)]  # (gain_dB, nf_dB)
    nf_total = rf.calculate_noise_figure(stages)
    print(f"\n3. Cascaded Noise Figure: {nf_total:.2f} dB")
    
    # Power amplifier design
    pa = PowerAmplifierDesign(vds=28, freq=28e9)
    r_opt = pa.calculate_load_impedance(pout_dbm=35)
    print(f"\n4. Optimal load impedance for 35dBm: {r_opt:.1f} Ω")
    
    bias = pa.design_bias_network(idc_ma=100)
    print(f"   Bias network components:")
    for key, value in bias.items():
        print(f"   - {key}: {value:.2f}")
    
    # Beamforming array
    array = BeamformingArray(freq=28e9, nx=8, ny=8)
    directivity = array.calculate_directivity()
    print(f"\n5. 8x8 Array Directivity: {directivity:.1f} dBi")
    
    # Digital predistortion
    print(f"\n6. Digital Predistortion Test:")
    
    # Generate test signal
    n_samples = 10000
    x = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)
    x = x / np.max(np.abs(x)) * 0.5  # Back off
    
    # Simple PA model (compression)
    def pa_model(x, compression=0.7):
        y = x.copy()
        mask = np.abs(x) > compression
        y[mask] = y[mask] * compression / np.abs(y[mask])
        y = y * np.exp(1j * 0.5 * np.abs(y)**2)  # AM-PM
        return y
    
    y_pa = pa_model(x)
    
    # Train DPD
    dpd = DigitalPredistortion(order=5, memory=3)
    dpd.train(x, y_pa)
    
    # Apply DPD
    x_dpd = dpd.predistort(x)
    y_dpd = pa_model(x_dpd)
    
    # Calculate metrics
    metrics_before = dpd.calculate_metrics(x, y_pa)
    metrics_after = dpd.calculate_metrics(x, y_dpd)
    
    print(f"   Before DPD - EVM: {metrics_before['EVM']:.2f}%, ACLR: {metrics_before['ACLR']:.1f} dB")
    print(f"   After DPD  - EVM: {metrics_after['EVM']:.2f}%, ACLR: {metrics_after['ACLR']:.1f} dB")
    
    print("\n" + "=" * 60)
    print("Toolkit demonstration complete!")
    print("=" * 60)

if __name__ == "__main__":
    example_usage()
    
    # Optional: Generate plots
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--plot':
        print("\nGenerating plots...")
        
        rf = RFDesignToolkit(frequency=28e9)
        pa = PowerAmplifierDesign()
        array = BeamformingArray()
        
        # Smith chart
        fig1 = rf.smith_chart_plot(12.5 + 8.3j)
        
        # Array pattern
        fig2 = array.plot_pattern(theta_steer=np.radians(30))
        
        plt.show()