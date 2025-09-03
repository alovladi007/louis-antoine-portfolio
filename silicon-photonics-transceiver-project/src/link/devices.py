"""Device models for silicon photonics transceiver components"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class MZMConfig:
    """Mach-Zehnder Modulator configuration"""
    vpi: float = 3.0  # V
    length: float = 0.5e-2  # m (5mm)
    vpi_length: float = 1.5  # V·cm
    insertion_loss_db: float = 3.0  # dB
    extinction_ratio_db: float = 10.0  # dB
    bandwidth_ghz: float = 30.0  # GHz
    bias_voltage: float = 1.5  # V


@dataclass 
class RingModConfig:
    """Ring modulator configuration"""
    radius: float = 10e-6  # m
    quality_factor: float = 10000
    coupling_coefficient: float = 0.1
    free_spectral_range: float = 10e9  # Hz
    insertion_loss_db: float = 1.0  # dB
    resonance_wavelength: float = 1310e-9  # m


@dataclass
class PhotodiodeConfig:
    """Photodiode configuration"""
    responsivity: float = 0.8  # A/W
    dark_current: float = 10e-9  # A
    capacitance: float = 50e-15  # F
    bandwidth_ghz: float = 20.0  # GHz
    quantum_efficiency: float = 0.65


@dataclass
class TIAConfig:
    """Transimpedance Amplifier configuration"""
    transimpedance: float = 5000  # Ohm
    bandwidth_ghz: float = 20.0  # GHz
    input_noise_current: float = 10e-12  # A/√Hz
    output_voltage_swing: float = 0.8  # V
    input_capacitance: float = 100e-15  # F


class MachZehnderModulator:
    """Mach-Zehnder Modulator model"""
    
    def __init__(self, config: MZMConfig):
        self.config = config
        self.vpi = config.vpi
        self.insertion_loss = 10**(-config.insertion_loss_db / 20)
        self.extinction_ratio = 10**(config.extinction_ratio_db / 10)
        
    def modulate(self, optical_input: np.ndarray, voltage: np.ndarray) -> np.ndarray:
        """Apply MZM transfer function
        
        Args:
            optical_input: Input optical field
            voltage: Applied voltage signal
            
        Returns:
            Modulated optical field
        """
        # Phase shift
        phase = np.pi * (voltage - self.config.bias_voltage) / self.vpi
        
        # MZM transfer function
        transfer = np.cos(phase / 2)**2
        
        # Apply extinction ratio limitation
        er_linear = self.extinction_ratio
        transfer = (transfer * (er_linear - 1) + 1) / er_linear
        
        # Apply insertion loss
        output = optical_input * np.sqrt(transfer) * self.insertion_loss
        
        return output
    
    def get_transfer_curve(self, v_range: Tuple[float, float] = (-5, 5), 
                          points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Get MZM transfer curve
        
        Args:
            v_range: Voltage range
            points: Number of points
            
        Returns:
            Voltage and transmission arrays
        """
        voltage = np.linspace(v_range[0], v_range[1], points)
        optical_in = np.ones(points)
        transmission = self.modulate(optical_in, voltage)**2
        
        return voltage, transmission


class RingModulator:
    """Ring modulator model"""
    
    def __init__(self, config: RingModConfig):
        self.config = config
        self.fsr = config.free_spectral_range
        self.q = config.quality_factor
        self.kappa = config.coupling_coefficient
        self.insertion_loss = 10**(-config.insertion_loss_db / 20)
        
    def modulate(self, optical_input: np.ndarray, voltage: np.ndarray,
                wavelength: float = 1310e-9) -> np.ndarray:
        """Apply ring modulator transfer function
        
        Args:
            optical_input: Input optical field
            voltage: Applied voltage signal
            wavelength: Operating wavelength
            
        Returns:
            Modulated optical field
        """
        # Voltage-induced resonance shift (simplified)
        delta_n = 1e-4 * voltage  # Simplified plasma dispersion
        freq_shift = delta_n * 3e8 / wavelength
        
        # Lorentzian transfer function
        f0 = 3e8 / self.config.resonance_wavelength
        delta_f = freq_shift
        gamma = f0 / (2 * self.q)
        
        # Through-port transmission
        transmission = 1 - self.kappa**2 / (1 + (delta_f / gamma)**2)
        
        # Apply insertion loss
        output = optical_input * np.sqrt(np.abs(transmission)) * self.insertion_loss
        
        return output


class Photodiode:
    """Photodiode model"""
    
    def __init__(self, config: PhotodiodeConfig):
        self.config = config
        self.responsivity = config.responsivity
        self.dark_current = config.dark_current
        self.bandwidth = config.bandwidth_ghz * 1e9
        
    def detect(self, optical_power: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Convert optical power to photocurrent
        
        Args:
            optical_power: Input optical power (W)
            add_noise: Whether to add shot noise
            
        Returns:
            Photocurrent (A)
        """
        # Photocurrent
        photocurrent = self.responsivity * optical_power + self.dark_current
        
        if add_noise:
            # Shot noise
            q = 1.602e-19  # Electron charge
            shot_noise_variance = 2 * q * np.mean(photocurrent) * self.bandwidth
            shot_noise = np.random.normal(0, np.sqrt(shot_noise_variance), len(photocurrent))
            photocurrent += shot_noise
        
        # Apply bandwidth limitation (simple first-order filter)
        # This is simplified - real implementation would use proper filtering
        alpha = 0.1  # Filter coefficient
        filtered = np.zeros_like(photocurrent)
        filtered[0] = photocurrent[0]
        for i in range(1, len(photocurrent)):
            filtered[i] = alpha * photocurrent[i] + (1 - alpha) * filtered[i-1]
        
        return filtered
    
    def get_responsivity_curve(self, wavelength_range: Tuple[float, float] = (1200e-9, 1400e-9),
                              points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Get responsivity vs wavelength
        
        Args:
            wavelength_range: Wavelength range in meters
            points: Number of points
            
        Returns:
            Wavelength and responsivity arrays
        """
        wavelength = np.linspace(wavelength_range[0], wavelength_range[1], points)
        # Simplified wavelength dependence
        lambda_nm = wavelength * 1e9
        responsivity = self.responsivity * np.exp(-((lambda_nm - 1310) / 100)**2)
        
        return wavelength, responsivity


class TransimpedanceAmplifier:
    """TIA model"""
    
    def __init__(self, config: TIAConfig):
        self.config = config
        self.gain = config.transimpedance
        self.bandwidth = config.bandwidth_ghz * 1e9
        self.noise_current = config.input_noise_current
        self.max_output = config.output_voltage_swing
        
    def amplify(self, current: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Convert photocurrent to voltage
        
        Args:
            current: Input current (A)
            add_noise: Whether to add amplifier noise
            
        Returns:
            Output voltage (V)
        """
        if add_noise:
            # Input-referred noise
            noise = np.random.normal(0, self.noise_current * np.sqrt(self.bandwidth), len(current))
            current = current + noise
        
        # Transimpedance gain
        voltage = current * self.gain
        
        # Output saturation
        voltage = np.clip(voltage, -self.max_output/2, self.max_output/2)
        
        # Bandwidth limitation (single-pole response)
        tau = 1 / (2 * np.pi * self.bandwidth)
        dt = 1e-12  # Assume 1ps time step
        alpha = dt / (tau + dt)
        
        filtered = np.zeros_like(voltage)
        filtered[0] = voltage[0]
        for i in range(1, len(voltage)):
            filtered[i] = alpha * voltage[i] + (1 - alpha) * filtered[i-1]
        
        return filtered
    
    def get_frequency_response(self, freq_range: Tuple[float, float] = (1e6, 100e9),
                              points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Get TIA frequency response
        
        Args:
            freq_range: Frequency range in Hz
            points: Number of points
            
        Returns:
            Frequency and gain arrays (dB)
        """
        freq = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), points)
        # Single-pole response
        gain_db = 20 * np.log10(self.gain) - 20 * np.log10(np.sqrt(1 + (freq/self.bandwidth)**2))
        
        return freq, gain_db