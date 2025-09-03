"""Optical channel model including fiber and connectors"""

import numpy as np
import os
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ChannelConfig:
    """Channel configuration parameters"""
    fiber_length_km: float = 2.0  # km
    fiber_loss_db_per_km: float = 0.35  # dB/km at 1310nm
    connector_loss_db: float = 0.5  # dB per connector
    num_connectors: int = 2  # Input and output
    dispersion_ps_per_nm_km: float = -3.5  # ps/nm/km at 1310nm
    wavelength_nm: float = 1310  # nm
    spectral_width_nm: float = 0.1  # nm


class OpticalChannel:
    """Optical channel model"""
    
    def __init__(self, config: ChannelConfig):
        self.config = config
        self.total_loss_db = (config.fiber_loss_db_per_km * config.fiber_length_km + 
                              config.connector_loss_db * config.num_connectors)
        self.loss_linear = 10**(-self.total_loss_db / 10)
        
    def propagate(self, optical_field: np.ndarray, 
                 include_dispersion: bool = False) -> np.ndarray:
        """Propagate optical signal through channel
        
        Args:
            optical_field: Input optical field
            include_dispersion: Whether to include chromatic dispersion
            
        Returns:
            Output optical field after propagation
        """
        # Apply loss
        output = optical_field * np.sqrt(self.loss_linear)
        
        if include_dispersion and self.config.fiber_length_km > 0:
            # Chromatic dispersion (simplified model)
            dispersion_total = (self.config.dispersion_ps_per_nm_km * 
                               self.config.fiber_length_km * 
                               self.config.spectral_width_nm)
            
            if dispersion_total > 0:
                # Apply dispersion as broadening (simplified)
                # In reality, this would be done in frequency domain
                sigma = dispersion_total * 1e-12  # Convert to seconds
                kernel_size = int(6 * sigma * 25e9)  # Assume 25 GHz sampling
                if kernel_size > 1:
                    kernel = np.exp(-0.5 * (np.arange(kernel_size) - kernel_size//2)**2 / (sigma**2))
                    kernel /= np.sum(kernel)
                    output = np.convolve(output, kernel, mode='same')
        
        return output
    
    def calculate_link_budget(self, tx_power_dbm: float = 0.0) -> dict:
        """Calculate link budget
        
        Args:
            tx_power_dbm: Transmitter power in dBm
            
        Returns:
            Link budget dictionary
        """
        budget = {
            'tx_power_dbm': tx_power_dbm,
            'fiber_loss_db': self.config.fiber_loss_db_per_km * self.config.fiber_length_km,
            'connector_loss_db': self.config.connector_loss_db * self.config.num_connectors,
            'total_loss_db': self.total_loss_db,
            'rx_power_dbm': tx_power_dbm - self.total_loss_db,
        }
        
        # Add margin calculations
        sensitivity_dbm = -20  # Typical receiver sensitivity
        budget['margin_db'] = budget['rx_power_dbm'] - sensitivity_dbm
        
        return budget
    
    def load_sparams(self, filepath: str) -> Optional[np.ndarray]:
        """Load S-parameters from file if available
        
        Args:
            filepath: Path to S-parameter file
            
        Returns:
            S-parameter data or None if not available
        """
        if not os.path.exists(filepath):
            return None
        
        try:
            # Simple touchstone file reader (s2p format)
            data = []
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('#') or line.startswith('!'):
                        continue
                    values = line.strip().split()
                    if len(values) >= 9:  # Frequency + 8 S-params
                        data.append([float(v) for v in values])
            
            if data:
                return np.array(data)
        except Exception as e:
            print(f"Error loading S-parameters: {e}")
        
        return None


class ElectricalChannel:
    """Electrical channel model for package and PCB"""
    
    def __init__(self, bandwidth_ghz: float = 30.0, 
                 trace_length_mm: float = 10.0):
        self.bandwidth = bandwidth_ghz * 1e9
        self.trace_length = trace_length_mm * 1e-3
        
        # Simple RC model
        self.r_per_mm = 0.1  # Ohm/mm
        self.c_per_mm = 0.1e-12  # F/mm
        self.l_per_mm = 1e-9  # H/mm
        
    def apply_channel(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        """Apply electrical channel effects
        
        Args:
            signal: Input electrical signal
            sample_rate: Sampling rate in Hz
            
        Returns:
            Output signal after channel
        """
        # Calculate RC time constant
        r_total = self.r_per_mm * self.trace_length * 1000  # Convert to mm
        c_total = self.c_per_mm * self.trace_length * 1000
        tau = r_total * c_total
        
        # Apply first-order lowpass filter
        alpha = 1 / (1 + 2 * np.pi * self.bandwidth * tau)
        
        filtered = np.zeros_like(signal)
        filtered[0] = signal[0]
        for i in range(1, len(signal)):
            filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i-1]
        
        # Add loss
        loss_db = 0.1 * self.trace_length  # 0.1 dB/mm typical
        loss_linear = 10**(-loss_db / 20)
        filtered *= loss_linear
        
        return filtered
    
    def get_transfer_function(self, freq_range: Tuple[float, float] = (1e6, 100e9),
                             points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Get channel transfer function
        
        Args:
            freq_range: Frequency range in Hz
            points: Number of points
            
        Returns:
            Frequency and magnitude arrays (dB)
        """
        freq = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), points)
        
        # Calculate impedance
        r = self.r_per_mm * self.trace_length * 1000
        l = self.l_per_mm * self.trace_length * 1000
        c = self.c_per_mm * self.trace_length * 1000
        
        omega = 2 * np.pi * freq
        z = r + 1j * omega * l
        y = 1j * omega * c
        
        # Transmission line model (simplified)
        gamma = np.sqrt(z * y)
        z0 = np.sqrt(z / y)
        
        # Assume matched load
        transfer = np.exp(-gamma * self.trace_length)
        magnitude_db = 20 * np.log10(np.abs(transfer))
        
        return freq, magnitude_db