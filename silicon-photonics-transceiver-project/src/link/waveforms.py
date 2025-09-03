"""Waveform generation for NRZ and PAM4 signaling"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class WaveformConfig:
    """Configuration for waveform generation"""
    bitrate: float = 25e9  # bits/second
    samples_per_bit: int = 32
    prbs_order: int = 15  # 2^15 - 1
    jitter_rms: float = 0.01  # UI
    rin_db: float = -140  # dB/Hz
    thermal_noise: float = 10e-12  # A/√Hz
    seed: int = 42


def generate_prbs(order: int, seed: int = 42) -> np.ndarray:
    """Generate PRBS pattern
    
    Args:
        order: PRBS order (e.g., 15 for PRBS15)
        seed: Random seed for reproducibility
        
    Returns:
        Binary sequence of length 2^order - 1
    """
    np.random.seed(seed)
    length = 2**order - 1
    
    # Initialize with random bits
    register = np.random.randint(0, 2, order)
    sequence = np.zeros(length, dtype=int)
    
    # Generate PRBS using shift register
    taps = {7: [7, 6], 9: [9, 5], 10: [10, 7], 
            11: [11, 9], 15: [15, 14], 23: [23, 18], 31: [31, 28]}
    
    if order not in taps:
        # Fallback to random for unsupported orders
        return np.random.randint(0, 2, length)
    
    tap_positions = [t - 1 for t in taps[order]]
    
    for i in range(length):
        # XOR tapped bits
        feedback = register[tap_positions[0]]
        for tap in tap_positions[1:]:
            feedback ^= register[tap]
        
        sequence[i] = register[-1]
        # Shift and insert feedback
        register = np.roll(register, 1)
        register[0] = feedback
    
    return sequence


def generate_pam4(bits: np.ndarray) -> np.ndarray:
    """Convert binary sequence to PAM4 symbols
    
    Args:
        bits: Binary sequence
        
    Returns:
        PAM4 symbol sequence (Gray coded: 0, 1, 3, 2)
    """
    # Ensure even length
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)
    
    # Group into 2-bit symbols
    symbols = bits.reshape(-1, 2)
    
    # Gray coding mapping
    gray_map = {(0, 0): 0, (0, 1): 1, (1, 1): 3, (1, 0): 2}
    pam4 = np.array([gray_map[(b[0], b[1])] for b in symbols])
    
    return pam4


def add_jitter(time: np.ndarray, jitter_rms: float, bitrate: float) -> np.ndarray:
    """Add timing jitter to signal
    
    Args:
        time: Time vector
        jitter_rms: RMS jitter in UI
        bitrate: Data rate in bits/second
        
    Returns:
        Jittered time vector
    """
    if jitter_rms == 0:
        return time
    
    ui = 1.0 / bitrate
    jitter_std = jitter_rms * ui
    jitter = np.random.normal(0, jitter_std, len(time))
    
    return time + jitter


def add_rin(signal: np.ndarray, rin_db: float, fs: float) -> np.ndarray:
    """Add Relative Intensity Noise (RIN)
    
    Args:
        signal: Optical signal
        rin_db: RIN in dB/Hz
        fs: Sampling frequency
        
    Returns:
        Signal with RIN
    """
    if rin_db >= 0:  # No RIN
        return signal
    
    rin_linear = 10**(rin_db / 10)
    noise_power = rin_linear * fs / 2  # Noise power in bandwidth
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    
    return signal * (1 + noise)


def add_thermal_noise(signal: np.ndarray, noise_density: float, 
                      bandwidth: float) -> np.ndarray:
    """Add thermal noise
    
    Args:
        signal: Electrical signal (current)
        noise_density: Noise current density in A/√Hz
        bandwidth: Electrical bandwidth in Hz
        
    Returns:
        Signal with thermal noise
    """
    if noise_density == 0:
        return signal
    
    noise_power = noise_density**2 * bandwidth
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    
    return signal + noise


def oversample_signal(symbols: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    """Oversample symbol sequence
    
    Args:
        symbols: Symbol sequence
        samples_per_symbol: Oversampling factor
        
    Returns:
        Oversampled signal
    """
    # Repeat each symbol
    oversampled = np.repeat(symbols, samples_per_symbol)
    
    # Apply raised cosine filter for pulse shaping
    if samples_per_symbol > 1:
        # Simple RC filter approximation
        alpha = 0.35  # Roll-off factor
        filter_len = 4 * samples_per_symbol + 1
        t = np.arange(filter_len) - filter_len // 2
        t = t / samples_per_symbol
        
        # Raised cosine impulse response
        h = np.sinc(t) * np.cos(np.pi * alpha * t) / (1 - (2 * alpha * t)**2 + 1e-10)
        h = h / np.sum(h)
        
        # Apply filter
        oversampled = np.convolve(oversampled, h, mode='same')
    
    return oversampled


def generate_nrz_waveform(config: WaveformConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate NRZ waveform with impairments
    
    Args:
        config: Waveform configuration
        
    Returns:
        Tuple of (time, signal, bits)
    """
    # Generate PRBS
    bits = generate_prbs(config.prbs_order, config.seed)
    
    # Oversample
    signal = oversample_signal(bits.astype(float), config.samples_per_bit)
    
    # Time vector
    dt = 1.0 / (config.bitrate * config.samples_per_bit)
    time = np.arange(len(signal)) * dt
    
    # Add jitter
    time = add_jitter(time, config.jitter_rms, config.bitrate)
    
    return time, signal, bits


def generate_pam4_waveform(config: WaveformConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate PAM4 waveform with impairments
    
    Args:
        config: Waveform configuration
        
    Returns:
        Tuple of (time, signal, symbols)
    """
    # Generate PRBS
    bits = generate_prbs(config.prbs_order, config.seed)
    
    # Convert to PAM4
    symbols = generate_pam4(bits)
    
    # Normalize levels to [0, 1]
    signal = symbols / 3.0
    
    # Oversample
    signal = oversample_signal(signal, config.samples_per_bit)
    
    # Time vector
    symbol_rate = config.bitrate / 2  # 2 bits per symbol
    dt = 1.0 / (symbol_rate * config.samples_per_bit)
    time = np.arange(len(signal)) * dt
    
    # Add jitter
    time = add_jitter(time, config.jitter_rms, symbol_rate)
    
    return time, signal, symbols