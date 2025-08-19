"""Unit tests for device module."""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcm.device import PCMDevice


def test_device_initialization():
    """Test device model initialization."""
    device = PCMDevice()
    assert device.params["thickness_m"] > 0
    assert device.params["area_m2"] > 0
    assert device.params["Cth_J_per_K"] > 0


def test_resistance_mixing():
    """Test resistance mixing model."""
    device = PCMDevice()
    T = 300.0
    
    # Pure amorphous
    R_amor = device.mix_resistivity(X=0.0, T=T)
    
    # Pure crystalline
    R_cryst = device.mix_resistivity(X=1.0, T=T)
    
    # Mixed state
    R_mix = device.mix_resistivity(X=0.5, T=T)
    
    assert R_cryst < R_amor  # Crystalline more conductive
    assert R_cryst < R_mix < R_amor  # Mixed in between


def test_voltage_pulse_simulation():
    """Test basic pulse simulation."""
    device = PCMDevice()
    
    # Create simple pulse
    t = np.linspace(0, 100e-9, 100)
    V = np.ones_like(t) * 1.0
    
    result = device.simulate_voltage_pulse(V, t, X0=0.5)
    
    assert "t_s" in result
    assert "T_K" in result
    assert "X" in result
    assert len(result["T_K"]) == len(t)


def test_reset_pulse():
    """Test RESET pulse generation."""
    device = PCMDevice()
    t, V = device.create_reset_pulse()
    
    assert len(t) == len(V)
    assert np.max(V) > 0
    assert V[-1] == 0  # Pulse should end at zero


def test_set_pulse():
    """Test SET pulse generation."""
    device = PCMDevice()
    t, V = device.create_set_pulse()
    
    assert len(t) == len(V)
    assert np.max(V) > 0
    assert V[-1] == 0  # Pulse should end at zero