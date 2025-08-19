"""Unit tests for JMAK kinetics module."""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcm.kinetics import JMAKModel


def test_jmak_initialization():
    """Test JMAK model initialization."""
    model = JMAKModel()
    assert model.params["k0_cryst"] > 0
    assert model.params["Ea_cryst_eV"] > 0
    assert model.params["avrami_n"] > 0


def test_jmak_monotonic():
    """Test that crystallization is monotonic."""
    model = JMAKModel()
    t = np.logspace(-9, -3, 100)
    X = model.simulate_isothermal(T=500.0, t_array=t, X0=0.0)
    
    # Check monotonic increase
    assert np.all(np.diff(X) >= -1e-10)  # Allow small numerical errors
    

def test_jmak_bounds():
    """Test that X stays within [0, 1]."""
    model = JMAKModel()
    t = np.logspace(-9, -2, 100)
    
    for T in [400, 500, 600, 700]:
        X = model.simulate_isothermal(T=T, t_array=t, X0=0.0)
        assert np.all(X >= 0.0)
        assert np.all(X <= 1.0)


def test_jmak_initial_condition():
    """Test that initial condition is preserved."""
    model = JMAKModel()
    t = np.array([0.0, 1e-6, 1e-3])
    
    for X0 in [0.0, 0.3, 0.7, 1.0]:
        X = model.simulate_isothermal(T=500.0, t_array=t, X0=X0)
        assert X[0] == pytest.approx(X0)


def test_jmak_below_tc():
    """Test no crystallization below Tc."""
    model = JMAKModel()
    t = np.logspace(-9, -3, 50)
    Tc = model.params["Tc_K"]
    
    X = model.simulate_isothermal(T=Tc-50, t_array=t, X0=0.1)
    assert np.all(X == pytest.approx(0.1))