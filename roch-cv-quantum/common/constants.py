"""
Physical constants and default parameters for CV quantum optics.

All quantities in SI units unless otherwise specified.
Quadrature ordering: (x1, p1, x2, p2, ...)
"""

import numpy as np

# Fundamental constants
HBAR = 1.054571817e-34  # J·s (Planck constant / 2π)
C_LIGHT = 299792458.0  # m/s (speed of light)
K_B = 1.380649e-23  # J/K (Boltzmann constant)

# Default wavelength
LAMBDA_TELECOM = 1550e-9  # m (telecom C-band)
OMEGA_TELECOM = 2 * np.pi * C_LIGHT / LAMBDA_TELECOM  # rad/s

# Quadrature conventions
# Convention 1: ℏ = 1, vacuum variance = 1/2
# Convention 2: Shot-noise units (SNU), vacuum variance = 1
VACUUM_VARIANCE_HBAR = 0.5  # ℏ = 1 convention
VACUUM_VARIANCE_SNU = 1.0  # Shot-noise units

# Default: use shot-noise units for practical calculations
SHOT_NOISE_VARIANCE = VACUUM_VARIANCE_SNU

# Material properties (defaults for SiN/TFLN)
N_SIN = 2.0  # Refractive index of Si3N4 @ 1550 nm
N_TFLN = 2.14  # Refractive index of thin-film LiNbO3 @ 1550 nm
N_SIO2 = 1.45  # Refractive index of SiO2 @ 1550 nm

# Nonlinear coefficients
N2_SIN = 2.5e-19  # m²/W (Kerr coefficient for Si3N4)
D33_LN = 27e-12  # m/V (χ⁽²⁾ coefficient for LiNbO3)

# Typical loss values (dB/cm)
LOSS_SIN_TYPICAL = 0.3  # dB/cm
LOSS_TFLN_TYPICAL = 1.0  # dB/cm

# Unit conversions
def db_to_linear(db: float) -> float:
    """Convert dB to linear scale."""
    return 10 ** (db / 10)

def linear_to_db(linear: float) -> float:
    """Convert linear to dB."""
    return 10 * np.log10(linear)

def squeezing_db_to_variance(sq_db: float, convention: str = "SNU") -> float:
    """
    Convert squeezing in dB to variance.

    Parameters
    ----------
    sq_db : float
        Squeezing in dB (negative for sub-shot-noise)
    convention : str
        "SNU" for shot-noise units (default) or "HBAR" for ℏ=1

    Returns
    -------
    float
        Variance in chosen convention
    """
    vac_var = VACUUM_VARIANCE_SNU if convention == "SNU" else VACUUM_VARIANCE_HBAR
    return vac_var * db_to_linear(sq_db)

def variance_to_squeezing_db(variance: float, convention: str = "SNU") -> float:
    """
    Convert variance to squeezing in dB.

    Parameters
    ----------
    variance : float
        Variance in chosen convention
    convention : str
        "SNU" for shot-noise units (default) or "HBAR" for ℏ=1

    Returns
    -------
    float
        Squeezing in dB (negative for sub-shot-noise)
    """
    vac_var = VACUUM_VARIANCE_SNU if convention == "SNU" else VACUUM_VARIANCE_HBAR
    return linear_to_db(variance / vac_var)
