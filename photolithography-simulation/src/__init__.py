"""
Photolithography & Optical Metrology Simulation Package

This package provides tools for modeling and simulating key steps in 
semiconductor photolithography and optical metrology workflows.
"""

from .mask_generation import MaskPatternGenerator
from .opc import OpticalProximityCorrection
from .defect_inspection import DefectInspector
from .fourier_optics import FourierOpticsSimulator
from .monte_carlo import MonteCarloSimulator

__version__ = "1.0.0"
__all__ = [
    "MaskPatternGenerator",
    "OpticalProximityCorrection",
    "DefectInspector",
    "FourierOpticsSimulator",
    "MonteCarloSimulator"
]