"""
Material properties database for III-N semiconductors
Temperature-dependent properties for thermal modeling of InGaN/GaN HEMTs
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Callable, Optional


@dataclass
class MaterialProperties:
    """Container for temperature-dependent material properties"""
    
    # Basic properties
    bandgap: Callable[[float], float]  # eV
    electron_affinity: float  # eV
    relative_permittivity: float
    
    # Thermal properties
    thermal_conductivity: Callable[[float], float]  # W/(m·K)
    specific_heat: Callable[[float], float]  # J/(kg·K)
    density: float  # kg/m³
    
    # Electrical properties
    electron_mobility: Callable[[float], float]  # cm²/(V·s)
    hole_mobility: Callable[[float], float]  # cm²/(V·s)
    saturation_velocity: float  # cm/s
    
    # Lattice properties
    lattice_constant_a: float  # Å
    lattice_constant_c: float  # Å
    
    # Polarization properties
    spontaneous_polarization: float  # C/m²
    piezoelectric_constants: Dict[str, float]  # e31, e33, e15
    elastic_constants: Dict[str, float]  # C11, C12, C13, C33, C44


class GaNProperties:
    """GaN material properties with temperature dependence"""
    
    @staticmethod
    def bandgap(T: float) -> float:
        """Varshni model for GaN bandgap (eV)"""
        Eg0 = 3.507  # eV at 0K
        alpha = 0.909e-3  # eV/K
        beta = 830  # K
        return Eg0 - alpha * T**2 / (T + beta)
    
    @staticmethod
    def thermal_conductivity(T: float) -> float:
        """Temperature-dependent thermal conductivity (W/m·K)"""
        # Based on experimental data for bulk GaN
        k300 = 230  # W/(m·K) at 300K
        return k300 * (300 / T)**1.4
    
    @staticmethod
    def specific_heat(T: float) -> float:
        """Temperature-dependent specific heat (J/kg·K)"""
        # Debye model approximation
        return 490 + 0.35 * (T - 300)
    
    @staticmethod
    def electron_mobility(T: float) -> float:
        """Temperature-dependent electron mobility (cm²/V·s)"""
        # Caughey-Thomas model
        mu_min = 55
        mu_max = 1000
        T_ref = 300
        alpha = 2.0
        return mu_min + (mu_max - mu_min) / (1 + (T / T_ref)**alpha)
    
    @staticmethod
    def hole_mobility(T: float) -> float:
        """Temperature-dependent hole mobility (cm²/V·s)"""
        return 30 * (300 / T)**1.5
    
    @staticmethod
    def get_properties() -> MaterialProperties:
        """Return complete GaN material properties"""
        return MaterialProperties(
            bandgap=GaNProperties.bandgap,
            electron_affinity=4.1,
            relative_permittivity=8.9,
            thermal_conductivity=GaNProperties.thermal_conductivity,
            specific_heat=GaNProperties.specific_heat,
            density=6150,  # kg/m³
            electron_mobility=GaNProperties.electron_mobility,
            hole_mobility=GaNProperties.hole_mobility,
            saturation_velocity=2.5e7,  # cm/s
            lattice_constant_a=3.189,  # Å
            lattice_constant_c=5.185,  # Å
            spontaneous_polarization=-0.034,  # C/m²
            piezoelectric_constants={
                'e31': -0.36,  # C/m²
                'e33': 0.65,   # C/m²
                'e15': -0.30   # C/m²
            },
            elastic_constants={
                'C11': 390,  # GPa
                'C12': 145,  # GPa
                'C13': 106,  # GPa
                'C33': 398,  # GPa
                'C44': 105   # GPa
            }
        )


class InGaNProperties:
    """InxGa1-xN material properties with composition and temperature dependence"""
    
    def __init__(self, x: float):
        """Initialize with In composition x (0 <= x <= 1)"""
        self.x = x
        self.gan = GaNProperties()
        self.inn = InNProperties()
    
    def bandgap(self, T: float) -> float:
        """Bandgap with bowing parameter"""
        Eg_GaN = self.gan.bandgap(T)
        Eg_InN = self.inn.bandgap(T)
        b = 1.4  # bowing parameter
        return self.x * Eg_InN + (1 - self.x) * Eg_GaN - b * self.x * (1 - self.x)
    
    def thermal_conductivity(self, T: float) -> float:
        """Thermal conductivity with alloy scattering"""
        k_GaN = self.gan.thermal_conductivity(T)
        k_InN = self.inn.thermal_conductivity(T)
        # Virtual crystal approximation with disorder scattering
        k_vca = self.x * k_InN + (1 - self.x) * k_GaN
        # Alloy disorder reduction factor
        disorder_factor = 1 - 0.5 * self.x * (1 - self.x)
        return k_vca * disorder_factor
    
    def electron_mobility(self, T: float) -> float:
        """Electron mobility with alloy scattering"""
        mu_GaN = self.gan.electron_mobility(T)
        mu_InN = self.inn.electron_mobility(T)
        # Matthiessen's rule with alloy scattering
        mu_alloy = 1 / (self.x / mu_InN + (1 - self.x) / mu_GaN)
        # Additional alloy scattering
        mu_disorder = 200 / (self.x * (1 - self.x))
        return 1 / (1 / mu_alloy + 1 / mu_disorder)
    
    def get_properties(self) -> MaterialProperties:
        """Return complete InGaN material properties"""
        return MaterialProperties(
            bandgap=self.bandgap,
            electron_affinity=4.1 - 0.8 * self.x,
            relative_permittivity=8.9 + 6.1 * self.x,
            thermal_conductivity=self.thermal_conductivity,
            specific_heat=lambda T: self.x * self.inn.specific_heat(T) + 
                                   (1 - self.x) * self.gan.specific_heat(T),
            density=6150 + (6810 - 6150) * self.x,  # kg/m³
            electron_mobility=self.electron_mobility,
            hole_mobility=lambda T: 30 * (300 / T)**1.5,
            saturation_velocity=2.5e7 - 0.5e7 * self.x,  # cm/s
            lattice_constant_a=3.189 + (3.545 - 3.189) * self.x,  # Å
            lattice_constant_c=5.185 + (5.703 - 5.185) * self.x,  # Å
            spontaneous_polarization=-0.034 - 0.008 * self.x,  # C/m²
            piezoelectric_constants={
                'e31': -0.36 - 0.21 * self.x,  # C/m²
                'e33': 0.65 + 0.32 * self.x,   # C/m²
                'e15': -0.30 - 0.18 * self.x   # C/m²
            },
            elastic_constants={
                'C11': 390 - 167 * self.x,  # GPa
                'C12': 145 - 30 * self.x,   # GPa
                'C13': 106 - 14 * self.x,   # GPa
                'C33': 398 - 174 * self.x,  # GPa
                'C44': 105 - 19 * self.x    # GPa
            }
        )


class InNProperties:
    """InN material properties with temperature dependence"""
    
    @staticmethod
    def bandgap(T: float) -> float:
        """Varshni model for InN bandgap (eV)"""
        Eg0 = 0.69  # eV at 0K
        alpha = 0.414e-3  # eV/K
        beta = 454  # K
        return Eg0 - alpha * T**2 / (T + beta)
    
    @staticmethod
    def thermal_conductivity(T: float) -> float:
        """Temperature-dependent thermal conductivity (W/m·K)"""
        k300 = 45  # W/(m·K) at 300K
        return k300 * (300 / T)**1.2
    
    @staticmethod
    def specific_heat(T: float) -> float:
        """Temperature-dependent specific heat (J/kg·K)"""
        return 320 + 0.25 * (T - 300)
    
    @staticmethod
    def electron_mobility(T: float) -> float:
        """Temperature-dependent electron mobility (cm²/V·s)"""
        mu_300 = 3200
        return mu_300 * (300 / T)**1.5


class AlNProperties:
    """AlN material properties for substrate and buffer layers"""
    
    @staticmethod
    def thermal_conductivity(T: float) -> float:
        """Temperature-dependent thermal conductivity (W/m·K)"""
        k300 = 285  # W/(m·K) at 300K
        return k300 * (300 / T)**1.3
    
    @staticmethod
    def get_properties() -> MaterialProperties:
        """Return complete AlN material properties"""
        return MaterialProperties(
            bandgap=lambda T: 6.2 - 1.799e-3 * T,
            electron_affinity=0.6,
            relative_permittivity=8.5,
            thermal_conductivity=AlNProperties.thermal_conductivity,
            specific_heat=lambda T: 600 + 0.4 * (T - 300),
            density=3260,  # kg/m³
            electron_mobility=lambda T: 300 * (300 / T)**1.5,
            hole_mobility=lambda T: 14 * (300 / T)**1.5,
            saturation_velocity=1.4e7,  # cm/s
            lattice_constant_a=3.112,  # Å
            lattice_constant_c=4.982,  # Å
            spontaneous_polarization=-0.090,  # C/m²
            piezoelectric_constants={
                'e31': -0.58,  # C/m²
                'e33': 1.55,   # C/m²
                'e15': -0.48   # C/m²
            },
            elastic_constants={
                'C11': 410,  # GPa
                'C12': 140,  # GPa
                'C13': 99,   # GPa
                'C33': 389,  # GPa
                'C44': 125   # GPa
            }
        )


# Substrate materials
class SiCProperties:
    """4H-SiC substrate properties"""
    
    @staticmethod
    def thermal_conductivity(T: float) -> float:
        """Temperature-dependent thermal conductivity (W/m·K)"""
        k300 = 370  # W/(m·K) at 300K
        return k300 * (300 / T)**1.3
    
    @staticmethod
    def get_properties() -> Dict[str, float]:
        """Return SiC substrate properties"""
        return {
            'thermal_conductivity': SiCProperties.thermal_conductivity,
            'specific_heat': lambda T: 690 + 0.3 * (T - 300),
            'density': 3210,  # kg/m³
            'thermal_expansion': 4.3e-6  # 1/K
        }


class SiProperties:
    """Silicon substrate properties"""
    
    @staticmethod
    def thermal_conductivity(T: float) -> float:
        """Temperature-dependent thermal conductivity (W/m·K)"""
        k300 = 148  # W/(m·K) at 300K
        return k300 * (300 / T)**1.3
    
    @staticmethod
    def get_properties() -> Dict[str, float]:
        """Return Si substrate properties"""
        return {
            'thermal_conductivity': SiProperties.thermal_conductivity,
            'specific_heat': lambda T: 700 + 0.2 * (T - 300),
            'density': 2329,  # kg/m³
            'thermal_expansion': 2.6e-6  # 1/K
        }


class SapphireProperties:
    """Sapphire (Al2O3) substrate properties"""
    
    @staticmethod
    def thermal_conductivity(T: float) -> float:
        """Temperature-dependent thermal conductivity (W/m·K)"""
        k300 = 35  # W/(m·K) at 300K
        return k300 * (300 / T)**1.0
    
    @staticmethod
    def get_properties() -> Dict[str, float]:
        """Return sapphire substrate properties"""
        return {
            'thermal_conductivity': SapphireProperties.thermal_conductivity,
            'specific_heat': lambda T: 760 + 0.4 * (T - 300),
            'density': 3980,  # kg/m³
            'thermal_expansion': 7.5e-6  # 1/K
        }


# Interface thermal resistance models
def thermal_boundary_resistance(T: float, material1: str, material2: str) -> float:
    """
    Calculate thermal boundary resistance between two materials (m²·K/W)
    Based on acoustic mismatch model
    """
    # Typical values for common interfaces
    tbr_values = {
        ('GaN', 'SiC'): 2e-9,
        ('GaN', 'Si'): 5e-9,
        ('GaN', 'Sapphire'): 1e-8,
        ('GaN', 'AlN'): 1e-9,
        ('InGaN', 'GaN'): 3e-9,
    }
    
    key = (material1, material2)
    reverse_key = (material2, material1)
    
    if key in tbr_values:
        tbr_300 = tbr_values[key]
    elif reverse_key in tbr_values:
        tbr_300 = tbr_values[reverse_key]
    else:
        # Default value for unknown interfaces
        tbr_300 = 5e-9
    
    # Temperature dependence (approximate)
    return tbr_300 * (T / 300)**0.5


if __name__ == "__main__":
    # Example usage
    T = 300  # K
    
    # GaN properties at 300K
    gan = GaNProperties.get_properties()
    print(f"GaN properties at {T}K:")
    print(f"  Bandgap: {gan.bandgap(T):.3f} eV")
    print(f"  Thermal conductivity: {gan.thermal_conductivity(T):.1f} W/(m·K)")
    print(f"  Electron mobility: {gan.electron_mobility(T):.0f} cm²/(V·s)")
    
    # In0.2Ga0.8N properties
    ingan = InGaNProperties(x=0.2)
    ingan_props = ingan.get_properties()
    print(f"\nIn0.2Ga0.8N properties at {T}K:")
    print(f"  Bandgap: {ingan_props.bandgap(T):.3f} eV")
    print(f"  Thermal conductivity: {ingan_props.thermal_conductivity(T):.1f} W/(m·K)")
    print(f"  Electron mobility: {ingan_props.electron_mobility(T):.0f} cm²/(V·s)")