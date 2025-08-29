"""
Fourier Optics Simulation Module

This module implements Fourier optics methods for simulating
optical imaging systems in photolithography.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special, interpolate
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import warnings


@dataclass
class OpticalSystem:
    """Parameters defining an optical system"""
    wavelength: float = 193.0  # nm (ArF laser)
    NA: float = 0.75  # Numerical aperture
    magnification: float = 4.0  # Reduction ratio
    sigma_inner: float = 0.0  # Inner sigma for annular illumination
    sigma_outer: float = 0.5  # Outer sigma
    aberrations: Optional[Dict[str, float]] = None  # Zernike coefficients
    immersion: bool = True  # Immersion lithography
    n_medium: float = 1.44  # Refractive index of immersion medium
    
    @property
    def k1_factor(self) -> float:
        """Calculate k1 factor"""
        return 0.61 * (1 - self.sigma_inner + self.sigma_outer) / 2
    
    @property
    def resolution(self) -> float:
        """Calculate resolution limit"""
        n = self.n_medium if self.immersion else 1.0
        return self.k1_factor * self.wavelength / (n * self.NA)
    
    @property
    def depth_of_focus(self) -> float:
        """Calculate depth of focus"""
        n = self.n_medium if self.immersion else 1.0
        return 0.5 * self.wavelength / (n * self.NA**2)


class FourierOpticsSimulator:
    """
    Simulate optical imaging using Fourier optics methods
    """
    
    def __init__(self, system: Optional[OpticalSystem] = None):
        """
        Initialize Fourier optics simulator
        
        Args:
            system: Optical system parameters
        """
        self.system = system or OpticalSystem()
        self.pupil_function = None
        self.illumination_source = None
        self.transfer_function = None
        
    def create_illumination_source(self, shape: Tuple[int, int],
                                  pixel_size: float,
                                  source_type: str = 'conventional') -> np.ndarray:
        """
        Create illumination source distribution
        
        Args:
            shape: Array shape (ny, nx)
            pixel_size: Physical size of each pixel in nm
            source_type: Type of illumination source
            
        Returns:
            Source distribution in frequency domain
        """
        ny, nx = shape
        
        # Create frequency coordinates
        fx = fftfreq(nx, d=pixel_size)
        fy = fftfreq(ny, d=pixel_size)
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        
        # Normalized frequency
        f_norm = np.sqrt(fx_grid**2 + fy_grid**2) * self.system.wavelength / self.system.NA
        
        if source_type == 'conventional':
            # Circular source
            source = np.where(f_norm <= self.system.sigma_outer, 1.0, 0.0)
            
        elif source_type == 'annular':
            # Annular illumination
            source = np.where(
                (f_norm >= self.system.sigma_inner) & 
                (f_norm <= self.system.sigma_outer),
                1.0, 0.0
            )
            
        elif source_type == 'quadrupole':
            # Quadrupole illumination
            angle = np.arctan2(fy_grid, fx_grid)
            quadrupole = (
                (np.abs(np.cos(2 * angle)) > 0.7) &
                (f_norm >= self.system.sigma_inner) &
                (f_norm <= self.system.sigma_outer)
            )
            source = np.where(quadrupole, 1.0, 0.0)
            
        elif source_type == 'dipole':
            # Dipole illumination (horizontal)
            dipole = (
                (np.abs(fx_grid) > 0.3 * self.system.NA / self.system.wavelength) &
                (np.abs(fy_grid) < 0.2 * self.system.NA / self.system.wavelength) &
                (f_norm <= self.system.sigma_outer)
            )
            source = np.where(dipole, 1.0, 0.0)
            
        elif source_type == 'quasar':
            # Quasar illumination
            angle = np.arctan2(fy_grid, fx_grid)
            quasar = (
                (np.abs(np.sin(2 * angle)) < 0.3) &
                (f_norm >= self.system.sigma_inner) &
                (f_norm <= self.system.sigma_outer)
            )
            source = np.where(quasar, 1.0, 0.0)
            
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        # Normalize
        if np.sum(source) > 0:
            source = source / np.sqrt(np.sum(source**2))
        
        self.illumination_source = source
        return source
    
    def create_pupil_function(self, shape: Tuple[int, int],
                             pixel_size: float,
                             include_aberrations: bool = True) -> np.ndarray:
        """
        Create pupil function including aberrations
        
        Args:
            shape: Array shape (ny, nx)
            pixel_size: Physical size of each pixel in nm
            include_aberrations: Whether to include aberrations
            
        Returns:
            Complex pupil function
        """
        ny, nx = shape
        
        # Create frequency coordinates
        fx = fftfreq(nx, d=pixel_size)
        fy = fftfreq(ny, d=pixel_size)
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        
        # Normalized radial coordinate
        rho = np.sqrt(fx_grid**2 + fy_grid**2) * self.system.wavelength / self.system.NA
        theta = np.arctan2(fy_grid, fx_grid)
        
        # Pupil aperture
        pupil = np.where(rho <= 1.0, 1.0, 0.0)
        
        # Add aberrations if specified
        if include_aberrations and self.system.aberrations:
            phase = self._calculate_aberration_phase(rho, theta)
            pupil = pupil * np.exp(1j * phase)
        
        self.pupil_function = pupil
        return pupil
    
    def _calculate_aberration_phase(self, rho: np.ndarray, 
                                   theta: np.ndarray) -> np.ndarray:
        """
        Calculate phase aberrations using Zernike polynomials
        
        Args:
            rho: Normalized radial coordinate
            theta: Azimuthal angle
            
        Returns:
            Phase aberration map
        """
        phase = np.zeros_like(rho)
        
        if self.system.aberrations is None:
            return phase
        
        # Zernike polynomial definitions (first 15 terms)
        zernike_terms = {
            'piston': lambda r, t: np.ones_like(r),
            'tilt_x': lambda r, t: r * np.cos(t),
            'tilt_y': lambda r, t: r * np.sin(t),
            'defocus': lambda r, t: 2 * r**2 - 1,
            'astigmatism_0': lambda r, t: r**2 * np.cos(2*t),
            'astigmatism_45': lambda r, t: r**2 * np.sin(2*t),
            'coma_x': lambda r, t: (3*r**3 - 2*r) * np.cos(t),
            'coma_y': lambda r, t: (3*r**3 - 2*r) * np.sin(t),
            'spherical': lambda r, t: 6*r**4 - 6*r**2 + 1,
            'trefoil_0': lambda r, t: r**3 * np.cos(3*t),
            'trefoil_30': lambda r, t: r**3 * np.sin(3*t),
        }
        
        # Add aberration contributions
        for name, coefficient in self.system.aberrations.items():
            if name in zernike_terms and coefficient != 0:
                # Mask to pupil area
                mask = rho <= 1.0
                term = zernike_terms[name](rho, theta)
                phase[mask] += coefficient * term[mask]
        
        return phase
    
    def calculate_psf(self, shape: Tuple[int, int] = (256, 256),
                     pixel_size: float = 5.0) -> np.ndarray:
        """
        Calculate Point Spread Function (PSF)
        
        Args:
            shape: Array shape
            pixel_size: Physical size of each pixel in nm
            
        Returns:
            PSF intensity distribution
        """
        # Create pupil function
        pupil = self.create_pupil_function(shape, pixel_size)
        
        # Calculate PSF (Fourier transform of pupil)
        psf_complex = fftshift(ifft2(ifftshift(pupil)))
        psf = np.abs(psf_complex)**2
        
        # Normalize
        psf = psf / np.max(psf)
        
        return psf
    
    def calculate_mtf(self, shape: Tuple[int, int] = (256, 256),
                     pixel_size: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Modulation Transfer Function (MTF)
        
        Args:
            shape: Array shape
            pixel_size: Physical size of each pixel in nm
            
        Returns:
            Frequency array and MTF values
        """
        # Calculate PSF
        psf = self.calculate_psf(shape, pixel_size)
        
        # Calculate OTF (Fourier transform of PSF)
        otf = fftshift(fft2(ifftshift(psf)))
        
        # MTF is magnitude of OTF
        mtf = np.abs(otf)
        mtf = mtf / np.max(mtf)  # Normalize
        
        # Create frequency axis
        freq = fftfreq(shape[0], d=pixel_size)
        
        # Extract radial profile
        center = shape[0] // 2
        radial_mtf = mtf[center, center:]
        radial_freq = freq[center:]
        
        return radial_freq, radial_mtf
    
    def simulate_image_formation(self, mask: np.ndarray,
                                pixel_size: float = 5.0,
                                source_type: str = 'conventional',
                                defocus: float = 0.0) -> np.ndarray:
        """
        Simulate image formation through the optical system
        
        Args:
            mask: Input mask pattern
            pixel_size: Physical size of each pixel in nm
            source_type: Type of illumination
            defocus: Defocus amount in nm
            
        Returns:
            Aerial image intensity
        """
        # Ensure mask is float
        mask = mask.astype(np.float64) / np.max(mask)
        
        # Create illumination source
        source = self.create_illumination_source(mask.shape, pixel_size, source_type)
        
        # Create pupil function
        pupil = self.create_pupil_function(mask.shape, pixel_size)
        
        # Add defocus if specified
        if defocus != 0:
            pupil = self._add_defocus(pupil, defocus, pixel_size)
        
        # Partially coherent imaging (Hopkins approximation)
        aerial_image = self._hopkins_imaging(mask, source, pupil)
        
        return aerial_image
    
    def _add_defocus(self, pupil: np.ndarray, defocus: float,
                    pixel_size: float) -> np.ndarray:
        """
        Add defocus aberration to pupil function
        
        Args:
            pupil: Pupil function
            defocus: Defocus amount in nm
            pixel_size: Physical size of each pixel in nm
            
        Returns:
            Modified pupil function
        """
        ny, nx = pupil.shape
        
        # Create frequency coordinates
        fx = fftfreq(nx, d=pixel_size)
        fy = fftfreq(ny, d=pixel_size)
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        
        # Calculate defocus phase
        rho_squared = (fx_grid**2 + fy_grid**2) * (self.system.wavelength / self.system.NA)**2
        defocus_phase = 2 * np.pi * defocus / self.system.wavelength * np.sqrt(1 - rho_squared)
        
        # Apply only within pupil
        mask = np.abs(pupil) > 0
        defocus_phase = np.where(mask, defocus_phase, 0)
        
        # Apply defocus
        pupil_defocused = pupil * np.exp(1j * defocus_phase)
        
        return pupil_defocused
    
    def _hopkins_imaging(self, mask: np.ndarray, source: np.ndarray,
                        pupil: np.ndarray) -> np.ndarray:
        """
        Simulate partially coherent imaging using Hopkins method
        
        Args:
            mask: Input mask pattern
            source: Illumination source distribution
            pupil: Pupil function
            
        Returns:
            Aerial image intensity
        """
        # Fourier transform of mask
        mask_spectrum = fftshift(fft2(mask))
        
        # Initialize aerial image
        aerial_image = np.zeros_like(mask, dtype=np.float64)
        
        # Find source points
        source_points = np.where(source > 0)
        
        if len(source_points[0]) == 0:
            # No source points, return zero image
            return aerial_image
        
        # Sample source points (limit for computational efficiency)
        n_samples = min(100, len(source_points[0]))
        indices = np.random.choice(len(source_points[0]), n_samples, replace=False)
        
        # Sum over source points (partial coherence)
        for idx in indices:
            sy, sx = source_points[0][idx], source_points[1][idx]
            weight = source[sy, sx]
            
            # Shift pupil for oblique illumination
            pupil_shifted = np.roll(np.roll(pupil, sx, axis=1), sy, axis=0)
            
            # Image formation for this source point
            filtered_spectrum = mask_spectrum * pupil_shifted
            partial_image = ifft2(ifftshift(filtered_spectrum))
            aerial_image += weight * np.abs(partial_image)**2
        
        # Normalize
        if np.max(aerial_image) > 0:
            aerial_image = aerial_image / np.max(aerial_image)
        
        return aerial_image
    
    def simulate_focus_exposure_matrix(self, mask: np.ndarray,
                                      focus_range: np.ndarray,
                                      exposure_range: np.ndarray,
                                      pixel_size: float = 5.0,
                                      threshold: float = 0.3) -> np.ndarray:
        """
        Simulate focus-exposure matrix (Bossung curves)
        
        Args:
            mask: Input mask pattern
            focus_range: Array of focus values in nm
            exposure_range: Array of exposure dose values (relative)
            pixel_size: Physical size of each pixel in nm
            threshold: Resist threshold
            
        Returns:
            3D array of CD measurements [focus, exposure, measurement]
        """
        n_focus = len(focus_range)
        n_exposure = len(exposure_range)
        
        # Initialize CD matrix
        cd_matrix = np.zeros((n_focus, n_exposure))
        
        for i, focus in enumerate(focus_range):
            # Simulate aerial image at this focus
            aerial = self.simulate_image_formation(mask, pixel_size, defocus=focus)
            
            for j, exposure in enumerate(exposure_range):
                # Apply exposure dose
                exposed = aerial * exposure
                
                # Apply threshold
                resist = exposed > threshold
                
                # Measure CD (simplified: count pixels)
                cd = np.sum(resist) * pixel_size
                cd_matrix[i, j] = cd
        
        return cd_matrix
    
    def calculate_process_window(self, cd_matrix: np.ndarray,
                                target_cd: float,
                                tolerance: float = 0.1) -> Dict[str, float]:
        """
        Calculate process window from focus-exposure matrix
        
        Args:
            cd_matrix: CD measurements from focus-exposure matrix
            target_cd: Target CD value
            tolerance: Acceptable CD variation (fraction)
            
        Returns:
            Process window metrics
        """
        # Find region within tolerance
        min_cd = target_cd * (1 - tolerance)
        max_cd = target_cd * (1 + tolerance)
        
        within_spec = (cd_matrix >= min_cd) & (cd_matrix <= max_cd)
        
        if not np.any(within_spec):
            return {
                'depth_of_focus': 0.0,
                'exposure_latitude': 0.0,
                'process_window_area': 0.0
            }
        
        # Find focus and exposure ranges
        focus_indices, exposure_indices = np.where(within_spec)
        
        depth_of_focus = np.ptp(focus_indices) if len(focus_indices) > 0 else 0
        exposure_latitude = np.ptp(exposure_indices) if len(exposure_indices) > 0 else 0
        
        # Calculate process window area
        process_window_area = np.sum(within_spec)
        
        return {
            'depth_of_focus': depth_of_focus,
            'exposure_latitude': exposure_latitude,
            'process_window_area': process_window_area
        }
    
    def visualize_optical_system(self) -> None:
        """
        Visualize the optical system components
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Create components
        shape = (256, 256)
        pixel_size = 5.0
        
        # Illumination sources
        sources = ['conventional', 'annular', 'quadrupole']
        for i, source_type in enumerate(sources):
            source = self.create_illumination_source(shape, pixel_size, source_type)
            axes[0, i].imshow(fftshift(source), cmap='hot')
            axes[0, i].set_title(f'{source_type.capitalize()} Illumination')
            axes[0, i].axis('off')
        
        # Pupil function
        pupil = self.create_pupil_function(shape, pixel_size)
        axes[1, 0].imshow(np.abs(fftshift(pupil)), cmap='gray')
        axes[1, 0].set_title('Pupil Amplitude')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(np.angle(fftshift(pupil)), cmap='hsv')
        axes[1, 1].set_title('Pupil Phase')
        axes[1, 1].axis('off')
        
        # PSF
        psf = self.calculate_psf(shape, pixel_size)
        axes[1, 2].imshow(psf, cmap='hot')
        axes[1, 2].set_title('Point Spread Function')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Optical System (λ={self.system.wavelength}nm, NA={self.system.NA})')
        plt.tight_layout()
        plt.show()
    
    def visualize_imaging_results(self, mask: np.ndarray,
                                 pixel_size: float = 5.0) -> None:
        """
        Visualize imaging results with different illumination modes
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original mask
        axes[0, 0].imshow(mask, cmap='gray')
        axes[0, 0].set_title('Original Mask')
        axes[0, 0].axis('off')
        
        # Different illumination modes
        modes = ['conventional', 'annular', 'quadrupole']
        
        for i, mode in enumerate(modes):
            aerial = self.simulate_image_formation(mask, pixel_size, mode)
            axes[0, i+1].imshow(aerial, cmap='gray')
            axes[0, i+1].set_title(f'{mode.capitalize()} Illumination')
            axes[0, i+1].axis('off')
            
            # Thresholded image
            resist = aerial > 0.3
            axes[1, i].imshow(resist, cmap='gray')
            axes[1, i].set_title(f'{mode.capitalize()} (Thresholded)')
            axes[1, i].axis('off')
        
        plt.suptitle('Imaging Results with Different Illumination Modes')
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create optical system
    system = OpticalSystem(
        wavelength=193,
        NA=0.93,
        sigma_outer=0.7,
        sigma_inner=0.5,
        immersion=True
    )
    
    # Create simulator
    simulator = FourierOpticsSimulator(system)
    
    # Print system properties
    print(f"Resolution: {system.resolution:.1f} nm")
    print(f"Depth of Focus: {system.depth_of_focus:.1f} nm")
    print(f"k1 factor: {system.k1_factor:.3f}")
    
    # Create test pattern
    mask = np.zeros((512, 512), dtype=np.uint8)
    # Line-space pattern
    for i in range(0, 512, 40):
        mask[:, i:i+20] = 255
    
    # Simulate imaging
    aerial = simulator.simulate_image_formation(mask, pixel_size=2.5)
    
    # Calculate MTF
    freq, mtf = simulator.calculate_mtf()
    
    # Visualize
    simulator.visualize_optical_system()
    simulator.visualize_imaging_results(mask, pixel_size=2.5)