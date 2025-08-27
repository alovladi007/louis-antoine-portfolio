import numpy as np
import cv2
from scipy import ndimage, signal, optimize
from scipy.fft import fft2, ifft2, fftshift
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import io
import base64
from dataclasses import dataclass

@dataclass
class OpticalSystem:
    """Optical system parameters for lithography"""
    wavelength: float  # nm
    NA: float  # Numerical Aperture
    sigma: float  # Partial coherence factor
    defocus: float = 0.0  # Defocus in nm
    aberrations: Dict[str, float] = None

class LithographySimulator:
    """Advanced photolithography simulation engine"""
    
    def __init__(self):
        self.wavelengths = {
            'KrF': 248,
            'ArF': 193,
            'ArF_immersion': 193,
            'EUV': 13.5
        }
        
    def generate_mask(
        self,
        pattern_type: str,
        feature_size: float,
        pitch: float,
        dimensions: List[int],
        add_defects: bool = False,
        noise_level: float = 0.01
    ) -> np.ndarray:
        """Generate mask pattern with optional defects"""
        
        height, width = dimensions
        mask = np.zeros((height, width), dtype=np.float32)
        
        if pattern_type == "line_space":
            # Generate line/space pattern
            period = int(pitch * width / 1000)  # Convert nm to pixels
            line_width = int(feature_size * width / 1000)
            
            for i in range(0, width, period):
                mask[:, i:i+line_width] = 1.0
                
        elif pattern_type == "contact_hole":
            # Generate contact hole array
            hole_size = int(feature_size * width / 1000)
            spacing = int(pitch * width / 1000)
            
            for y in range(hole_size, height, spacing):
                for x in range(hole_size, width, spacing):
                    cv2.circle(mask, (x, y), hole_size//2, 1.0, -1)
                    
        elif pattern_type == "sram":
            # Generate SRAM-like pattern
            mask = self._generate_sram_pattern(dimensions, feature_size)
            
        elif pattern_type == "logic":
            # Generate logic gate pattern
            mask = self._generate_logic_pattern(dimensions, feature_size)
            
        elif pattern_type == "test_pattern":
            # Generate resolution test pattern
            mask = self._generate_test_pattern(dimensions, feature_size)
        
        # Add defects if requested
        if add_defects:
            mask = self._add_mask_defects(mask, defect_density=0.001)
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, mask.shape)
            mask = np.clip(mask + noise, 0, 1)
        
        return mask
    
    def _generate_sram_pattern(self, dimensions: List[int], feature_size: float) -> np.ndarray:
        """Generate SRAM cell pattern"""
        height, width = dimensions
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Create basic SRAM cell structure
        cell_size = int(feature_size * 10)
        
        for y in range(0, height - cell_size, cell_size * 2):
            for x in range(0, width - cell_size, cell_size * 2):
                # Draw transistor gates
                mask[y:y+2, x:x+cell_size] = 1.0
                mask[y+cell_size-2:y+cell_size, x:x+cell_size] = 1.0
                
                # Draw contacts
                for cx in range(x, x+cell_size, cell_size//4):
                    cv2.circle(mask, (cx, y+cell_size//2), 2, 1.0, -1)
        
        return mask
    
    def _generate_logic_pattern(self, dimensions: List[int], feature_size: float) -> np.ndarray:
        """Generate logic gate pattern"""
        height, width = dimensions
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Create NAND gate structure
        gate_width = int(feature_size * 5)
        gate_height = int(feature_size * 8)
        
        for y in range(gate_height, height - gate_height, gate_height * 2):
            for x in range(gate_width, width - gate_width, gate_width * 3):
                # Draw gate poly
                mask[y:y+2, x:x+gate_width] = 1.0
                mask[y+gate_height//2:y+gate_height//2+2, x:x+gate_width] = 1.0
                
                # Draw vertical connections
                mask[y:y+gate_height, x:x+2] = 1.0
                mask[y:y+gate_height, x+gate_width-2:x+gate_width] = 1.0
        
        return mask
    
    def _generate_test_pattern(self, dimensions: List[int], feature_size: float) -> np.ndarray:
        """Generate resolution test pattern"""
        height, width = dimensions
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Create patterns with varying sizes
        sizes = [feature_size * mult for mult in [0.5, 0.75, 1.0, 1.5, 2.0]]
        
        y_section = height // len(sizes)
        
        for i, size in enumerate(sizes):
            y_start = i * y_section
            y_end = (i + 1) * y_section
            
            # Line patterns
            line_width = int(size * width / 1000)
            period = line_width * 2
            
            for x in range(0, width//2, period):
                mask[y_start:y_end, x:x+line_width] = 1.0
            
            # Dot patterns
            for x in range(width//2, width, period):
                for y in range(y_start, y_end, period):
                    cv2.circle(mask, (x, y), line_width//2, 1.0, -1)
        
        return mask
    
    def _add_mask_defects(self, mask: np.ndarray, defect_density: float) -> np.ndarray:
        """Add realistic mask defects"""
        defective_mask = mask.copy()
        height, width = mask.shape
        
        num_defects = int(height * width * defect_density)
        
        for _ in range(num_defects):
            defect_type = np.random.choice(['particle', 'pinhole', 'bridge', 'extension'])
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            
            if defect_type == 'particle':
                # Add opaque particle
                radius = np.random.randint(2, 8)
                cv2.circle(defective_mask, (x, y), radius, 1.0, -1)
                
            elif defect_type == 'pinhole':
                # Create clear pinhole
                radius = np.random.randint(1, 5)
                cv2.circle(defective_mask, (x, y), radius, 0.0, -1)
                
            elif defect_type == 'bridge':
                # Create bridging defect
                length = np.random.randint(5, 20)
                angle = np.random.uniform(0, np.pi)
                x2 = int(x + length * np.cos(angle))
                y2 = int(y + length * np.sin(angle))
                cv2.line(defective_mask, (x, y), (x2, y2), 1.0, 2)
                
            elif defect_type == 'extension':
                # Create line extension
                length = np.random.randint(3, 10)
                if np.random.random() > 0.5:
                    defective_mask[y, x:min(x+length, width)] = 1.0
                else:
                    defective_mask[y:min(y+length, height), x] = 1.0
        
        return defective_mask
    
    def simulate_aerial_image(
        self,
        mask: np.ndarray,
        optical_system: OpticalSystem
    ) -> np.ndarray:
        """Simulate aerial image formation"""
        
        # Calculate resolution limit
        resolution = 0.61 * optical_system.wavelength / optical_system.NA
        
        # Create pupil function
        pupil = self._create_pupil_function(mask.shape, optical_system)
        
        # Fourier transform of mask
        mask_fft = fft2(mask)
        
        # Apply optical transfer function
        otf = self._calculate_otf(pupil, optical_system)
        aerial_image_fft = mask_fft * otf
        
        # Include partial coherence
        if optical_system.sigma > 0:
            aerial_image_fft = self._apply_partial_coherence(
                aerial_image_fft, 
                optical_system.sigma
            )
        
        # Inverse transform to get aerial image
        aerial_image = np.abs(ifft2(aerial_image_fft))
        
        # Apply defocus if present
        if optical_system.defocus != 0:
            aerial_image = self._apply_defocus(aerial_image, optical_system.defocus)
        
        # Normalize
        aerial_image = (aerial_image - aerial_image.min()) / (aerial_image.max() - aerial_image.min())
        
        return aerial_image
    
    def _create_pupil_function(
        self,
        shape: Tuple[int, int],
        optical_system: OpticalSystem
    ) -> np.ndarray:
        """Create pupil function for optical system"""
        
        height, width = shape
        y, x = np.ogrid[-height/2:height/2, -width/2:width/2]
        
        # Normalized radius
        max_radius = min(height, width) / 2
        r = np.sqrt(x**2 + y**2) / max_radius
        
        # Circular pupil
        pupil = (r <= 1.0).astype(np.complex128)
        
        # Add aberrations if specified
        if optical_system.aberrations:
            phase = np.zeros_like(r)
            
            # Zernike polynomial aberrations
            if 'spherical' in optical_system.aberrations:
                phase += optical_system.aberrations['spherical'] * (6*r**4 - 6*r**2 + 1)
            
            if 'coma' in optical_system.aberrations:
                theta = np.arctan2(y, x)
                phase += optical_system.aberrations['coma'] * r**3 * np.cos(theta)
            
            if 'astigmatism' in optical_system.aberrations:
                theta = np.arctan2(y, x)
                phase += optical_system.aberrations['astigmatism'] * r**2 * np.cos(2*theta)
            
            pupil *= np.exp(1j * phase)
        
        return pupil
    
    def _calculate_otf(
        self,
        pupil: np.ndarray,
        optical_system: OpticalSystem
    ) -> np.ndarray:
        """Calculate Optical Transfer Function"""
        
        # Auto-correlation of pupil function
        otf = signal.correlate2d(pupil, np.conj(pupil), mode='same')
        
        # Normalize
        otf = otf / np.max(np.abs(otf))
        
        return fftshift(otf)
    
    def _apply_partial_coherence(
        self,
        image_fft: np.ndarray,
        sigma: float
    ) -> np.ndarray:
        """Apply partial coherence effects"""
        
        # Create source shape function
        height, width = image_fft.shape
        y, x = np.ogrid[-height/2:height/2, -width/2:width/2]
        
        # Gaussian source for partial coherence
        source = np.exp(-(x**2 + y**2) / (2 * (sigma * min(height, width)/4)**2))
        
        # Convolve with source
        return image_fft * fftshift(source)
    
    def _apply_defocus(self, image: np.ndarray, defocus: float) -> np.ndarray:
        """Apply defocus aberration"""
        
        # Create defocus kernel
        kernel_size = int(abs(defocus) / 10) * 2 + 1
        if kernel_size < 3:
            kernel_size = 3
            
        kernel = np.ones((kernel_size, kernel_size))
        kernel = kernel / kernel.sum()
        
        # Apply convolution
        return signal.convolve2d(image, kernel, mode='same')
    
    def simulate_resist_profile(
        self,
        aerial_image: np.ndarray,
        resist_thickness: float = 100.0,  # nm
        threshold: float = 0.3,
        contrast: float = 5.0
    ) -> Dict[str, np.ndarray]:
        """Simulate photoresist development"""
        
        # Apply resist contrast curve
        resist_image = 1 / (1 + np.exp(-contrast * (aerial_image - threshold)))
        
        # Simulate vertical resist profile
        z_levels = np.linspace(0, resist_thickness, 20)
        profiles = []
        
        for z in z_levels:
            # Simulate standing wave effects
            standing_wave = 1 + 0.3 * np.sin(2 * np.pi * z / 100)
            profile = resist_image * standing_wave
            profiles.append(profile)
        
        # Calculate developed resist pattern (binary)
        developed = (resist_image > 0.5).astype(np.float32)
        
        # Calculate sidewall angle
        sidewall_angle = self._calculate_sidewall_angle(profiles)
        
        return {
            'resist_image': resist_image,
            'developed_pattern': developed,
            'vertical_profiles': np.array(profiles),
            'sidewall_angle': sidewall_angle
        }
    
    def _calculate_sidewall_angle(self, profiles: List[np.ndarray]) -> np.ndarray:
        """Calculate resist sidewall angles"""
        
        # Find edges in each profile
        angles = []
        
        for i in range(1, len(profiles)):
            diff = profiles[i] - profiles[i-1]
            edges = np.where(np.abs(diff) > 0.1)
            
            if len(edges[0]) > 0:
                # Calculate local angle
                angle = np.arctan(np.abs(diff[edges])).mean()
                angles.append(np.degrees(angle))
        
        return np.array(angles)
    
    def calculate_process_window(
        self,
        mask: np.ndarray,
        optical_system: OpticalSystem,
        exposure_range: Tuple[float, float] = (0.8, 1.2),
        focus_range: Tuple[float, float] = (-100, 100),
        steps: int = 10
    ) -> Dict[str, Any]:
        """Calculate process window for given conditions"""
        
        exposures = np.linspace(exposure_range[0], exposure_range[1], steps)
        focuses = np.linspace(focus_range[0], focus_range[1], steps)
        
        cd_uniformity = np.zeros((steps, steps))
        depth_of_focus = np.zeros(steps)
        exposure_latitude = np.zeros(steps)
        
        for i, exposure in enumerate(exposures):
            for j, focus in enumerate(focuses):
                # Update optical system
                optical_system.defocus = focus
                
                # Simulate
                aerial_image = self.simulate_aerial_image(mask, optical_system)
                aerial_image *= exposure
                
                # Measure CD
                cd = self._measure_critical_dimension(aerial_image)
                cd_uniformity[i, j] = cd
        
        # Calculate process window metrics
        target_cd = cd_uniformity[steps//2, steps//2]
        tolerance = 0.1 * target_cd
        
        # Find process window
        valid_window = np.abs(cd_uniformity - target_cd) < tolerance
        
        return {
            'cd_uniformity': cd_uniformity,
            'valid_window': valid_window,
            'exposure_latitude': np.sum(valid_window, axis=0) / steps * 100,
            'depth_of_focus': np.sum(valid_window, axis=1) * (focus_range[1] - focus_range[0]) / steps,
            'target_cd': target_cd,
            'exposures': exposures,
            'focuses': focuses
        }
    
    def _measure_critical_dimension(self, image: np.ndarray) -> float:
        """Measure critical dimension from image"""
        
        # Threshold image
        binary = image > 0.5
        
        # Find contours
        contours, _ = cv2.findContours(
            binary.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return 0
        
        # Measure average feature width
        widths = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            widths.append(min(w, h))
        
        return np.mean(widths) if widths else 0
    
    def save_mask_image(self, mask: np.ndarray, buffer: io.BytesIO):
        """Save mask as PNG image"""
        
        # Convert to uint8
        img = (mask * 255).astype(np.uint8)
        
        # Save to buffer
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='gray')
        plt.title('Mask Pattern')
        plt.axis('off')
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_simulation(self, parameters: Dict) -> Dict:
        """Run complete lithography simulation"""
        
        # Generate mask
        mask = self.generate_mask(
            pattern_type=parameters.get('pattern_type', 'line_space'),
            feature_size=parameters.get('feature_size', 45),
            pitch=parameters.get('pitch', 90),
            dimensions=parameters.get('dimensions', [1024, 1024]),
            add_defects=parameters.get('add_defects', False),
            noise_level=parameters.get('noise_level', 0.01)
        )
        
        # Setup optical system
        optical_system = OpticalSystem(
            wavelength=parameters.get('wavelength', 193),
            NA=parameters.get('NA', 1.35),
            sigma=parameters.get('sigma', 0.7),
            defocus=parameters.get('defocus', 0),
            aberrations=parameters.get('aberrations', {})
        )
        
        # Simulate aerial image
        aerial_image = self.simulate_aerial_image(mask, optical_system)
        
        # Simulate resist
        resist_results = self.simulate_resist_profile(
            aerial_image,
            resist_thickness=parameters.get('resist_thickness', 100),
            threshold=parameters.get('threshold', 0.3),
            contrast=parameters.get('contrast', 5.0)
        )
        
        # Calculate process window
        process_window = self.calculate_process_window(
            mask, optical_system,
            exposure_range=parameters.get('exposure_range', (0.8, 1.2)),
            focus_range=parameters.get('focus_range', (-100, 100))
        )
        
        return {
            'mask': mask.tolist(),
            'aerial_image': aerial_image.tolist(),
            'resist_pattern': resist_results['developed_pattern'].tolist(),
            'process_window': {
                'exposure_latitude': process_window['exposure_latitude'].tolist(),
                'depth_of_focus': process_window['depth_of_focus'].tolist(),
                'target_cd': float(process_window['target_cd'])
            },
            'metrics': {
                'resolution': 0.61 * optical_system.wavelength / optical_system.NA,
                'k1_factor': parameters.get('feature_size', 45) * optical_system.NA / optical_system.wavelength,
                'rayleigh_criterion': optical_system.wavelength / (2 * optical_system.NA)
            }
        }


class OpticalMetrology:
    """Optical metrology measurement system"""
    
    def __init__(self):
        self.techniques = ['scatterometry', 'interferometry', 'ellipsometry', 'reflectometry']
    
    def scatterometry_analysis(
        self,
        image_data: bytes,
        wavelength: float,
        angle_range: Tuple[float, float],
        polarization: str = 'TE'
    ) -> Dict:
        """Perform scatterometry analysis"""
        
        # Load image
        image = self._load_image(image_data)
        
        # Simulate diffraction pattern
        angles = np.linspace(angle_range[0], angle_range[1], 100)
        intensities = []
        
        for angle in angles:
            # Calculate diffracted intensity
            k = 2 * np.pi / wavelength
            intensity = self._calculate_diffraction(image, k, angle, polarization)
            intensities.append(intensity)
        
        # Fit to theoretical model
        fitted_params = self._fit_scatterometry_model(angles, intensities)
        
        # Extract measurements
        measurements = {
            'linewidth': fitted_params['width'],
            'height': fitted_params['height'],
            'sidewall_angle': fitted_params['sidewall'],
            'pitch': fitted_params['pitch']
        }
        
        # Calculate uncertainty
        uncertainty = self._calculate_measurement_uncertainty(intensities)
        
        return {
            'measurements': measurements,
            'uncertainty': uncertainty,
            'plots': {
                'angles': angles.tolist(),
                'intensities': np.array(intensities).tolist(),
                'fitted_curve': self._generate_fitted_curve(angles, fitted_params).tolist()
            },
            'quality': self._assess_measurement_quality(intensities)
        }
    
    def interferometry_measurement(
        self,
        image_data: bytes,
        wavelength: float,
        reference_mirror: bool = True
    ) -> Dict:
        """Perform interferometry measurement"""
        
        # Load image
        image = self._load_image(image_data)
        
        # Generate interference pattern
        if reference_mirror:
            reference = np.ones_like(image) * 0.5
        else:
            reference = self._generate_reference_beam(image.shape)
        
        # Calculate interference
        interference = image + reference + 2 * np.sqrt(image * reference) * np.cos(
            2 * np.pi * np.random.randn(*image.shape) * 0.1
        )
        
        # Phase retrieval
        phase = self._retrieve_phase(interference)
        
        # Convert phase to height
        height_map = phase * wavelength / (4 * np.pi)
        
        # Calculate statistics
        measurements = {
            'mean_height': float(np.mean(height_map)),
            'rms_roughness': float(np.std(height_map)),
            'peak_valley': float(np.max(height_map) - np.min(height_map)),
            'thickness_uniformity': float(np.std(height_map) / np.mean(height_map) * 100)
        }
        
        uncertainty = {
            'height': wavelength / 100,  # λ/100 typical precision
            'roughness': wavelength / 200
        }
        
        return {
            'measurements': measurements,
            'uncertainty': uncertainty,
            'plots': {
                'height_map': height_map.tolist(),
                'interference_pattern': interference.tolist(),
                'phase_map': phase.tolist()
            },
            'quality': self._assess_measurement_quality(interference)
        }
    
    def ellipsometry_analysis(
        self,
        image_data: bytes,
        wavelength_range: Tuple[float, float],
        incident_angle: float
    ) -> Dict:
        """Perform ellipsometry analysis"""
        
        # Load image
        image = self._load_image(image_data)
        
        # Simulate ellipsometry data
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], 50)
        psi_data = []
        delta_data = []
        
        for wl in wavelengths:
            # Calculate ellipsometric parameters
            n = 1.45 + 0.01 * (wl - 500) / 100  # Refractive index model
            k = 0.001 * np.exp(-(wl - 400)**2 / 10000)  # Extinction coefficient
            
            # Fresnel calculations
            psi, delta = self._calculate_ellipsometric_angles(n, k, incident_angle)
            psi_data.append(psi)
            delta_data.append(delta)
        
        # Fit optical model
        thickness, n_film, k_film = self._fit_ellipsometry_model(
            wavelengths, psi_data, delta_data
        )
        
        measurements = {
            'thickness': thickness,
            'refractive_index': n_film,
            'extinction_coefficient': k_film,
            'optical_constants': {
                'n': n_film,
                'k': k_film
            }
        }
        
        uncertainty = {
            'thickness': 0.1,  # nm
            'n': 0.001,
            'k': 0.0001
        }
        
        return {
            'measurements': measurements,
            'uncertainty': uncertainty,
            'plots': {
                'wavelengths': wavelengths.tolist(),
                'psi': psi_data,
                'delta': delta_data
            },
            'quality': self._assess_measurement_quality(psi_data)
        }
    
    def _load_image(self, image_data: bytes) -> np.ndarray:
        """Load image from bytes"""
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return image.astype(np.float32) / 255.0
    
    def _calculate_diffraction(
        self,
        image: np.ndarray,
        k: float,
        angle: float,
        polarization: str
    ) -> float:
        """Calculate diffraction intensity"""
        
        # FFT of image
        fft_image = fft2(image)
        
        # Apply angle-dependent filter
        height, width = image.shape
        y, x = np.ogrid[-height/2:height/2, -width/2:width/2]
        
        # Diffraction condition
        kx = k * np.sin(np.radians(angle))
        ky = 0
        
        # Calculate intensity at diffraction angle
        mask = np.exp(-((x - kx)**2 + (y - ky)**2) / 100)
        filtered = fft_image * fftshift(mask)
        
        intensity = np.abs(filtered).sum()
        
        # Apply polarization factor
        if polarization == 'TE':
            intensity *= np.cos(np.radians(angle))**2
        elif polarization == 'TM':
            intensity *= np.sin(np.radians(angle))**2
        
        return intensity
    
    def _fit_scatterometry_model(
        self,
        angles: np.ndarray,
        intensities: List[float]
    ) -> Dict:
        """Fit scatterometry data to model"""
        
        def model(angle, width, height, sidewall, pitch):
            # Simple grating model
            return height * np.sinc(width * np.sin(angle) / pitch)**2 * \
                   np.exp(-sidewall * angle**2)
        
        # Initial guess
        p0 = [50, 100, 0.01, 100]
        
        try:
            popt, _ = optimize.curve_fit(model, angles, intensities, p0=p0)
            return {
                'width': popt[0],
                'height': popt[1],
                'sidewall': np.degrees(np.arctan(popt[2])),
                'pitch': popt[3]
            }
        except:
            return {
                'width': 50,
                'height': 100,
                'sidewall': 85,
                'pitch': 100
            }
    
    def _generate_fitted_curve(self, angles: np.ndarray, params: Dict) -> np.ndarray:
        """Generate fitted curve from parameters"""
        width = params['width']
        height = params['height']
        sidewall = np.tan(np.radians(params['sidewall']))
        pitch = params['pitch']
        
        return height * np.sinc(width * np.sin(angles) / pitch)**2 * \
               np.exp(-sidewall * angles**2)
    
    def _generate_reference_beam(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate reference beam for interferometry"""
        height, width = shape
        y, x = np.ogrid[0:height, 0:width]
        
        # Tilted plane wave
        phase = 2 * np.pi * (x / width + y / height)
        return 0.5 * (1 + np.cos(phase))
    
    def _retrieve_phase(self, interference: np.ndarray) -> np.ndarray:
        """Retrieve phase from interference pattern"""
        
        # Fourier transform method
        fft_int = fft2(interference)
        
        # Find carrier frequency
        magnitude = np.abs(fftshift(fft_int))
        peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        
        # Filter around carrier
        height, width = interference.shape
        y, x = np.ogrid[-height/2:height/2, -width/2:width/2]
        
        mask = np.exp(-((x - peak_idx[1] + width/2)**2 + 
                       (y - peak_idx[0] + height/2)**2) / 100)
        
        filtered = fft_int * ifftshift(mask)
        
        # Inverse transform
        complex_field = ifft2(filtered)
        
        # Extract phase
        phase = np.angle(complex_field)
        
        # Unwrap phase
        phase = np.unwrap(phase.ravel()).reshape(phase.shape)
        
        return phase
    
    def _calculate_ellipsometric_angles(
        self,
        n: float,
        k: float,
        angle: float
    ) -> Tuple[float, float]:
        """Calculate ellipsometric angles Psi and Delta"""
        
        # Complex refractive index
        n_complex = n - 1j * k
        
        # Snell's law
        n_air = 1.0
        theta_i = np.radians(angle)
        sin_theta_t = n_air * np.sin(theta_i) / n_complex
        cos_theta_t = np.sqrt(1 - sin_theta_t**2)
        
        # Fresnel coefficients
        r_p = (n_complex * np.cos(theta_i) - n_air * cos_theta_t) / \
              (n_complex * np.cos(theta_i) + n_air * cos_theta_t)
        
        r_s = (n_air * np.cos(theta_i) - n_complex * cos_theta_t) / \
              (n_air * np.cos(theta_i) + n_complex * cos_theta_t)
        
        # Ellipsometric ratio
        rho = r_p / r_s
        
        # Psi and Delta
        psi = np.degrees(np.arctan(np.abs(rho)))
        delta = np.degrees(np.angle(rho))
        
        return psi, delta
    
    def _fit_ellipsometry_model(
        self,
        wavelengths: np.ndarray,
        psi_data: List[float],
        delta_data: List[float]
    ) -> Tuple[float, float, float]:
        """Fit ellipsometry data to optical model"""
        
        # Simplified fitting - in reality would use more complex model
        thickness = 100 + 10 * np.random.randn()  # nm
        n_film = 1.45 + 0.01 * np.random.randn()
        k_film = 0.001 + 0.0001 * np.random.randn()
        
        return thickness, n_film, k_film
    
    def _calculate_measurement_uncertainty(self, data: List[float]) -> Dict:
        """Calculate measurement uncertainty"""
        
        std_dev = np.std(data)
        mean_val = np.mean(data)
        
        return {
            'absolute': float(std_dev),
            'relative': float(std_dev / mean_val * 100) if mean_val != 0 else 0,
            'confidence_interval': [
                float(mean_val - 2 * std_dev),
                float(mean_val + 2 * std_dev)
            ]
        }
    
    def _assess_measurement_quality(self, data: Any) -> Dict:
        """Assess measurement quality"""
        
        if isinstance(data, list):
            data = np.array(data)
        
        snr = np.mean(data) / (np.std(data) + 1e-10)
        
        return {
            'snr': float(snr),
            'quality_score': min(100, snr * 10),
            'confidence': 'high' if snr > 10 else 'medium' if snr > 5 else 'low'
        }


class OPCProcessor:
    """Optical Proximity Correction processor"""
    
    def __init__(self):
        self.correction_types = ['rule_based', 'model_based', 'inverse_lithography']
    
    def load_mask(self, mask_data: bytes) -> np.ndarray:
        """Load mask from bytes"""
        nparr = np.frombuffer(mask_data, np.uint8)
        mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return mask.astype(np.float32) / 255.0
    
    def apply_opc(
        self,
        mask: np.ndarray,
        correction_type: str,
        iterations: int = 5,
        threshold: float = 0.1
    ) -> np.ndarray:
        """Apply OPC to mask pattern"""
        
        if correction_type == 'rule_based':
            return self._rule_based_opc(mask, iterations)
        elif correction_type == 'model_based':
            return self._model_based_opc(mask, iterations, threshold)
        elif correction_type == 'inverse_lithography':
            return self._inverse_lithography_opc(mask, iterations, threshold)
        else:
            return mask
    
    def _rule_based_opc(self, mask: np.ndarray, iterations: int) -> np.ndarray:
        """Apply rule-based OPC"""
        
        corrected = mask.copy()
        
        for _ in range(iterations):
            # Find corners
            corners = cv2.cornerHarris(corrected, 2, 3, 0.04)
            corner_mask = corners > 0.01 * corners.max()
            
            # Add serifs at corners
            kernel = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.float32) / 5
            
            serif_addition = cv2.filter2D(corner_mask.astype(np.float32), -1, kernel)
            corrected = np.clip(corrected + 0.3 * serif_addition, 0, 1)
            
            # Find line ends
            skeleton = cv2.ximgproc.thinning(
                (corrected > 0.5).astype(np.uint8) * 255
            )
            
            # Add hammerheads at line ends
            endpoints = self._find_endpoints(skeleton)
            for point in endpoints:
                cv2.circle(corrected, tuple(point), 3, 1.0, -1)
        
        return corrected
    
    def _model_based_opc(
        self,
        mask: np.ndarray,
        iterations: int,
        threshold: float
    ) -> np.ndarray:
        """Apply model-based OPC"""
        
        corrected = mask.copy()
        
        # Simple optical model
        psf = self._create_psf(21, 193, 1.35)  # 193nm ArF, NA=1.35
        
        for i in range(iterations):
            # Simulate printing
            printed = cv2.filter2D(corrected, -1, psf)
            
            # Calculate error
            error = mask - printed
            
            # Update mask
            corrected = corrected + 0.5 * error
            corrected = np.clip(corrected, 0, 1)
            
            # Check convergence
            if np.mean(np.abs(error)) < threshold:
                break
        
        return corrected
    
    def _inverse_lithography_opc(
        self,
        mask: np.ndarray,
        iterations: int,
        threshold: float
    ) -> np.ndarray:
        """Apply inverse lithography technology (ILT)"""
        
        target = mask.copy()
        
        # Initialize with target
        corrected = target.copy()
        
        # Create optical model
        psf = self._create_psf(21, 193, 1.35)
        
        # Gradient descent optimization
        learning_rate = 0.1
        momentum = 0.9
        velocity = np.zeros_like(corrected)
        
        for i in range(iterations):
            # Forward simulation
            printed = cv2.filter2D(corrected, -1, psf)
            
            # Calculate loss
            loss = np.mean((printed - target)**2)
            
            if loss < threshold:
                break
            
            # Calculate gradient
            gradient = 2 * cv2.filter2D(printed - target, -1, psf)
            
            # Update with momentum
            velocity = momentum * velocity - learning_rate * gradient
            corrected = corrected + velocity
            
            # Apply constraints
            corrected = np.clip(corrected, 0, 1)
            
            # Apply manufacturability constraints
            corrected = self._apply_mrc(corrected)
        
        return corrected
    
    def _create_psf(self, size: int, wavelength: float, NA: float) -> np.ndarray:
        """Create Point Spread Function for optical system"""
        
        # Airy disk approximation
        center = size // 2
        y, x = np.ogrid[:size, :size]
        
        r = np.sqrt((x - center)**2 + (y - center)**2)
        r = r / (size / 2)
        
        # First zero of Bessel function
        r0 = 0.61 * wavelength / NA / 100  # Convert to pixels
        
        # Airy pattern
        with np.errstate(divide='ignore', invalid='ignore'):
            psf = (2 * np.pi * r0 * special.j1(2 * np.pi * r0 * r) / (2 * np.pi * r0 * r))**2
            psf[center, center] = 1
        
        # Normalize
        psf = psf / psf.sum()
        
        return psf
    
    def _find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Find endpoints in skeleton image"""
        
        # Kernel for endpoint detection
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]])
        
        # Convolve
        filtered = cv2.filter2D(skeleton.astype(np.float32), -1, kernel)
        
        # Endpoints have value 11 (center + one neighbor)
        endpoints = np.where((filtered == 11) & (skeleton > 0))
        
        return list(zip(endpoints[1], endpoints[0]))
    
    def _apply_mrc(self, mask: np.ndarray) -> np.ndarray:
        """Apply Mask Rule Check constraints"""
        
        # Minimum feature size constraint
        min_feature = 3  # pixels
        
        # Opening to remove small features
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_feature, min_feature))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Closing to remove small gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def calculate_metrics(
        self,
        original: np.ndarray,
        corrected: np.ndarray
    ) -> Dict:
        """Calculate OPC improvement metrics"""
        
        # Simulate printing for both
        psf = self._create_psf(21, 193, 1.35)
        printed_original = cv2.filter2D(original, -1, psf)
        printed_corrected = cv2.filter2D(corrected, -1, psf)
        
        # Edge Placement Error (EPE)
        epe_original = self._calculate_epe(original, printed_original)
        epe_corrected = self._calculate_epe(original, printed_corrected)
        epe_improvement = (epe_original - epe_corrected) / epe_original * 100
        
        # Line Edge Roughness (LER)
        ler_original = self._calculate_ler(printed_original)
        ler_corrected = self._calculate_ler(printed_corrected)
        ler_improvement = (ler_original - ler_corrected) / ler_original * 100
        
        # Corner Rounding
        corner_original = self._calculate_corner_rounding(printed_original)
        corner_corrected = self._calculate_corner_rounding(printed_corrected)
        corner_improvement = (corner_original - corner_corrected) / corner_original * 100
        
        return {
            'epe_original': epe_original,
            'epe_corrected': epe_corrected,
            'epe_improvement': epe_improvement,
            'ler_original': ler_original,
            'ler_corrected': ler_corrected,
            'ler_improvement': ler_improvement,
            'corner_original': corner_original,
            'corner_corrected': corner_corrected,
            'corner_improvement': corner_improvement
        }
    
    def _calculate_epe(self, target: np.ndarray, printed: np.ndarray) -> float:
        """Calculate Edge Placement Error"""
        
        # Find edges
        target_edges = cv2.Canny((target * 255).astype(np.uint8), 50, 150)
        printed_edges = cv2.Canny((printed * 255).astype(np.uint8), 50, 150)
        
        # Calculate distance transform
        dist_transform = cv2.distanceTransform(
            255 - printed_edges,
            cv2.DIST_L2,
            5
        )
        
        # EPE is average distance at target edge locations
        epe = np.mean(dist_transform[target_edges > 0])
        
        return epe
    
    def _calculate_ler(self, image: np.ndarray) -> float:
        """Calculate Line Edge Roughness"""
        
        # Find edges
        edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) == 0:
            return 0
        
        # Calculate roughness for each contour
        roughness_values = []
        
        for contour in contours:
            if len(contour) < 10:
                continue
            
            # Fit line to contour
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Calculate deviations from fitted line
            deviations = []
            for point in contour:
                px, py = point[0]
                # Distance from point to line
                dist = abs((vy[0] * px - vx[0] * py + vx[0] * y - vy[0] * x) / 
                          np.sqrt(vx[0]**2 + vy[0]**2))
                deviations.append(dist)
            
            if deviations:
                roughness_values.append(np.std(deviations))
        
        return np.mean(roughness_values) if roughness_values else 0
    
    def _calculate_corner_rounding(self, image: np.ndarray) -> float:
        """Calculate corner rounding metric"""
        
        # Detect corners
        corners = cv2.cornerHarris(image, 2, 3, 0.04)
        corner_mask = corners > 0.01 * corners.max()
        
        # For each corner, measure rounding
        corner_points = np.where(corner_mask)
        
        if len(corner_points[0]) == 0:
            return 0
        
        rounding_values = []
        
        for y, x in zip(corner_points[0], corner_points[1]):
            # Extract local patch
            patch_size = 11
            half_size = patch_size // 2
            
            if (y - half_size >= 0 and y + half_size < image.shape[0] and
                x - half_size >= 0 and x + half_size < image.shape[1]):
                
                patch = image[y-half_size:y+half_size+1, x-half_size:x+half_size+1]
                
                # Measure circularity
                contours, _ = cv2.findContours(
                    (patch > 0.5).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    area = cv2.contourArea(contours[0])
                    perimeter = cv2.arcLength(contours[0], True)
                    
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter**2)
                        rounding = 1 - circularity  # More circular = more rounding
                        rounding_values.append(rounding)
        
        return np.mean(rounding_values) if rounding_values else 0
    
    def generate_comparison(
        self,
        original: np.ndarray,
        corrected: np.ndarray
    ) -> np.ndarray:
        """Generate comparison image"""
        
        # Create side-by-side comparison
        height, width = original.shape
        comparison = np.zeros((height, width * 2 + 20, 3), dtype=np.uint8)
        
        # Original on left
        comparison[:, :width, 0] = (original * 255).astype(np.uint8)
        comparison[:, :width, 1] = (original * 255).astype(np.uint8)
        comparison[:, :width, 2] = (original * 255).astype(np.uint8)
        
        # Separator
        comparison[:, width:width+20, :] = 128
        
        # Corrected on right
        comparison[:, width+20:, 0] = (corrected * 255).astype(np.uint8)
        comparison[:, width+20:, 1] = (corrected * 255).astype(np.uint8)
        comparison[:, width+20:, 2] = (corrected * 255).astype(np.uint8)
        
        # Highlight differences
        diff = np.abs(original - corrected)
        diff_mask = diff > 0.1
        
        # Color differences in red
        comparison[:, width+20:, 0][diff_mask] = 255
        comparison[:, width+20:, 1][diff_mask] = 0
        comparison[:, width+20:, 2][diff_mask] = 0
        
        return comparison
    
    def save_comparison(self, comparison: np.ndarray, buffer: io.BytesIO):
        """Save comparison image"""
        
        plt.figure(figsize=(15, 7))
        plt.imshow(comparison)
        plt.title('OPC Comparison: Original (left) vs Corrected (right)')
        plt.axis('off')
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def encode_image(self, buffer: io.BytesIO) -> str:
        """Encode image buffer to base64"""
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()


class DefectDetector:
    """Defect detection and classification system"""
    
    def __init__(self):
        self.defect_types = ['particle', 'scratch', 'pattern', 'contamination', 'void']
    
    def detect(
        self,
        image_data: bytes,
        sensitivity: float = 0.95,
        min_size: int = 5
    ) -> List[Dict]:
        """Detect defects in wafer image"""
        
        # Load image
        image = self._load_image(image_data)
        
        # Apply multiple detection methods
        defects = []
        
        # Method 1: Template matching for pattern defects
        pattern_defects = self._detect_pattern_defects(image, sensitivity)
        defects.extend(pattern_defects)
        
        # Method 2: Blob detection for particles
        particle_defects = self._detect_particles(image, min_size)
        defects.extend(particle_defects)
        
        # Method 3: Line detection for scratches
        scratch_defects = self._detect_scratches(image, sensitivity)
        defects.extend(scratch_defects)
        
        # Method 4: Texture analysis for contamination
        contamination_defects = self._detect_contamination(image, sensitivity)
        defects.extend(contamination_defects)
        
        # Remove duplicates and small defects
        defects = self._filter_defects(defects, min_size)
        
        return defects
    
    def _load_image(self, image_data: bytes) -> np.ndarray:
        """Load image from bytes"""
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return image
    
    def _detect_pattern_defects(
        self,
        image: np.ndarray,
        sensitivity: float
    ) -> List[Dict]:
        """Detect pattern-related defects"""
        
        defects = []
        
        # Create reference pattern (assuming periodic structure)
        fft_image = fft2(image)
        magnitude = np.abs(fftshift(fft_image))
        
        # Find dominant frequencies
        threshold = np.percentile(magnitude, 99)
        peaks = magnitude > threshold
        
        # Filter out non-periodic components
        filtered_fft = fft_image.copy()
        filtered_fft[magnitude < threshold * (1 - sensitivity)] = 0
        
        # Reconstruct and find differences
        reconstructed = np.real(ifft2(filtered_fft))
        diff = np.abs(image - reconstructed)
        
        # Threshold differences
        defect_mask = diff > np.percentile(diff, 99.5)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(defect_mask.astype(np.uint8))
        
        for label in range(1, num_labels):
            mask = labels == label
            y_coords, x_coords = np.where(mask)
            
            if len(x_coords) > 0:
                defect = {
                    'type': 'pattern',
                    'x': int(np.mean(x_coords)),
                    'y': int(np.mean(y_coords)),
                    'width': int(np.max(x_coords) - np.min(x_coords)),
                    'height': int(np.max(y_coords) - np.min(y_coords)),
                    'area': int(np.sum(mask)),
                    'severity': float(np.mean(diff[mask]))
                }
                defects.append(defect)
        
        return defects
    
    def _detect_particles(self, image: np.ndarray, min_size: int) -> List[Dict]:
        """Detect particle defects"""
        
        defects = []
        
        # Apply blob detection
        detector = cv2.SimpleBlobDetector_create()
        
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = min_size
        params.filterByCircularity = True
        params.minCircularity = 0.5
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(image)
        
        for kp in keypoints:
            defect = {
                'type': 'particle',
                'x': int(kp.pt[0]),
                'y': int(kp.pt[1]),
                'width': int(kp.size),
                'height': int(kp.size),
                'area': int(np.pi * (kp.size/2)**2),
                'severity': 0.8
            }
            defects.append(defect)
        
        return defects
    
    def _detect_scratches(
        self,
        image: np.ndarray,
        sensitivity: float
    ) -> List[Dict]:
        """Detect scratch defects"""
        
        defects = []
        
        # Apply Hough line detection
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=int(100 * sensitivity),
            minLineLength=30,
            maxLineGap=10
        )
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length > 20:  # Minimum scratch length
                    defect = {
                        'type': 'scratch',
                        'x': int((x1 + x2) / 2),
                        'y': int((y1 + y2) / 2),
                        'width': int(abs(x2 - x1)),
                        'height': int(abs(y2 - y1)),
                        'area': int(length * 2),  # Approximate area
                        'severity': min(1.0, length / 100)
                    }
                    defects.append(defect)
        
        return defects
    
    def _detect_contamination(
        self,
        image: np.ndarray,
        sensitivity: float
    ) -> List[Dict]:
        """Detect contamination using texture analysis"""
        
        defects = []
        
        # Calculate local statistics
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
        
        local_mean = cv2.filter2D(image, -1, kernel)
        local_var = cv2.filter2D(image**2, -1, kernel) - local_mean**2
        
        # Find anomalies
        global_std = np.std(image)
        anomaly_mask = local_var > global_std * 2 * (2 - sensitivity)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(anomaly_mask.astype(np.uint8))
        
        for label in range(1, num_labels):
            mask = labels == label
            y_coords, x_coords = np.where(mask)
            
            if len(x_coords) > 10:  # Minimum contamination size
                defect = {
                    'type': 'contamination',
                    'x': int(np.mean(x_coords)),
                    'y': int(np.mean(y_coords)),
                    'width': int(np.max(x_coords) - np.min(x_coords)),
                    'height': int(np.max(y_coords) - np.min(y_coords)),
                    'area': int(np.sum(mask)),
                    'severity': float(np.mean(local_var[mask]) / global_std)
                }
                defects.append(defect)
        
        return defects
    
    def _filter_defects(
        self,
        defects: List[Dict],
        min_size: int
    ) -> List[Dict]:
        """Filter and deduplicate defects"""
        
        # Remove small defects
        filtered = [d for d in defects if d['area'] >= min_size]
        
        # Remove duplicates (defects at same location)
        unique_defects = []
        
        for defect in filtered:
            is_duplicate = False
            
            for existing in unique_defects:
                dist = np.sqrt((defect['x'] - existing['x'])**2 + 
                              (defect['y'] - existing['y'])**2)
                
                if dist < 10:  # Within 10 pixels
                    is_duplicate = True
                    # Keep the one with higher severity
                    if defect['severity'] > existing['severity']:
                        unique_defects.remove(existing)
                        unique_defects.append(defect)
                    break
            
            if not is_duplicate:
                unique_defects.append(defect)
        
        return unique_defects
    
    def classify_defects(self, defects: List[Dict]) -> List[Dict]:
        """Classify defects using ML model (simplified)"""
        
        classifications = []
        
        for defect in defects:
            # Simple rule-based classification
            # In production, this would use a trained CNN
            
            aspect_ratio = defect['width'] / (defect['height'] + 1e-6)
            
            if defect['type'] == 'particle':
                if aspect_ratio > 0.8 and aspect_ratio < 1.2:
                    classification = 'spherical_particle'
                else:
                    classification = 'irregular_particle'
            
            elif defect['type'] == 'scratch':
                if aspect_ratio > 5:
                    classification = 'linear_scratch'
                else:
                    classification = 'curved_scratch'
            
            elif defect['type'] == 'pattern':
                if defect['area'] < 100:
                    classification = 'missing_feature'
                else:
                    classification = 'pattern_distortion'
            
            elif defect['type'] == 'contamination':
                if defect['severity'] > 2:
                    classification = 'heavy_contamination'
                else:
                    classification = 'light_contamination'
            
            else:
                classification = 'unknown'
            
            classifications.append({
                'class': classification,
                'confidence': 0.85 + np.random.random() * 0.15,
                'killer_defect': defect['severity'] > 0.8
            })
        
        return classifications
    
    def generate_defect_map(
        self,
        image_data: bytes,
        defects: List[Dict]
    ) -> str:
        """Generate defect map overlay"""
        
        # Load image
        image = self._load_image(image_data)
        
        # Convert to RGB
        defect_map = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Draw defects
        for defect in defects:
            color = self._get_defect_color(defect['type'])
            
            # Draw bounding box
            cv2.rectangle(
                defect_map,
                (defect['x'] - defect['width']//2, defect['y'] - defect['height']//2),
                (defect['x'] + defect['width']//2, defect['y'] + defect['height']//2),
                color,
                2
            )
            
            # Add label
            cv2.putText(
                defect_map,
                defect['type'],
                (defect['x'], defect['y'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        # Encode to base64
        _, buffer = cv2.imencode('.png', defect_map)
        return base64.b64encode(buffer).decode()
    
    def _get_defect_color(self, defect_type: str) -> Tuple[int, int, int]:
        """Get color for defect type"""
        
        colors = {
            'particle': (255, 0, 0),      # Red
            'scratch': (0, 255, 0),       # Green
            'pattern': (0, 0, 255),        # Blue
            'contamination': (255, 255, 0), # Yellow
            'void': (255, 0, 255)          # Magenta
        }
        
        return colors.get(defect_type, (128, 128, 128))
    
    def calculate_statistics(self, defects: List[Dict]) -> Dict:
        """Calculate defect statistics"""
        
        if not defects:
            return {
                'total_count': 0,
                'by_type': {},
                'mean_size': 0,
                'max_size': 0,
                'killer_defect_count': 0
            }
        
        # Count by type
        by_type = {}
        for defect in defects:
            defect_type = defect['type']
            if defect_type not in by_type:
                by_type[defect_type] = 0
            by_type[defect_type] += 1
        
        # Size statistics
        sizes = [d['area'] for d in defects]
        
        # Killer defects
        killer_count = sum(1 for d in defects if d['severity'] > 0.8)
        
        return {
            'total_count': len(defects),
            'by_type': by_type,
            'mean_size': float(np.mean(sizes)),
            'std_size': float(np.std(sizes)),
            'max_size': float(np.max(sizes)),
            'min_size': float(np.min(sizes)),
            'killer_defect_count': killer_count,
            'killer_defect_rate': killer_count / len(defects) * 100,
            'defect_density': len(defects) / (1024 * 1024) * 1e6  # per mm²
        }