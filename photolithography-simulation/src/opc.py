"""
Optical Proximity Correction (OPC) Module

This module implements various OPC techniques to compensate for optical
diffraction effects in photolithography.
"""

import numpy as np
import cv2
from scipy import ndimage, optimize
from scipy.signal import convolve2d
from typing import Tuple, List, Optional, Dict, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class OPCParameters:
    """Parameters for OPC simulation"""
    wavelength: float = 193.0  # nm
    NA: float = 0.75  # Numerical aperture
    sigma: float = 0.5  # Partial coherence
    k1: float = 0.4  # Process factor
    threshold: float = 0.3  # Threshold for resist
    iterations: int = 10  # Number of OPC iterations
    
    @property
    def resolution(self) -> float:
        """Calculate resolution"""
        return self.k1 * self.wavelength / self.NA


class OpticalProximityCorrection:
    """
    Implement Optical Proximity Correction techniques
    """
    
    def __init__(self, params: Optional[OPCParameters] = None):
        """
        Initialize OPC processor
        
        Args:
            params: OPC parameters
        """
        self.params = params or OPCParameters()
        self.original_mask = None
        self.corrected_mask = None
        self.aerial_image = None
        
    def simulate_aerial_image(self, mask: np.ndarray, 
                             pixel_size: float = 5.0) -> np.ndarray:
        """
        Simulate aerial image formation through optical system
        
        Args:
            mask: Input mask pattern
            pixel_size: Physical size of each pixel in nm
            
        Returns:
            Aerial image intensity distribution
        """
        # Create pupil function
        pupil = self._create_pupil_function(mask.shape, pixel_size)
        
        # Fourier transform of mask
        mask_ft = np.fft.fft2(mask.astype(np.float64))
        
        # Apply pupil function (band-limiting)
        filtered_ft = mask_ft * pupil
        
        # Inverse Fourier transform to get aerial image
        aerial_image = np.abs(np.fft.ifft2(filtered_ft))**2
        
        # Normalize
        aerial_image = aerial_image / np.max(aerial_image)
        
        self.aerial_image = aerial_image
        return aerial_image
    
    def _create_pupil_function(self, shape: Tuple[int, int], 
                               pixel_size: float) -> np.ndarray:
        """Create pupil function for optical system"""
        ny, nx = shape
        
        # Frequency coordinates
        fx = np.fft.fftfreq(nx, d=pixel_size)
        fy = np.fft.fftfreq(ny, d=pixel_size)
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        
        # Radial frequency
        f_radial = np.sqrt(fx_grid**2 + fy_grid**2)
        
        # Cutoff frequency
        f_cutoff = self.params.NA / self.params.wavelength
        
        # Pupil function (circular aperture)
        pupil = np.where(f_radial <= f_cutoff, 1.0, 0.0)
        
        # Apply partial coherence
        if self.params.sigma < 1.0:
            # Simplified partial coherence model
            coherence_filter = np.exp(-(f_radial / (self.params.sigma * f_cutoff))**2)
            pupil = pupil * coherence_filter
        
        return pupil
    
    def apply_rule_based_opc(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply rule-based OPC corrections
        
        Args:
            mask: Input mask pattern
            
        Returns:
            Corrected mask
        """
        self.original_mask = mask.copy()
        corrected = mask.copy()
        
        # Apply various correction rules
        corrected = self._add_serifs(corrected)
        corrected = self._add_assist_features(corrected)
        corrected = self._apply_line_end_correction(corrected)
        corrected = self._apply_corner_correction(corrected)
        
        self.corrected_mask = corrected
        return corrected
    
    def _add_serifs(self, mask: np.ndarray, serif_size: int = 10) -> np.ndarray:
        """Add serifs to line ends and corners"""
        corrected = mask.copy()
        
        # Detect corners using Harris corner detection
        corners = cv2.cornerHarris(mask.astype(np.float32), 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        
        # Add serifs at corner locations
        corner_points = np.where(corners > 0.01 * corners.max())
        
        for y, x in zip(corner_points[0], corner_points[1]):
            # Add small serif squares at corners
            y_start = max(0, y - serif_size // 2)
            y_end = min(mask.shape[0], y + serif_size // 2)
            x_start = max(0, x - serif_size // 2)
            x_end = min(mask.shape[1], x + serif_size // 2)
            
            if mask[y, x] > 0:
                corrected[y_start:y_end, x_start:x_end] = 255
        
        return corrected
    
    def _add_assist_features(self, mask: np.ndarray, 
                            min_spacing: int = 50) -> np.ndarray:
        """Add sub-resolution assist features (SRAFs)"""
        corrected = mask.copy()
        
        # Find edges of features
        edges = cv2.Canny(mask, 50, 150)
        
        # Create distance transform
        dist_transform = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        
        # Add assist features at specific distances from main features
        assist_distance = min_spacing
        assist_width = 15  # Sub-resolution width
        
        # Find locations for assist features
        assist_mask = np.logical_and(
            dist_transform > assist_distance,
            dist_transform < assist_distance + assist_width
        )
        
        # Add assist features
        corrected[assist_mask] = 128  # Use different value for visualization
        
        return corrected
    
    def _apply_line_end_correction(self, mask: np.ndarray, 
                                   extension: int = 20) -> np.ndarray:
        """Apply line end shortening correction"""
        corrected = mask.copy()
        
        # Detect line ends using morphological operations
        kernel_h = np.array([[0, 0, 0],
                            [1, 1, 0],
                            [0, 0, 0]], dtype=np.uint8)
        kernel_v = np.array([[0, 1, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=np.uint8)
        
        # Find horizontal line ends
        h_ends_r = cv2.morphologyEx(mask, cv2.MORPH_HITMISS, kernel_h)
        h_ends_l = cv2.morphologyEx(mask, cv2.MORPH_HITMISS, np.flip(kernel_h, 1))
        
        # Find vertical line ends
        v_ends_b = cv2.morphologyEx(mask, cv2.MORPH_HITMISS, kernel_v)
        v_ends_t = cv2.morphologyEx(mask, cv2.MORPH_HITMISS, np.flip(kernel_v, 0))
        
        # Extend line ends
        if np.any(h_ends_r):
            points = np.where(h_ends_r > 0)
            for y, x in zip(points[0], points[1]):
                x_end = min(mask.shape[1], x + extension)
                corrected[y, x:x_end] = 255
        
        if np.any(h_ends_l):
            points = np.where(h_ends_l > 0)
            for y, x in zip(points[0], points[1]):
                x_start = max(0, x - extension)
                corrected[y, x_start:x+1] = 255
        
        if np.any(v_ends_b):
            points = np.where(v_ends_b > 0)
            for y, x in zip(points[0], points[1]):
                y_end = min(mask.shape[0], y + extension)
                corrected[y:y_end, x] = 255
        
        if np.any(v_ends_t):
            points = np.where(v_ends_t > 0)
            for y, x in zip(points[0], points[1]):
                y_start = max(0, y - extension)
                corrected[y_start:y+1, x] = 255
        
        return corrected
    
    def _apply_corner_correction(self, mask: np.ndarray) -> np.ndarray:
        """Apply corner rounding correction"""
        corrected = mask.copy()
        
        # Detect corners
        corners = cv2.goodFeaturesToTrack(
            mask.astype(np.float32), 
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=10
        )
        
        if corners is not None:
            corners = corners.reshape(-1, 2).astype(int)
            
            # Add corner compensation
            for corner in corners:
                x, y = corner
                # Add small squares at corners to compensate for rounding
                size = 8
                y_start = max(0, y - size // 2)
                y_end = min(mask.shape[0], y + size // 2)
                x_start = max(0, x - size // 2)
                x_end = min(mask.shape[1], x + size // 2)
                
                if mask[y, x] > 0:
                    corrected[y_start:y_end, x_start:x_end] = 255
        
        return corrected
    
    def apply_model_based_opc(self, mask: np.ndarray, 
                             target_pattern: Optional[np.ndarray] = None,
                             pixel_size: float = 5.0) -> np.ndarray:
        """
        Apply model-based OPC using iterative optimization
        
        Args:
            mask: Input mask pattern
            target_pattern: Desired pattern on wafer
            pixel_size: Physical size of each pixel in nm
            
        Returns:
            Optimized mask
        """
        self.original_mask = mask.copy()
        
        if target_pattern is None:
            target_pattern = mask.copy()
        
        corrected = mask.astype(np.float64)
        
        for iteration in range(self.params.iterations):
            # Simulate aerial image
            aerial = self.simulate_aerial_image(corrected, pixel_size)
            
            # Apply threshold to get resist pattern
            resist = (aerial > self.params.threshold).astype(np.float64) * 255
            
            # Calculate error
            error = target_pattern - resist
            
            # Update mask based on error
            gradient = self._calculate_gradient(corrected, error, pixel_size)
            corrected = corrected + 0.1 * gradient
            
            # Clip values
            corrected = np.clip(corrected, 0, 255)
            
            # Calculate cost
            cost = np.mean(np.abs(error))
            
            if iteration % 2 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.4f}")
            
            if cost < 0.01:
                break
        
        self.corrected_mask = corrected.astype(np.uint8)
        return self.corrected_mask
    
    def _calculate_gradient(self, mask: np.ndarray, error: np.ndarray,
                           pixel_size: float) -> np.ndarray:
        """Calculate gradient for mask optimization"""
        # Simplified gradient calculation
        # In practice, this would involve the derivative of the imaging model
        
        # Smooth the error
        smoothed_error = ndimage.gaussian_filter(error, sigma=2)
        
        # Calculate gradient based on error direction
        gradient = smoothed_error * 0.5
        
        # Apply constraints to maintain mask manufacturability
        gradient = self._apply_manufacturing_constraints(gradient)
        
        return gradient
    
    def _apply_manufacturing_constraints(self, gradient: np.ndarray) -> np.ndarray:
        """Apply manufacturing constraints to gradient"""
        # Minimum feature size constraint
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
        gradient = convolve2d(gradient, kernel, mode='same')
        
        return gradient
    
    def apply_inverse_lithography(self, target_pattern: np.ndarray,
                                 pixel_size: float = 5.0) -> np.ndarray:
        """
        Apply Inverse Lithography Technology (ILT)
        
        Args:
            target_pattern: Desired pattern on wafer
            pixel_size: Physical size of each pixel in nm
            
        Returns:
            Optimized mask
        """
        # Initialize with target pattern
        mask = target_pattern.astype(np.float64)
        
        def cost_function(mask_flat):
            mask = mask_flat.reshape(target_pattern.shape)
            aerial = self.simulate_aerial_image(mask, pixel_size)
            resist = (aerial > self.params.threshold).astype(np.float64) * 255
            return np.sum((target_pattern - resist)**2)
        
        # Flatten for optimization
        mask_flat = mask.flatten()
        
        # Optimize
        result = optimize.minimize(
            cost_function,
            mask_flat,
            method='L-BFGS-B',
            bounds=[(0, 255)] * len(mask_flat),
            options={'maxiter': 50}
        )
        
        # Reshape result
        optimized_mask = result.x.reshape(target_pattern.shape)
        
        self.corrected_mask = optimized_mask.astype(np.uint8)
        return self.corrected_mask
    
    def evaluate_correction(self, pixel_size: float = 5.0) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of OPC
        
        Args:
            pixel_size: Physical size of each pixel in nm
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.original_mask is None or self.corrected_mask is None:
            raise ValueError("Must apply OPC before evaluation")
        
        # Simulate aerial images
        original_aerial = self.simulate_aerial_image(self.original_mask, pixel_size)
        corrected_aerial = self.simulate_aerial_image(self.corrected_mask, pixel_size)
        
        # Apply threshold
        original_resist = (original_aerial > self.params.threshold).astype(np.uint8) * 255
        corrected_resist = (corrected_aerial > self.params.threshold).astype(np.uint8) * 255
        
        # Calculate metrics
        metrics = {
            'original_critical_dimension': self._measure_cd(original_resist),
            'corrected_critical_dimension': self._measure_cd(corrected_resist),
            'original_edge_placement_error': self._calculate_epe(
                self.original_mask, original_resist),
            'corrected_edge_placement_error': self._calculate_epe(
                self.original_mask, corrected_resist),
            'pattern_fidelity': self._calculate_pattern_fidelity(
                self.original_mask, corrected_resist),
            'mask_complexity': self._calculate_mask_complexity(self.corrected_mask)
        }
        
        return metrics
    
    def _measure_cd(self, pattern: np.ndarray) -> float:
        """Measure critical dimension"""
        # Find horizontal lines
        horizontal_projection = np.sum(pattern, axis=1)
        lines = np.where(horizontal_projection > pattern.shape[1] * 0.5)[0]
        
        if len(lines) > 1:
            line_widths = []
            for line in lines:
                line_pattern = pattern[line, :]
                edges = np.where(np.diff(line_pattern) != 0)[0]
                if len(edges) >= 2:
                    line_widths.append(edges[1] - edges[0])
            
            if line_widths:
                return np.mean(line_widths)
        
        return 0.0
    
    def _calculate_epe(self, target: np.ndarray, actual: np.ndarray) -> float:
        """Calculate Edge Placement Error"""
        # Find edges
        target_edges = cv2.Canny(target, 50, 150)
        actual_edges = cv2.Canny(actual, 50, 150)
        
        # Calculate distance transform
        dist_transform = cv2.distanceTransform(
            255 - actual_edges, cv2.DIST_L2, 5)
        
        # Calculate EPE at target edge locations
        epe_values = dist_transform[target_edges > 0]
        
        if len(epe_values) > 0:
            return np.mean(epe_values)
        return 0.0
    
    def _calculate_pattern_fidelity(self, target: np.ndarray, 
                                   actual: np.ndarray) -> float:
        """Calculate pattern fidelity (similarity)"""
        intersection = np.logical_and(target > 0, actual > 0).sum()
        union = np.logical_or(target > 0, actual > 0).sum()
        
        if union > 0:
            return intersection / union
        return 0.0
    
    def _calculate_mask_complexity(self, mask: np.ndarray) -> float:
        """Calculate mask complexity metric"""
        # Edge density as complexity metric
        edges = cv2.Canny(mask, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return edge_density
    
    def visualize_opc_results(self, pixel_size: float = 5.0) -> None:
        """Visualize OPC results"""
        if self.original_mask is None or self.corrected_mask is None:
            raise ValueError("Must apply OPC before visualization")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original mask
        axes[0, 0].imshow(self.original_mask, cmap='gray')
        axes[0, 0].set_title('Original Mask')
        axes[0, 0].axis('off')
        
        # Corrected mask
        axes[0, 1].imshow(self.corrected_mask, cmap='gray')
        axes[0, 1].set_title('Corrected Mask (with OPC)')
        axes[0, 1].axis('off')
        
        # Difference
        diff = np.abs(self.corrected_mask.astype(float) - 
                     self.original_mask.astype(float))
        axes[0, 2].imshow(diff, cmap='hot')
        axes[0, 2].set_title('OPC Corrections')
        axes[0, 2].axis('off')
        
        # Aerial images
        original_aerial = self.simulate_aerial_image(self.original_mask, pixel_size)
        corrected_aerial = self.simulate_aerial_image(self.corrected_mask, pixel_size)
        
        axes[1, 0].imshow(original_aerial, cmap='gray')
        axes[1, 0].set_title('Original Aerial Image')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(corrected_aerial, cmap='gray')
        axes[1, 1].set_title('Corrected Aerial Image')
        axes[1, 1].axis('off')
        
        # Resist patterns
        original_resist = (original_aerial > self.params.threshold).astype(np.uint8)
        corrected_resist = (corrected_aerial > self.params.threshold).astype(np.uint8)
        
        comparison = np.zeros((*original_resist.shape, 3))
        comparison[:, :, 0] = original_resist  # Red: original
        comparison[:, :, 1] = corrected_resist  # Green: corrected
        comparison[:, :, 2] = np.logical_and(original_resist, corrected_resist)  # Blue: overlap
        
        axes[1, 2].imshow(comparison)
        axes[1, 2].set_title('Resist Comparison (R:Original, G:Corrected)')
        axes[1, 2].axis('off')
        
        plt.suptitle('Optical Proximity Correction Results')
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create test pattern
    test_mask = np.zeros((512, 512), dtype=np.uint8)
    
    # Add some features
    test_mask[100:150, 100:400] = 255  # Horizontal line
    test_mask[200:250, 100:400] = 255  # Another horizontal line
    test_mask[100:400, 200:250] = 255  # Vertical line
    test_mask[300:350, 300:350] = 255  # Square
    
    # Create OPC processor
    opc_params = OPCParameters(wavelength=193, NA=0.75, iterations=5)
    opc = OpticalProximityCorrection(opc_params)
    
    # Apply rule-based OPC
    corrected = opc.apply_rule_based_opc(test_mask)
    
    # Evaluate results
    metrics = opc.evaluate_correction()
    print("OPC Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Visualize results
    opc.visualize_opc_results()