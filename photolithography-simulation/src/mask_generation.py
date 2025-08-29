"""
Mask Pattern Generation Module

This module provides tools for generating photolithography mask patterns
including various geometric shapes, test patterns, and industry-standard layouts.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import cv2
from scipy import ndimage
from scipy.signal import convolve2d


@dataclass
class MaskParameters:
    """Parameters for mask generation"""
    width: int = 2048  # Width in pixels
    height: int = 2048  # Height in pixels
    pixel_size: float = 10.0  # nm per pixel
    wavelength: float = 193.0  # nm (ArF laser)
    NA: float = 0.75  # Numerical aperture
    sigma: float = 0.5  # Partial coherence factor
    
    @property
    def resolution_limit(self) -> float:
        """Calculate theoretical resolution limit"""
        return 0.61 * self.wavelength / self.NA


class MaskPatternGenerator:
    """
    Generate various mask patterns for photolithography simulation
    """
    
    def __init__(self, params: Optional[MaskParameters] = None):
        """
        Initialize mask pattern generator
        
        Args:
            params: Mask generation parameters
        """
        self.params = params or MaskParameters()
        self.mask = np.zeros((self.params.height, self.params.width), dtype=np.uint8)
        
    def create_blank_mask(self, value: int = 0) -> np.ndarray:
        """Create a blank mask with specified value"""
        self.mask = np.full((self.params.height, self.params.width), value, dtype=np.uint8)
        return self.mask
    
    def add_rectangle(self, x: int, y: int, width: int, height: int, 
                      value: int = 255) -> np.ndarray:
        """Add a rectangle to the mask"""
        self.mask[y:y+height, x:x+width] = value
        return self.mask
    
    def add_circle(self, cx: int, cy: int, radius: int, 
                   value: int = 255) -> np.ndarray:
        """Add a circle to the mask"""
        y, x = np.ogrid[:self.params.height, :self.params.width]
        mask_circle = (x - cx)**2 + (y - cy)**2 <= radius**2
        self.mask[mask_circle] = value
        return self.mask
    
    def add_polygon(self, vertices: List[Tuple[int, int]], 
                    value: int = 255) -> np.ndarray:
        """Add a polygon to the mask"""
        vertices_array = np.array(vertices, dtype=np.int32)
        cv2.fillPoly(self.mask, [vertices_array], value)
        return self.mask
    
    def create_line_space_pattern(self, line_width: int, space_width: int,
                                  orientation: str = 'vertical',
                                  value: int = 255) -> np.ndarray:
        """Create line and space pattern"""
        self.create_blank_mask()
        period = line_width + space_width
        
        if orientation == 'vertical':
            for x in range(0, self.params.width, period):
                self.add_rectangle(x, 0, line_width, self.params.height, value)
        else:  # horizontal
            for y in range(0, self.params.height, period):
                self.add_rectangle(0, y, self.params.width, line_width, value)
        
        return self.mask
    
    def create_contact_hole_array(self, hole_diameter: int, pitch: int,
                                  value: int = 255) -> np.ndarray:
        """Create array of contact holes"""
        self.create_blank_mask()
        radius = hole_diameter // 2
        
        for y in range(radius, self.params.height - radius, pitch):
            for x in range(radius, self.params.width - radius, pitch):
                self.add_circle(x, y, radius, value)
        
        return self.mask
    
    def create_resolution_test_pattern(self) -> np.ndarray:
        """Create resolution test pattern with various feature sizes"""
        self.create_blank_mask()
        
        # Different feature sizes to test
        feature_sizes = [10, 20, 30, 40, 50, 75, 100, 150, 200]
        y_offset = 50
        
        for i, size in enumerate(feature_sizes):
            x_offset = 50 + i * 220
            
            # Line patterns
            for j in range(5):
                self.add_rectangle(x_offset + j * 2 * size, y_offset, 
                                 size, size * 3, 255)
            
            # Contact holes
            for j in range(3):
                for k in range(3):
                    cx = x_offset + j * size * 3
                    cy = y_offset + 200 + k * size * 3
                    self.add_circle(cx, cy, size // 2, 255)
            
            # Isolated line
            self.add_rectangle(x_offset, y_offset + 400, size, size * 5, 255)
        
        return self.mask
    
    def create_phase_shift_mask(self, pattern_type: str = 'alternating') -> np.ndarray:
        """Create phase shift mask pattern"""
        self.create_blank_mask()
        
        if pattern_type == 'alternating':
            # Create alternating phase shift pattern
            line_width = 50
            for i in range(0, self.params.width, line_width * 2):
                # 0 degree phase
                self.add_rectangle(i, 0, line_width, self.params.height, 128)
                # 180 degree phase
                if i + line_width < self.params.width:
                    self.add_rectangle(i + line_width, 0, line_width, 
                                     self.params.height, 255)
        
        elif pattern_type == 'attenuated':
            # Create attenuated PSM pattern
            self.create_line_space_pattern(50, 50, value=180)
        
        return self.mask
    
    def add_defect(self, defect_type: str, x: int, y: int, 
                   size: int = 10) -> np.ndarray:
        """Add intentional defect for inspection testing"""
        if defect_type == 'particle':
            # Add particle (extra chrome)
            self.add_circle(x, y, size, 255)
        elif defect_type == 'pinhole':
            # Add pinhole (missing chrome)
            self.add_circle(x, y, size, 0)
        elif defect_type == 'bridge':
            # Add bridge defect
            self.add_rectangle(x, y, size * 3, size, 255)
        elif defect_type == 'break':
            # Add break in line
            self.add_rectangle(x, y, size, size, 0)
        
        return self.mask
    
    def apply_corner_rounding(self, radius: int = 5) -> np.ndarray:
        """Apply corner rounding to simulate manufacturing effects"""
        # Morphological operations to round corners
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        return self.mask
    
    def add_alignment_marks(self) -> np.ndarray:
        """Add alignment marks to mask"""
        mark_size = 100
        mark_width = 20
        margin = 50
        
        # Top-left corner mark
        self.add_rectangle(margin, margin, mark_size, mark_width, 255)
        self.add_rectangle(margin, margin, mark_width, mark_size, 255)
        
        # Top-right corner mark
        self.add_rectangle(self.params.width - margin - mark_size, margin, 
                         mark_size, mark_width, 255)
        self.add_rectangle(self.params.width - margin - mark_width, margin, 
                         mark_width, mark_size, 255)
        
        # Bottom-left corner mark
        self.add_rectangle(margin, self.params.height - margin - mark_width, 
                         mark_size, mark_width, 255)
        self.add_rectangle(margin, self.params.height - margin - mark_size, 
                         mark_width, mark_size, 255)
        
        # Bottom-right corner mark
        self.add_rectangle(self.params.width - margin - mark_size, 
                         self.params.height - margin - mark_width, 
                         mark_size, mark_width, 255)
        self.add_rectangle(self.params.width - margin - mark_width, 
                         self.params.height - margin - mark_size, 
                         mark_width, mark_size, 255)
        
        return self.mask
    
    def create_sram_pattern(self, cell_width: int = 100, 
                           cell_height: int = 150) -> np.ndarray:
        """Create SRAM cell pattern"""
        self.create_blank_mask()
        
        # Create repeating SRAM cell pattern
        for y in range(0, self.params.height - cell_height, cell_height):
            for x in range(0, self.params.width - cell_width, cell_width):
                # Active areas
                self.add_rectangle(x + 10, y + 10, 30, 60, 255)
                self.add_rectangle(x + 60, y + 10, 30, 60, 255)
                
                # Gate poly
                self.add_rectangle(x + 5, y + 30, 90, 15, 255)
                self.add_rectangle(x + 5, y + 55, 90, 15, 255)
                
                # Contacts
                self.add_circle(x + 25, y + 85, 8, 255)
                self.add_circle(x + 75, y + 85, 8, 255)
                
                # Metal lines
                self.add_rectangle(x + 20, y + 80, 10, 60, 255)
                self.add_rectangle(x + 70, y + 80, 10, 60, 255)
        
        return self.mask
    
    def export_mask(self, filename: str, format: str = 'png') -> None:
        """Export mask to file"""
        if format == 'png':
            cv2.imwrite(filename, self.mask)
        elif format == 'npy':
            np.save(filename, self.mask)
        elif format == 'gds':
            # Placeholder for GDS export
            print(f"GDS export not yet implemented. Saving as numpy array.")
            np.save(filename.replace('.gds', '.npy'), self.mask)
    
    def visualize_mask(self, title: str = "Mask Pattern", 
                      colormap: str = 'gray') -> None:
        """Visualize the mask pattern"""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.mask, cmap=colormap)
        plt.title(title)
        plt.colorbar(label='Mask Value')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        
        # Add scale information
        scale_text = f"Scale: {self.params.pixel_size} nm/pixel"
        plt.text(0.02, 0.98, scale_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def get_mask_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current mask"""
        stats = {
            'dimensions': (self.params.width, self.params.height),
            'pixel_size_nm': self.params.pixel_size,
            'total_area_um2': (self.params.width * self.params.height * 
                              self.params.pixel_size**2) / 1e6,
            'chrome_coverage': np.sum(self.mask > 0) / self.mask.size,
            'min_value': np.min(self.mask),
            'max_value': np.max(self.mask),
            'mean_value': np.mean(self.mask),
            'unique_values': len(np.unique(self.mask))
        }
        
        # Find minimum feature size
        if np.any(self.mask > 0):
            labeled, num_features = ndimage.label(self.mask > 0)
            if num_features > 0:
                sizes = ndimage.sum(self.mask > 0, labeled, range(num_features + 1))
                min_feature_pixels = np.min(sizes[1:])  # Exclude background
                stats['min_feature_size_nm'] = np.sqrt(min_feature_pixels) * self.params.pixel_size
            else:
                stats['min_feature_size_nm'] = None
        else:
            stats['min_feature_size_nm'] = None
        
        return stats


# Example usage
if __name__ == "__main__":
    # Create mask generator
    params = MaskParameters(width=2048, height=2048, pixel_size=5.0)
    generator = MaskPatternGenerator(params)
    
    # Create resolution test pattern
    mask = generator.create_resolution_test_pattern()
    generator.add_alignment_marks()
    
    # Add some intentional defects for testing
    generator.add_defect('particle', 500, 500, 15)
    generator.add_defect('pinhole', 700, 700, 10)
    
    # Get statistics
    stats = generator.get_mask_statistics()
    print("Mask Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Export mask
    generator.export_mask('test_mask.png')
    print("\nMask saved as test_mask.png")