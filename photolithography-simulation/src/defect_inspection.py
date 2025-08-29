"""
Defect Inspection Analysis Module

This module provides tools for detecting and analyzing defects in 
photolithography masks and wafer patterns.
"""

import numpy as np
import cv2
from scipy import ndimage, stats
from scipy.spatial import distance
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
import warnings


class DefectType(Enum):
    """Enumeration of defect types"""
    PARTICLE = "particle"
    PINHOLE = "pinhole"
    BRIDGE = "bridge"
    BREAK = "break"
    EXTENSION = "extension"
    INTRUSION = "intrusion"
    MOUSE_BITE = "mouse_bite"
    MISSING_PATTERN = "missing_pattern"
    EXTRA_PATTERN = "extra_pattern"
    EDGE_ROUGHNESS = "edge_roughness"


@dataclass
class Defect:
    """Data class for defect information"""
    type: DefectType
    location: Tuple[int, int]  # (x, y) coordinates
    size: float  # Size in pixels or nm
    severity: str  # 'critical', 'major', 'minor'
    area: float  # Area of defect
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float  # Detection confidence (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert defect to dictionary"""
        return {
            'type': self.type.value,
            'location': self.location,
            'size': self.size,
            'severity': self.severity,
            'area': self.area,
            'bounding_box': self.bounding_box,
            'confidence': self.confidence
        }


class DefectInspector:
    """
    Detect and analyze defects in photolithography patterns
    """
    
    def __init__(self, pixel_size: float = 5.0):
        """
        Initialize defect inspector
        
        Args:
            pixel_size: Physical size of each pixel in nm
        """
        self.pixel_size = pixel_size
        self.reference_pattern = None
        self.test_pattern = None
        self.defects = []
        self.defect_map = None
        
    def set_reference_pattern(self, pattern: np.ndarray) -> None:
        """Set the reference (golden) pattern"""
        self.reference_pattern = pattern.copy()
        
    def die_to_die_inspection(self, test_pattern: np.ndarray,
                             reference_pattern: Optional[np.ndarray] = None,
                             threshold: float = 10) -> List[Defect]:
        """
        Perform die-to-die comparison for defect detection
        
        Args:
            test_pattern: Pattern to inspect
            reference_pattern: Reference pattern (uses stored if None)
            threshold: Difference threshold for defect detection
            
        Returns:
            List of detected defects
        """
        if reference_pattern is None:
            if self.reference_pattern is None:
                raise ValueError("No reference pattern provided")
            reference_pattern = self.reference_pattern
        
        self.test_pattern = test_pattern.copy()
        
        # Align patterns if necessary
        aligned_test = self._align_patterns(test_pattern, reference_pattern)
        
        # Calculate difference
        diff = cv2.absdiff(reference_pattern, aligned_test)
        
        # Apply threshold
        _, defect_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Find connected components (defects)
        num_defects, labels, stats_array, centroids = cv2.connectedComponentsWithStats(
            defect_mask.astype(np.uint8), connectivity=8)
        
        self.defects = []
        self.defect_map = labels
        
        # Analyze each defect (skip background)
        for i in range(1, num_defects):
            defect = self._analyze_defect(
                labels == i,
                reference_pattern,
                aligned_test,
                stats_array[i],
                centroids[i]
            )
            if defect:
                self.defects.append(defect)
        
        return self.defects
    
    def _align_patterns(self, test: np.ndarray, 
                       reference: np.ndarray) -> np.ndarray:
        """Align test pattern to reference using feature matching"""
        # Convert to uint8 if necessary
        test_uint8 = test.astype(np.uint8)
        ref_uint8 = reference.astype(np.uint8)
        
        # Detect ORB features
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(ref_uint8, None)
        kp2, des2 = orb.detectAndCompute(test_uint8, None)
        
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            # Not enough features for alignment
            return test
        
        # Match features
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        
        if len(matches) < 4:
            return test
        
        # Get matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find transformation
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            return test
        
        # Apply transformation
        aligned = cv2.warpPerspective(test, M, (reference.shape[1], reference.shape[0]))
        
        return aligned
    
    def _analyze_defect(self, defect_mask: np.ndarray,
                       reference: np.ndarray,
                       test: np.ndarray,
                       stats: np.ndarray,
                       centroid: np.ndarray) -> Optional[Defect]:
        """Analyze a single defect and classify it"""
        x, y, w, h, area = stats
        cx, cy = centroid
        
        # Skip very small defects (noise)
        if area < 5:
            return None
        
        # Extract defect regions
        ref_region = reference[y:y+h, x:x+w]
        test_region = test[y:y+h, x:x+w]
        defect_region = defect_mask[y:y+h, x:x+w]
        
        # Classify defect type
        defect_type = self._classify_defect(ref_region, test_region, defect_region)
        
        # Calculate size
        size = np.sqrt(area) * self.pixel_size  # Size in nm
        
        # Determine severity
        severity = self._determine_severity(size, defect_type)
        
        # Calculate confidence
        confidence = self._calculate_confidence(defect_region, ref_region, test_region)
        
        return Defect(
            type=defect_type,
            location=(int(cx), int(cy)),
            size=size,
            severity=severity,
            area=area * self.pixel_size**2,  # Area in nm²
            bounding_box=(x, y, w, h),
            confidence=confidence
        )
    
    def _classify_defect(self, reference: np.ndarray,
                        test: np.ndarray,
                        defect_mask: np.ndarray) -> DefectType:
        """Classify the type of defect"""
        ref_sum = np.sum(reference > 0)
        test_sum = np.sum(test > 0)
        
        if test_sum > ref_sum * 1.2:
            # Extra material
            if self._is_bridging(defect_mask):
                return DefectType.BRIDGE
            elif self._is_extension(defect_mask):
                return DefectType.EXTENSION
            else:
                return DefectType.PARTICLE
        elif test_sum < ref_sum * 0.8:
            # Missing material
            if self._is_break(defect_mask):
                return DefectType.BREAK
            elif self._is_pinhole(defect_mask):
                return DefectType.PINHOLE
            else:
                return DefectType.MISSING_PATTERN
        else:
            # Shape distortion
            if self._has_edge_roughness(test):
                return DefectType.EDGE_ROUGHNESS
            elif self._is_mouse_bite(defect_mask):
                return DefectType.MOUSE_BITE
            else:
                return DefectType.INTRUSION
    
    def _is_bridging(self, defect_mask: np.ndarray) -> bool:
        """Check if defect is a bridge between features"""
        # Simplified check: elongated defect
        if defect_mask.size == 0:
            return False
        
        contours, _ = cv2.findContours(
            defect_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            contour = contours[0]
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                aspect_ratio = max(ellipse[1]) / (min(ellipse[1]) + 1e-6)
                return aspect_ratio > 3
        return False
    
    def _is_extension(self, defect_mask: np.ndarray) -> bool:
        """Check if defect is a line extension"""
        # Check if defect is at edge of pattern
        edges = cv2.Canny(defect_mask.astype(np.uint8), 50, 150)
        edge_ratio = np.sum(edges > 0) / (defect_mask.size + 1e-6)
        return edge_ratio > 0.3
    
    def _is_break(self, defect_mask: np.ndarray) -> bool:
        """Check if defect is a line break"""
        # Similar to extension but for missing material
        return self._is_extension(defect_mask)
    
    def _is_pinhole(self, defect_mask: np.ndarray) -> bool:
        """Check if defect is a pinhole"""
        # Check circularity
        contours, _ = cv2.findContours(
            defect_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            contour = contours[0]
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter**2)
                return circularity > 0.7
        return False
    
    def _is_mouse_bite(self, defect_mask: np.ndarray) -> bool:
        """Check if defect is a mouse bite (edge intrusion)"""
        # Check if defect is small and at edge
        return defect_mask.size < 100 and self._is_extension(defect_mask)
    
    def _has_edge_roughness(self, pattern: np.ndarray) -> bool:
        """Check for edge roughness"""
        edges = cv2.Canny(pattern.astype(np.uint8), 50, 150)
        
        # Calculate edge roughness metric
        if np.sum(edges) > 0:
            # Simplified: use standard deviation of edge positions
            edge_points = np.where(edges > 0)
            if len(edge_points[0]) > 10:
                std_dev = np.std(edge_points[1])
                return std_dev > 5
        return False
    
    def _determine_severity(self, size: float, defect_type: DefectType) -> str:
        """Determine defect severity based on size and type"""
        # Critical defects
        if defect_type in [DefectType.BRIDGE, DefectType.BREAK]:
            return 'critical'
        
        # Size-based classification
        if size > 100:  # > 100 nm
            return 'critical'
        elif size > 50:  # 50-100 nm
            return 'major'
        else:
            return 'minor'
    
    def _calculate_confidence(self, defect_mask: np.ndarray,
                            reference: np.ndarray,
                            test: np.ndarray) -> float:
        """Calculate detection confidence"""
        # Based on contrast and clarity of defect
        if defect_mask.size == 0:
            return 0.0
        
        # Calculate contrast
        ref_mean = np.mean(reference)
        test_mean = np.mean(test)
        contrast = abs(ref_mean - test_mean) / (ref_mean + test_mean + 1e-6)
        
        # Calculate defect clarity (how well defined)
        defect_density = np.sum(defect_mask > 0) / defect_mask.size
        
        confidence = min(1.0, contrast + defect_density)
        return confidence
    
    def pattern_fidelity_analysis(self, pattern: np.ndarray,
                                 target_cd: float) -> Dict[str, Any]:
        """
        Analyze pattern fidelity metrics
        
        Args:
            pattern: Pattern to analyze
            target_cd: Target critical dimension in nm
            
        Returns:
            Dictionary of fidelity metrics
        """
        metrics = {}
        
        # Measure critical dimensions
        cd_measurements = self._measure_critical_dimensions(pattern)
        metrics['mean_cd'] = np.mean(cd_measurements) * self.pixel_size
        metrics['cd_uniformity'] = np.std(cd_measurements) * self.pixel_size
        metrics['cd_deviation'] = abs(metrics['mean_cd'] - target_cd)
        
        # Measure line edge roughness
        metrics['line_edge_roughness'] = self._measure_line_edge_roughness(pattern)
        
        # Measure line width roughness
        metrics['line_width_roughness'] = self._measure_line_width_roughness(pattern)
        
        # Pattern density
        metrics['pattern_density'] = np.sum(pattern > 0) / pattern.size
        
        # Edge angle analysis
        metrics['edge_angle'] = self._measure_edge_angles(pattern)
        
        return metrics
    
    def _measure_critical_dimensions(self, pattern: np.ndarray) -> List[float]:
        """Measure critical dimensions in pattern"""
        measurements = []
        
        # Find horizontal lines
        for y in range(pattern.shape[0]):
            line = pattern[y, :]
            edges = np.where(np.diff(line) != 0)[0]
            
            if len(edges) >= 2:
                for i in range(0, len(edges)-1, 2):
                    width = edges[i+1] - edges[i]
                    if width > 5:  # Filter noise
                        measurements.append(width)
        
        return measurements if measurements else [0]
    
    def _measure_line_edge_roughness(self, pattern: np.ndarray) -> float:
        """Measure line edge roughness (LER)"""
        edges = cv2.Canny(pattern.astype(np.uint8), 50, 150)
        
        # Find vertical edges
        vertical_edges = []
        for x in range(edges.shape[1]):
            edge_points = np.where(edges[:, x] > 0)[0]
            if len(edge_points) > 0:
                vertical_edges.extend(edge_points)
        
        if len(vertical_edges) > 10:
            # Calculate standard deviation of edge positions
            ler = np.std(vertical_edges) * self.pixel_size
            return ler
        return 0.0
    
    def _measure_line_width_roughness(self, pattern: np.ndarray) -> float:
        """Measure line width roughness (LWR)"""
        widths = self._measure_critical_dimensions(pattern)
        if len(widths) > 1:
            lwr = np.std(widths) * self.pixel_size
            return lwr
        return 0.0
    
    def _measure_edge_angles(self, pattern: np.ndarray) -> float:
        """Measure average edge angle (ideally 90 degrees)"""
        edges = cv2.Canny(pattern.astype(np.uint8), 50, 150)
        
        # Calculate gradient
        sobelx = cv2.Sobel(pattern, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(pattern, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate angles
        angles = np.arctan2(sobely, sobelx) * 180 / np.pi
        edge_angles = angles[edges > 0]
        
        if len(edge_angles) > 0:
            # Convert to 0-90 range
            edge_angles = np.abs(edge_angles)
            edge_angles = np.minimum(edge_angles, 180 - edge_angles)
            return np.mean(edge_angles)
        return 90.0
    
    def generate_inspection_report(self) -> Dict[str, Any]:
        """Generate comprehensive inspection report"""
        if not self.defects:
            return {'status': 'No defects detected'}
        
        report = {
            'total_defects': len(self.defects),
            'defects_by_type': {},
            'defects_by_severity': {},
            'critical_defects': [],
            'defect_density': len(self.defects) / (
                self.test_pattern.size * self.pixel_size**2 / 1e6),  # defects/mm²
            'largest_defect': None,
            'defect_list': []
        }
        
        # Count by type
        for defect in self.defects:
            defect_type = defect.type.value
            report['defects_by_type'][defect_type] = \
                report['defects_by_type'].get(defect_type, 0) + 1
            
            # Count by severity
            report['defects_by_severity'][defect.severity] = \
                report['defects_by_severity'].get(defect.severity, 0) + 1
            
            # Track critical defects
            if defect.severity == 'critical':
                report['critical_defects'].append(defect.to_dict())
            
            # Track largest defect
            if report['largest_defect'] is None or \
               defect.size > report['largest_defect']['size']:
                report['largest_defect'] = defect.to_dict()
            
            # Add to list
            report['defect_list'].append(defect.to_dict())
        
        return report
    
    def visualize_inspection_results(self) -> None:
        """Visualize inspection results"""
        if self.test_pattern is None:
            print("No inspection performed yet")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Reference pattern
        if self.reference_pattern is not None:
            axes[0, 0].imshow(self.reference_pattern, cmap='gray')
            axes[0, 0].set_title('Reference Pattern')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Reference', ha='center', va='center')
            axes[0, 0].set_title('Reference Pattern')
        axes[0, 0].axis('off')
        
        # Test pattern
        axes[0, 1].imshow(self.test_pattern, cmap='gray')
        axes[0, 1].set_title('Test Pattern')
        axes[0, 1].axis('off')
        
        # Defect map
        if self.defect_map is not None:
            axes[0, 2].imshow(self.defect_map, cmap='jet')
            axes[0, 2].set_title('Defect Map')
        else:
            axes[0, 2].text(0.5, 0.5, 'No Defects', ha='center', va='center')
            axes[0, 2].set_title('Defect Map')
        axes[0, 2].axis('off')
        
        # Defect overlay
        overlay = self.test_pattern.copy()
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        
        for defect in self.defects:
            x, y, w, h = defect.bounding_box
            color = (255, 0, 0) if defect.severity == 'critical' else \
                   (255, 255, 0) if defect.severity == 'major' else (0, 255, 0)
            cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
            cv2.putText(overlay, defect.type.value[:3], (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Defect Overlay')
        axes[1, 0].axis('off')
        
        # Defect statistics
        if self.defects:
            defect_types = {}
            for defect in self.defects:
                defect_types[defect.type.value] = \
                    defect_types.get(defect.type.value, 0) + 1
            
            axes[1, 1].bar(defect_types.keys(), defect_types.values())
            axes[1, 1].set_title('Defects by Type')
            axes[1, 1].set_xlabel('Defect Type')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Defects', ha='center', va='center')
            axes[1, 1].set_title('Defects by Type')
        
        # Severity distribution
        if self.defects:
            severities = {}
            for defect in self.defects:
                severities[defect.severity] = \
                    severities.get(defect.severity, 0) + 1
            
            colors = {'critical': 'red', 'major': 'yellow', 'minor': 'green'}
            bar_colors = [colors.get(s, 'gray') for s in severities.keys()]
            axes[1, 2].bar(severities.keys(), severities.values(), color=bar_colors)
            axes[1, 2].set_title('Defects by Severity')
            axes[1, 2].set_xlabel('Severity')
            axes[1, 2].set_ylabel('Count')
        else:
            axes[1, 2].text(0.5, 0.5, 'No Defects', ha='center', va='center')
            axes[1, 2].set_title('Defects by Severity')
        
        plt.suptitle('Defect Inspection Results')
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create test patterns
    reference = np.zeros((512, 512), dtype=np.uint8)
    reference[100:150, 100:400] = 255  # Horizontal line
    reference[200:250, 100:400] = 255  # Another horizontal line
    reference[100:400, 200:250] = 255  # Vertical line
    
    # Create test pattern with defects
    test = reference.copy()
    
    # Add defects
    test[120:130, 300:350] = 255  # Bridge defect
    test[220:230, 150:160] = 0     # Pinhole
    test[350:360, 220:230] = 255   # Particle
    test[110:120, 180:190] = 0     # Break in line
    
    # Create inspector
    inspector = DefectInspector(pixel_size=5.0)
    
    # Perform inspection
    defects = inspector.die_to_die_inspection(test, reference)
    
    # Generate report
    report = inspector.generate_inspection_report()
    print("\nInspection Report:")
    print(f"Total defects: {report['total_defects']}")
    print(f"Defects by type: {report['defects_by_type']}")
    print(f"Defects by severity: {report['defects_by_severity']}")
    print(f"Defect density: {report['defect_density']:.2f} defects/mm²")
    
    # Visualize results
    inspector.visualize_inspection_results()