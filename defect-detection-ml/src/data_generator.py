"""
Data Generation Module for Defect Detection

This module provides tools for generating synthetic wafer defect data
for training and testing ML models.
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import random
from scipy import ndimage
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from enum import Enum


class DefectType(Enum):
    """Enumeration of defect types"""
    PARTICLE = "particle"
    SCRATCH = "scratch"
    PATTERN = "pattern"
    EDGE = "edge"
    CLUSTER = "cluster"
    RING = "ring"
    RANDOM = "random"
    NONE = "none"


@dataclass
class WaferMap:
    """Data class for wafer map information"""
    data: np.ndarray  # Wafer map data
    defect_type: DefectType
    die_size: Tuple[int, int]
    wafer_diameter: float  # in mm
    defect_count: int
    yield_rate: float
    metadata: Dict[str, Any]


class DefectDataGenerator:
    """
    Generate synthetic defect data for training ML models
    """
    
    def __init__(self, wafer_size: int = 300, die_size: Tuple[int, int] = (10, 10)):
        """
        Initialize defect data generator
        
        Args:
            wafer_size: Diameter of wafer in mm
            die_size: Size of each die in mm (width, height)
        """
        self.wafer_size = wafer_size
        self.die_size = die_size
        self.image_size = 512  # Pixels for image representation
        
    def generate_wafer_map(self, defect_type: DefectType = DefectType.RANDOM,
                          defect_density: float = 0.05,
                          seed: Optional[int] = None) -> WaferMap:
        """
        Generate a synthetic wafer map with defects
        
        Args:
            defect_type: Type of defect pattern to generate
            defect_density: Fraction of defective dies
            seed: Random seed for reproducibility
            
        Returns:
            WaferMap object containing the generated data
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Calculate grid dimensions
        dies_per_row = int(self.wafer_size / self.die_size[0])
        dies_per_col = int(self.wafer_size / self.die_size[1])
        
        # Create wafer map grid
        wafer_map = np.zeros((dies_per_col, dies_per_row), dtype=np.uint8)
        
        # Create circular mask for wafer
        center = (dies_per_col // 2, dies_per_row // 2)
        radius = min(dies_per_col, dies_per_row) // 2
        
        y, x = np.ogrid[:dies_per_col, :dies_per_row]
        mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
        
        # Apply defect pattern
        if defect_type == DefectType.RANDOM or defect_type == DefectType.NONE:
            defect_type = random.choice(list(DefectType)[:-2])  # Exclude RANDOM and NONE
        
        if defect_type == DefectType.PARTICLE:
            wafer_map = self._add_particle_defects(wafer_map, mask, defect_density)
        elif defect_type == DefectType.SCRATCH:
            wafer_map = self._add_scratch_defects(wafer_map, mask, defect_density)
        elif defect_type == DefectType.PATTERN:
            wafer_map = self._add_pattern_defects(wafer_map, mask, defect_density)
        elif defect_type == DefectType.EDGE:
            wafer_map = self._add_edge_defects(wafer_map, mask, defect_density)
        elif defect_type == DefectType.CLUSTER:
            wafer_map = self._add_cluster_defects(wafer_map, mask, defect_density)
        elif defect_type == DefectType.RING:
            wafer_map = self._add_ring_defects(wafer_map, mask, defect_density)
        
        # Calculate statistics
        defect_count = np.sum(wafer_map > 0)
        total_dies = np.sum(mask)
        yield_rate = 1 - (defect_count / total_dies) if total_dies > 0 else 0
        
        # Create metadata
        metadata = {
            'generation_params': {
                'defect_type': defect_type.value,
                'defect_density': defect_density,
                'seed': seed
            },
            'wafer_info': {
                'diameter_mm': self.wafer_size,
                'die_size_mm': self.die_size,
                'total_dies': int(total_dies),
                'grid_size': (dies_per_col, dies_per_row)
            }
        }
        
        return WaferMap(
            data=wafer_map,
            defect_type=defect_type,
            die_size=self.die_size,
            wafer_diameter=self.wafer_size,
            defect_count=int(defect_count),
            yield_rate=float(yield_rate),
            metadata=metadata
        )
    
    def _add_particle_defects(self, wafer_map: np.ndarray, mask: np.ndarray,
                             density: float) -> np.ndarray:
        """Add random particle defects"""
        n_defects = int(np.sum(mask) * density)
        valid_positions = np.where(mask)
        
        for _ in range(n_defects):
            idx = random.randint(0, len(valid_positions[0]) - 1)
            y, x = valid_positions[0][idx], valid_positions[1][idx]
            wafer_map[y, x] = 1
        
        return wafer_map
    
    def _add_scratch_defects(self, wafer_map: np.ndarray, mask: np.ndarray,
                            density: float) -> np.ndarray:
        """Add scratch-like linear defects"""
        n_scratches = max(1, int(density * 10))
        
        for _ in range(n_scratches):
            # Random line parameters
            angle = random.uniform(0, np.pi)
            length = random.randint(5, min(wafer_map.shape) // 2)
            start_y = random.randint(0, wafer_map.shape[0] - 1)
            start_x = random.randint(0, wafer_map.shape[1] - 1)
            
            # Generate line points
            for i in range(length):
                y = int(start_y + i * np.sin(angle))
                x = int(start_x + i * np.cos(angle))
                
                if 0 <= y < wafer_map.shape[0] and 0 <= x < wafer_map.shape[1]:
                    if mask[y, x]:
                        wafer_map[y, x] = 1
                        # Add some width to scratch
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < wafer_map.shape[0] and 0 <= nx < wafer_map.shape[1]:
                                    if mask[ny, nx] and random.random() < 0.3:
                                        wafer_map[ny, nx] = 1
        
        return wafer_map
    
    def _add_pattern_defects(self, wafer_map: np.ndarray, mask: np.ndarray,
                            density: float) -> np.ndarray:
        """Add systematic pattern defects"""
        # Create repeating pattern
        period_y = random.randint(3, 8)
        period_x = random.randint(3, 8)
        offset_y = random.randint(0, period_y - 1)
        offset_x = random.randint(0, period_x - 1)
        
        for y in range(wafer_map.shape[0]):
            for x in range(wafer_map.shape[1]):
                if mask[y, x]:
                    if (y - offset_y) % period_y == 0 or (x - offset_x) % period_x == 0:
                        if random.random() < density * 5:  # Adjust probability
                            wafer_map[y, x] = 1
        
        return wafer_map
    
    def _add_edge_defects(self, wafer_map: np.ndarray, mask: np.ndarray,
                         density: float) -> np.ndarray:
        """Add defects concentrated at wafer edge"""
        center = (wafer_map.shape[0] // 2, wafer_map.shape[1] // 2)
        max_radius = min(wafer_map.shape) // 2
        
        for y in range(wafer_map.shape[0]):
            for x in range(wafer_map.shape[1]):
                if mask[y, x]:
                    dist = np.sqrt((y - center[0])**2 + (x - center[1])**2)
                    # Higher probability near edge
                    edge_prob = (dist / max_radius)**2
                    if random.random() < density * edge_prob * 5:
                        wafer_map[y, x] = 1
        
        return wafer_map
    
    def _add_cluster_defects(self, wafer_map: np.ndarray, mask: np.ndarray,
                            density: float) -> np.ndarray:
        """Add clustered defects"""
        n_clusters = max(1, int(density * 5))
        n_defects = int(np.sum(mask) * density)
        
        # Generate cluster centers
        valid_positions = np.where(mask)
        cluster_centers = []
        
        for _ in range(n_clusters):
            idx = random.randint(0, len(valid_positions[0]) - 1)
            center_y = valid_positions[0][idx]
            center_x = valid_positions[1][idx]
            cluster_centers.append((center_y, center_x))
        
        # Add defects around clusters
        defects_per_cluster = n_defects // n_clusters
        
        for center_y, center_x in cluster_centers:
            for _ in range(defects_per_cluster):
                # Gaussian distribution around center
                dy = int(np.random.normal(0, 3))
                dx = int(np.random.normal(0, 3))
                
                y = center_y + dy
                x = center_x + dx
                
                if 0 <= y < wafer_map.shape[0] and 0 <= x < wafer_map.shape[1]:
                    if mask[y, x]:
                        wafer_map[y, x] = 1
        
        return wafer_map
    
    def _add_ring_defects(self, wafer_map: np.ndarray, mask: np.ndarray,
                         density: float) -> np.ndarray:
        """Add ring-shaped defects"""
        center = (wafer_map.shape[0] // 2, wafer_map.shape[1] // 2)
        max_radius = min(wafer_map.shape) // 2
        
        # Create random rings
        n_rings = random.randint(1, 3)
        ring_radii = [random.uniform(0.3, 0.8) * max_radius for _ in range(n_rings)]
        ring_width = 2
        
        for y in range(wafer_map.shape[0]):
            for x in range(wafer_map.shape[1]):
                if mask[y, x]:
                    dist = np.sqrt((y - center[0])**2 + (x - center[1])**2)
                    
                    for ring_radius in ring_radii:
                        if abs(dist - ring_radius) < ring_width:
                            if random.random() < density * 10:
                                wafer_map[y, x] = 1
        
        return wafer_map
    
    def generate_defect_image(self, size: Tuple[int, int] = (224, 224),
                             defect_class: str = "particle",
                             add_noise: bool = True) -> np.ndarray:
        """
        Generate a synthetic defect image for CNN training
        
        Args:
            size: Image size (height, width)
            defect_class: Type of defect to generate
            add_noise: Whether to add background noise
            
        Returns:
            Synthetic defect image
        """
        image = np.zeros(size, dtype=np.uint8)
        
        # Add background pattern (simulating wafer surface)
        if add_noise:
            noise = np.random.normal(128, 20, size)
            image = np.clip(noise, 0, 255).astype(np.uint8)
        else:
            image.fill(128)
        
        # Add defect based on class
        if defect_class == "particle":
            # Add circular particle
            center = (random.randint(size[0]//4, 3*size[0]//4),
                     random.randint(size[1]//4, 3*size[1]//4))
            radius = random.randint(5, min(size)//8)
            cv2.circle(image, center, radius, 
                      random.randint(0, 100), -1)
            
        elif defect_class == "scratch":
            # Add line scratch
            pt1 = (random.randint(0, size[1]), random.randint(0, size[0]))
            pt2 = (random.randint(0, size[1]), random.randint(0, size[0]))
            thickness = random.randint(1, 5)
            cv2.line(image, pt1, pt2, random.randint(0, 100), thickness)
            
        elif defect_class == "void":
            # Add irregular void
            center = (size[0]//2, size[1]//2)
            axes = (random.randint(10, 30), random.randint(10, 30))
            angle = random.randint(0, 180)
            cv2.ellipse(image, center, axes, angle, 0, 360,
                       random.randint(200, 255), -1)
            
        elif defect_class == "bridge":
            # Add bridging defect
            rect_start = (random.randint(0, size[1]//2),
                         random.randint(0, size[0]//2))
            rect_end = (random.randint(size[1]//2, size[1]),
                       random.randint(size[0]//2, size[0]))
            cv2.rectangle(image, rect_start, rect_end,
                         random.randint(0, 100), -1)
            
        elif defect_class == "none":
            # No defect (for negative samples)
            pass
        
        # Apply some image processing for realism
        if add_noise:
            # Add Gaussian blur
            image = cv2.GaussianBlur(image, (3, 3), 0)
            
            # Add some salt and pepper noise
            noise = np.random.random(size)
            image[noise < 0.005] = 0
            image[noise > 0.995] = 255
        
        return image
    
    def generate_dataset(self, n_samples: int = 1000,
                        defect_classes: Optional[List[str]] = None,
                        train_split: float = 0.8) -> Dict[str, Any]:
        """
        Generate a complete dataset for training
        
        Args:
            n_samples: Total number of samples to generate
            defect_classes: List of defect classes to include
            train_split: Fraction of data for training
            
        Returns:
            Dictionary containing train and test datasets
        """
        if defect_classes is None:
            defect_classes = ["particle", "scratch", "void", "bridge", "none"]
        
        samples_per_class = n_samples // len(defect_classes)
        
        images = []
        labels = []
        
        for class_idx, defect_class in enumerate(defect_classes):
            for _ in range(samples_per_class):
                image = self.generate_defect_image(defect_class=defect_class)
                images.append(image)
                labels.append(class_idx)
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # Shuffle
        indices = np.random.permutation(len(images))
        images = images[indices]
        labels = labels[indices]
        
        # Split into train and test
        n_train = int(len(images) * train_split)
        
        dataset = {
            'train': {
                'images': images[:n_train],
                'labels': labels[:n_train]
            },
            'test': {
                'images': images[n_train:],
                'labels': labels[n_train:]
            },
            'classes': defect_classes,
            'n_classes': len(defect_classes)
        }
        
        return dataset


class SyntheticWaferGenerator:
    """
    Generate synthetic full wafer images with defects
    """
    
    def __init__(self, wafer_diameter: int = 300, resolution: int = 2048):
        """
        Initialize synthetic wafer generator
        
        Args:
            wafer_diameter: Wafer diameter in mm
            resolution: Image resolution in pixels
        """
        self.wafer_diameter = wafer_diameter
        self.resolution = resolution
        self.pixels_per_mm = resolution / wafer_diameter
        
    def generate_wafer_image(self, wafer_map: WaferMap,
                            add_texture: bool = True) -> np.ndarray:
        """
        Generate a realistic wafer image from wafer map
        
        Args:
            wafer_map: WaferMap object with defect data
            add_texture: Whether to add realistic texture
            
        Returns:
            Synthetic wafer image
        """
        # Create base image
        image = np.ones((self.resolution, self.resolution), dtype=np.uint8) * 200
        
        # Create circular mask
        center = self.resolution // 2
        y, x = np.ogrid[:self.resolution, :self.resolution]
        mask = (x - center)**2 + (y - center)**2 <= (center - 10)**2
        
        # Apply wafer background
        if add_texture:
            # Add silicon wafer texture
            texture = self._generate_wafer_texture()
            image[mask] = texture[mask]
        
        # Add die grid
        die_width = int(wafer_map.die_size[0] * self.pixels_per_mm)
        die_height = int(wafer_map.die_size[1] * self.pixels_per_mm)
        
        # Calculate starting position to center the grid
        grid_width = wafer_map.data.shape[1] * die_width
        grid_height = wafer_map.data.shape[0] * die_height
        start_x = (self.resolution - grid_width) // 2
        start_y = (self.resolution - grid_height) // 2
        
        # Draw die boundaries and defects
        for i in range(wafer_map.data.shape[0]):
            for j in range(wafer_map.data.shape[1]):
                x1 = start_x + j * die_width
                y1 = start_y + i * die_height
                x2 = x1 + die_width
                y2 = y1 + die_height
                
                # Check if die is within wafer
                die_center_x = (x1 + x2) // 2
                die_center_y = (y1 + y2) // 2
                
                if (die_center_x - center)**2 + (die_center_y - center)**2 <= (center - 20)**2:
                    # Draw die boundary
                    cv2.rectangle(image, (x1, y1), (x2, y2), 150, 1)
                    
                    # Mark defective dies
                    if wafer_map.data[i, j] > 0:
                        # Add defect visualization
                        defect_color = 50  # Dark color for defects
                        cv2.rectangle(image, (x1+2, y1+2), (x2-2, y2-2),
                                    defect_color, -1)
        
        # Add wafer edge
        cv2.circle(image, (center, center), center - 5, 100, 2)
        
        # Set background to black
        image[~mask] = 0
        
        return image
    
    def _generate_wafer_texture(self) -> np.ndarray:
        """Generate realistic silicon wafer texture"""
        # Base silicon color/intensity
        texture = np.ones((self.resolution, self.resolution), dtype=np.float32) * 180
        
        # Add some periodic patterns (crystal structure)
        x = np.linspace(0, 4*np.pi, self.resolution)
        y = np.linspace(0, 4*np.pi, self.resolution)
        X, Y = np.meshgrid(x, y)
        
        pattern1 = np.sin(X) * np.cos(Y) * 5
        pattern2 = np.sin(2*X + Y) * 3
        
        texture += pattern1 + pattern2
        
        # Add some random variations
        noise = np.random.normal(0, 2, texture.shape)
        texture += noise
        
        # Add radial gradient (common in wafers)
        center = self.resolution // 2
        y, x = np.ogrid[:self.resolution, :self.resolution]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        max_r = self.resolution // 2
        gradient = 1 - (r / max_r) * 0.1  # Slight darkening towards edge
        texture *= gradient
        
        # Clip and convert to uint8
        texture = np.clip(texture, 0, 255).astype(np.uint8)
        
        return texture
    
    def add_inspection_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Add realistic inspection artifacts to wafer image
        
        Args:
            image: Input wafer image
            
        Returns:
            Image with added artifacts
        """
        result = image.copy()
        
        # Add illumination non-uniformity
        center = self.resolution // 2
        y, x = np.ogrid[:self.resolution, :self.resolution]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        illumination = 1 + 0.1 * np.cos(r / 100)
        result = (result * illumination).astype(np.uint8)
        
        # Add some dust particles
        n_dust = random.randint(5, 20)
        for _ in range(n_dust):
            x = random.randint(0, self.resolution - 1)
            y = random.randint(0, self.resolution - 1)
            radius = random.randint(1, 5)
            intensity = random.randint(0, 100)
            cv2.circle(result, (x, y), radius, intensity, -1)
        
        # Add slight motion blur (scanning artifact)
        if random.random() < 0.3:
            kernel_size = 3
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size//2, :] = 1 / kernel_size
            result = cv2.filter2D(result, -1, kernel)
        
        return result


# Example usage
if __name__ == "__main__":
    # Create defect generator
    generator = DefectDataGenerator(wafer_size=300, die_size=(10, 10))
    
    # Generate wafer maps with different defect types
    defect_types = [DefectType.PARTICLE, DefectType.SCRATCH, DefectType.CLUSTER,
                   DefectType.RING, DefectType.EDGE, DefectType.PATTERN]
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    
    for idx, defect_type in enumerate(defect_types):
        wafer_map = generator.generate_wafer_map(
            defect_type=defect_type,
            defect_density=0.05
        )
        
        # Visualize wafer map
        axes[idx].imshow(wafer_map.data, cmap='RdYlGn_r')
        axes[idx].set_title(f'{defect_type.value.capitalize()} Defects\n'
                           f'Yield: {wafer_map.yield_rate:.1%}')
        axes[idx].axis('off')
    
    plt.suptitle('Synthetic Wafer Defect Maps')
    plt.tight_layout()
    plt.show()
    
    # Generate synthetic wafer image
    wafer_gen = SyntheticWaferGenerator()
    wafer_image = wafer_gen.generate_wafer_image(wafer_map)
    wafer_image = wafer_gen.add_inspection_artifacts(wafer_image)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(wafer_image, cmap='gray')
    plt.title('Synthetic Wafer Image')
    plt.axis('off')
    plt.show()
    
    # Generate defect image dataset
    dataset = generator.generate_dataset(n_samples=100)
    print(f"Generated dataset with {len(dataset['train']['images'])} training samples")
    print(f"and {len(dataset['test']['images'])} test samples")
    print(f"Classes: {dataset['classes']}")