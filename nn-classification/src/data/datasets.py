"""
Dataset classes for semiconductor defects and medical imaging
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Tuple, List, Callable
import random


class SemiconductorDataset(Dataset):
    """Dataset for semiconductor wafer defect images"""
    
    DEFECT_CLASSES = [
        'wafer_scratch',
        'wafer_particle',
        'wafer_ring',
        'wafer_edge_loc'
    ]
    
    def __init__(self, root_dir: str, transform: Optional[Callable] = None,
                 split: str = 'train'):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.DEFECT_CLASSES)}
        
        # Load all images
        for class_name in self.DEFECT_CLASSES:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        self.samples.append((
                            os.path.join(class_dir, img_name),
                            self.class_to_idx[class_name]
                        ))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255.0
        
        return image, label


class MedicalImageDataset(Dataset):
    """Dataset for medical imaging classification"""
    
    MEDICAL_CLASSES = [
        'med_tumor_small',
        'med_tumor_large',
        'med_inflammation',
        'med_normal'
    ]
    
    def __init__(self, root_dir: str, transform: Optional[Callable] = None,
                 split: str = 'train'):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.MEDICAL_CLASSES)}
        
        # Load all images
        for class_name in self.MEDICAL_CLASSES:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                        self.samples.append((
                            os.path.join(class_dir, img_name),
                            self.class_to_idx[class_name]
                        ))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        if img_path.endswith('.dcm'):
            # Handle DICOM files if needed
            image = self._load_dicom(img_path)
        else:
            image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255.0
        
        return image, label
    
    def _load_dicom(self, path: str):
        """Load DICOM file (placeholder - requires pydicom)"""
        # For now, return a dummy image
        return Image.fromarray(np.zeros((224, 224), dtype=np.uint8))


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing and development"""
    
    def __init__(self, num_samples: int = 1000, img_size: int = 224,
                 num_classes: int = 8, transform: Optional[Callable] = None,
                 seed: int = 42):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.transform = transform
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate synthetic data
        self.images = []
        self.labels = []
        
        for i in range(num_samples):
            label = i % num_classes
            image = self._generate_pattern(label)
            self.images.append(image)
            self.labels.append(label)
    
    def _generate_pattern(self, label: int) -> np.ndarray:
        """Generate synthetic pattern based on label"""
        img = np.random.randn(self.img_size, self.img_size) * 0.1
        
        if label == 0:  # Scratch pattern
            # Horizontal line
            y = self.img_size // 2 + np.random.randint(-20, 20)
            img[y-2:y+2, :] = 1.0
            # Add some noise scratches
            for _ in range(3):
                x1, y1 = np.random.randint(0, self.img_size, 2)
                x2, y2 = np.random.randint(0, self.img_size, 2)
                self._draw_line(img, x1, y1, x2, y2)
        
        elif label == 1:  # Particle pattern
            # Random circular particles
            for _ in range(np.random.randint(10, 20)):
                cx = np.random.randint(10, self.img_size - 10)
                cy = np.random.randint(10, self.img_size - 10)
                radius = np.random.randint(2, 5)
                self._draw_circle(img, cx, cy, radius)
        
        elif label == 2:  # Ring pattern
            # Concentric rings
            cx, cy = self.img_size // 2, self.img_size // 2
            for r in range(20, min(self.img_size // 2, 80), 15):
                self._draw_ring(img, cx, cy, r, r + 5)
        
        elif label == 3:  # Edge defect
            # Edge artifacts
            edge = np.random.choice(['top', 'bottom', 'left', 'right'])
            thickness = np.random.randint(5, 15)
            if edge == 'top':
                img[:thickness, :] = 1.0
            elif edge == 'bottom':
                img[-thickness:, :] = 1.0
            elif edge == 'left':
                img[:, :thickness] = 1.0
            else:
                img[:, -thickness:] = 1.0
        
        elif label == 4:  # Small tumor
            # Small circular region
            cx = np.random.randint(20, self.img_size - 20)
            cy = np.random.randint(20, self.img_size - 20)
            radius = np.random.randint(8, 15)
            self._draw_circle(img, cx, cy, radius, filled=True)
        
        elif label == 5:  # Large tumor
            # Large irregular region
            cx = np.random.randint(30, self.img_size - 30)
            cy = np.random.randint(30, self.img_size - 30)
            radius = np.random.randint(20, 35)
            self._draw_circle(img, cx, cy, radius, filled=True)
            # Add irregularity
            for _ in range(5):
                offset_x = np.random.randint(-10, 10)
                offset_y = np.random.randint(-10, 10)
                r = np.random.randint(10, 20)
                self._draw_circle(img, cx + offset_x, cy + offset_y, r, filled=True)
        
        elif label == 6:  # Inflammation
            # Scattered bright spots
            for _ in range(np.random.randint(30, 50)):
                x = np.random.randint(0, self.img_size)
                y = np.random.randint(0, self.img_size)
                img[max(0, y-1):min(self.img_size, y+2),
                    max(0, x-1):min(self.img_size, x+2)] = np.random.uniform(0.5, 1.0)
        
        elif label == 7:  # Normal
            # Just noise, maybe slight gradient
            gradient = np.linspace(0, 0.2, self.img_size)
            img += gradient.reshape(-1, 1)
        
        # Normalize to [0, 1]
        img = np.clip(img, 0, 1).astype(np.float32)
        return img
    
    def _draw_circle(self, img: np.ndarray, cx: int, cy: int, radius: int, filled: bool = False):
        """Draw a circle on the image"""
        y, x = np.ogrid[:self.img_size, :self.img_size]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        if filled:
            img[mask] = 1.0
        else:
            mask_outer = (x - cx) ** 2 + (y - cy) ** 2 <= (radius + 1) ** 2
            img[mask_outer & ~mask] = 1.0
    
    def _draw_ring(self, img: np.ndarray, cx: int, cy: int, r_inner: int, r_outer: int):
        """Draw a ring on the image"""
        y, x = np.ogrid[:self.img_size, :self.img_size]
        mask_outer = (x - cx) ** 2 + (y - cy) ** 2 <= r_outer ** 2
        mask_inner = (x - cx) ** 2 + (y - cy) ** 2 <= r_inner ** 2
        img[mask_outer & ~mask_inner] = 1.0
    
    def _draw_line(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int):
        """Draw a line on the image"""
        # Simple line drawing (Bresenham's algorithm simplified)
        num_points = max(abs(x2 - x1), abs(y2 - y1))
        if num_points > 0:
            for i in range(num_points):
                t = i / num_points
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                if 0 <= x < self.img_size and 0 <= y < self.img_size:
                    img[y, x] = 1.0
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image for transforms
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        
        if self.transform:
            image = self.transform(image_pil)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        return image, label


class CombinedDataset(Dataset):
    """Combined dataset for both semiconductor and medical images"""
    
    def __init__(self, semiconductor_root: Optional[str] = None,
                 medical_root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 split: str = 'train',
                 use_synthetic: bool = False):
        
        self.datasets = []
        self.cumulative_sizes = []
        cumsum = 0
        
        if use_synthetic:
            # Use synthetic data
            dataset = SyntheticDataset(
                num_samples=1000,
                num_classes=8,
                transform=transform
            )
            self.datasets.append(dataset)
            cumsum += len(dataset)
            self.cumulative_sizes.append(cumsum)
        else:
            # Use real data
            if semiconductor_root and os.path.exists(semiconductor_root):
                semi_dataset = SemiconductorDataset(semiconductor_root, transform, split)
                self.datasets.append(semi_dataset)
                cumsum += len(semi_dataset)
                self.cumulative_sizes.append(cumsum)
            
            if medical_root and os.path.exists(medical_root):
                med_dataset = MedicalImageDataset(medical_root, transform, split)
                # Offset medical labels by 4 to avoid overlap
                med_dataset.class_to_idx = {k: v + 4 for k, v in med_dataset.class_to_idx.items()}
                self.datasets.append(med_dataset)
                cumsum += len(med_dataset)
                self.cumulative_sizes.append(cumsum)
        
        self.num_classes = 8
    
    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        dataset_idx = 0
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                dataset_idx = i
                break
        
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx][sample_idx]