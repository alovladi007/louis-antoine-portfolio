"""
CNN Models for Defect Classification

This module implements various CNN architectures for semiconductor
defect classification including custom CNNs and transfer learning models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple, Dict, Any
import numpy as np


class DefectCNN(nn.Module):
    """
    Custom CNN architecture for defect classification
    """
    
    def __init__(self, num_classes: int = 5, input_channels: int = 1,
                 dropout_rate: float = 0.5):
        """
        Initialize DefectCNN
        
        Args:
            num_classes: Number of defect classes
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            dropout_rate: Dropout rate for regularization
        """
        super(DefectCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Conv Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Conv Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor, layer: str = 'conv4') -> torch.Tensor:
        """
        Get intermediate feature maps for visualization
        
        Args:
            x: Input tensor
            layer: Name of layer to extract features from
            
        Returns:
            Feature maps from specified layer
        """
        features = {}
        
        # Conv Block 1
        x = self.conv1(x)
        features['conv1'] = x
        x = self.pool(F.relu(self.bn1(x)))
        
        # Conv Block 2
        x = self.conv2(x)
        features['conv2'] = x
        x = self.pool(F.relu(self.bn2(x)))
        
        # Conv Block 3
        x = self.conv3(x)
        features['conv3'] = x
        x = self.pool(F.relu(self.bn3(x)))
        
        # Conv Block 4
        x = self.conv4(x)
        features['conv4'] = x
        
        return features.get(layer, features['conv4'])


class ResNetDefectClassifier(nn.Module):
    """
    ResNet-based defect classifier with transfer learning
    """
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True,
                 freeze_backbone: bool = False, resnet_version: str = 'resnet50'):
        """
        Initialize ResNet classifier
        
        Args:
            num_classes: Number of defect classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone layers
            resnet_version: ResNet version ('resnet18', 'resnet34', 'resnet50', 'resnet101')
        """
        super(ResNetDefectClassifier, self).__init__()
        
        # Load ResNet model
        if resnet_version == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = 512
        elif resnet_version == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            in_features = 512
        elif resnet_version == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = 2048
        elif resnet_version == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            in_features = 2048
        else:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final FC layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Add input adaptation for grayscale images
        self.input_adapt = nn.Conv2d(1, 3, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        # Adapt grayscale to RGB if needed
        if x.shape[1] == 1:
            x = self.input_adapt(x)
        
        return self.backbone(x)
    
    def unfreeze_layers(self, n_layers: int = -1):
        """
        Unfreeze last n layers of backbone
        
        Args:
            n_layers: Number of layers to unfreeze (-1 for all)
        """
        if n_layers == -1:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Get all layers
            layers = list(self.backbone.children())
            
            # Unfreeze last n layers
            for layer in layers[-n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-based defect classifier
    """
    
    def __init__(self, num_classes: int = 5, efficientnet_version: str = 'b0',
                 pretrained: bool = True):
        """
        Initialize EfficientNet classifier
        
        Args:
            num_classes: Number of defect classes
            efficientnet_version: EfficientNet version (b0-b7)
            pretrained: Whether to use pretrained weights
        """
        super(EfficientNetClassifier, self).__init__()
        
        # Note: This requires torchvision >= 0.11
        model_name = f'efficientnet_{efficientnet_version}'
        
        if efficientnet_version == 'b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = 1280
        elif efficientnet_version == 'b1':
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            in_features = 1280
        else:
            # For other versions, use b0 as fallback
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = 1280
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Input adaptation for grayscale
        self.input_adapt = nn.Conv2d(1, 3, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if x.shape[1] == 1:
            x = self.input_adapt(x)
        return self.backbone(x)


class AttentionDefectCNN(nn.Module):
    """
    CNN with attention mechanism for defect classification
    """
    
    def __init__(self, num_classes: int = 5, input_channels: int = 1):
        """
        Initialize Attention CNN
        
        Args:
            num_classes: Number of defect classes
            input_channels: Number of input channels
        """
        super(AttentionDefectCNN, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Attention module
        self.attention = SpatialAttention(512)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            Tuple of (logits, attention_map)
        """
        # Encode features
        features = self.encoder(x)
        
        # Apply attention
        attended_features, attention_map = self.attention(features)
        
        # Global pooling
        pooled = self.global_pool(attended_features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits, attention_map


class SpatialAttention(nn.Module):
    """
    Spatial attention module
    """
    
    def __init__(self, in_channels: int):
        """
        Initialize spatial attention
        
        Args:
            in_channels: Number of input channels
        """
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply spatial attention
        
        Args:
            x: Input features
            
        Returns:
            Tuple of (attended features, attention map)
        """
        # Generate attention map
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = torch.sigmoid(attention)
        
        # Apply attention
        attended = x * attention
        
        return attended, attention


class MultiScaleDefectCNN(nn.Module):
    """
    Multi-scale CNN for detecting defects of various sizes
    """
    
    def __init__(self, num_classes: int = 5, input_channels: int = 1):
        """
        Initialize multi-scale CNN
        
        Args:
            num_classes: Number of defect classes
            input_channels: Number of input channels
        """
        super(MultiScaleDefectCNN, self).__init__()
        
        # Multiple branches for different scales
        self.branch1 = self._make_branch(input_channels, 64, kernel_size=3)
        self.branch2 = self._make_branch(input_channels, 64, kernel_size=5)
        self.branch3 = self._make_branch(input_channels, 64, kernel_size=7)
        
        # Fusion layer
        self.fusion = nn.Conv2d(192, 256, 1)
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def _make_branch(self, in_channels: int, out_channels: int,
                    kernel_size: int) -> nn.Module:
        """Create a branch with specific kernel size"""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Process through multiple scales
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        
        # Concatenate features
        multi_scale = torch.cat([feat1, feat2, feat3], dim=1)
        
        # Fusion
        fused = self.fusion(multi_scale)
        
        # Final processing
        features = self.final_conv(fused)
        
        # Global pooling and classification
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        logits = self.classifier(pooled)
        
        return logits


def create_model(model_name: str = 'defect_cnn', 
                num_classes: int = 5,
                **kwargs) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_name: Name of the model to create
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments
        
    Returns:
        Created model
    """
    models_dict = {
        'defect_cnn': DefectCNN,
        'resnet': ResNetDefectClassifier,
        'efficientnet': EfficientNetClassifier,
        'attention_cnn': AttentionDefectCNN,
        'multiscale_cnn': MultiScaleDefectCNN
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models_dict[model_name](num_classes=num_classes, **kwargs)


# Example usage
if __name__ == "__main__":
    # Create different models
    models_to_test = [
        ('defect_cnn', {}),
        ('resnet', {'resnet_version': 'resnet50'}),
        ('attention_cnn', {}),
        ('multiscale_cnn', {})
    ]
    
    # Test input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 224, 224)
    
    for model_name, kwargs in models_to_test:
        print(f"\nTesting {model_name}...")
        model = create_model(model_name, num_classes=5, **kwargs)
        
        # Forward pass
        if model_name == 'attention_cnn':
            output, attention = model(input_tensor)
            print(f"  Output shape: {output.shape}")
            print(f"  Attention shape: {attention.shape}")
        else:
            output = model(input_tensor)
            print(f"  Output shape: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")