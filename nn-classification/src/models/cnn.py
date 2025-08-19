"""
CNN architectures with attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from .attention import CBAM, SEBlock


class ConvBlock(nn.Module):
    """Basic convolutional block with batch norm and activation"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, use_bn: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with optional attention"""
    
    def __init__(self, channels: int, attention_type: Optional[str] = None):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        
        if attention_type == 'cbam':
            self.attention = CBAM(channels)
        elif attention_type == 'se':
            self.attention = SEBlock(channels)
        else:
            self.attention = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.attention(out)
        out += residual
        return self.relu(out)


class CNNClassifier(nn.Module):
    """Base CNN classifier for semiconductor and medical imaging"""
    
    def __init__(self, num_classes: int = 8, in_channels: int = 1,
                 base_channels: int = 32, num_blocks: List[int] = [2, 2, 2, 2]):
        super().__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            ConvBlock(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Feature extraction stages
        self.stages = nn.ModuleList()
        channels = base_channels
        for i, num in enumerate(num_blocks):
            stage = []
            out_channels = base_channels * (2 ** i)
            
            # First block may downsample
            if i > 0:
                stage.append(ConvBlock(channels, out_channels, stride=2, padding=1))
            else:
                stage.append(ConvBlock(channels, out_channels))
            
            # Additional blocks
            for _ in range(num - 1):
                stage.append(ResidualBlock(out_channels))
            
            self.stages.append(nn.Sequential(*stage))
            channels = out_channels
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(channels // 2, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.global_pool(x).flatten(1)
        x = self.classifier(x)
        return x


class CNNWithAttention(nn.Module):
    """CNN with integrated attention mechanisms"""
    
    def __init__(self, num_classes: int = 8, in_channels: int = 1,
                 attention_type: str = 'cbam', dropout: float = 0.5):
        super().__init__()
        
        # Feature extraction with attention
        self.features = nn.Sequential(
            # Stage 1
            ConvBlock(in_channels, 32, kernel_size=3),
            ConvBlock(32, 32, kernel_size=3),
            CBAM(32) if attention_type == 'cbam' else SEBlock(32),
            nn.MaxPool2d(2, 2),
            
            # Stage 2
            ConvBlock(32, 64, kernel_size=3),
            ConvBlock(64, 64, kernel_size=3),
            CBAM(64) if attention_type == 'cbam' else SEBlock(64),
            nn.MaxPool2d(2, 2),
            
            # Stage 3
            ConvBlock(64, 128, kernel_size=3),
            ConvBlock(128, 128, kernel_size=3),
            CBAM(128) if attention_type == 'cbam' else SEBlock(128),
            nn.MaxPool2d(2, 2),
            
            # Stage 4
            ConvBlock(128, 256, kernel_size=3),
            ConvBlock(256, 256, kernel_size=3),
            CBAM(256) if attention_type == 'cbam' else SEBlock(256),
        )
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.global_pool(features).flatten(1)
        output = self.classifier(pooled)
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representations"""
        features = self.features(x)
        return self.global_pool(features).flatten(1)