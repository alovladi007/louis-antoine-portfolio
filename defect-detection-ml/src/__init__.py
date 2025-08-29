"""
Defect Detection & Yield Prediction with ML

This package provides machine learning models for semiconductor defect
classification and yield prediction using CNNs and Bayesian methods.
"""

from .data_generator import DefectDataGenerator, SyntheticWaferGenerator
from .cnn_models import DefectCNN, ResNetDefectClassifier, EfficientNetClassifier
from .bayesian_models import BayesianDefectClassifier, RareDefectDetector
from .yield_prediction import YieldPredictor, YieldOptimizer
from .preprocessing import WaferPreprocessor, DefectAugmentation
from .evaluation import ModelEvaluator, YieldAnalyzer

__version__ = "1.0.0"
__all__ = [
    "DefectDataGenerator",
    "SyntheticWaferGenerator", 
    "DefectCNN",
    "ResNetDefectClassifier",
    "EfficientNetClassifier",
    "BayesianDefectClassifier",
    "RareDefectDetector",
    "YieldPredictor",
    "YieldOptimizer",
    "WaferPreprocessor",
    "DefectAugmentation",
    "ModelEvaluator",
    "YieldAnalyzer"
]