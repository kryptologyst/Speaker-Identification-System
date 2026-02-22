"""Speaker Identification System for Research and Education.

This package provides a modern, privacy-preserving speaker identification system
suitable for research and educational purposes. It includes both traditional
machine learning approaches (MFCC + KNN/SVM) and deep learning models
(x-vector, ECAPA-TDNN).

PRIVACY NOTICE: This system is designed for research and educational purposes only.
It should not be used for biometric identification in production environments
without proper privacy safeguards and user consent.
"""

__version__ = "1.0.0"
__author__ = "AI Projects"
__email__ = "ai@example.com"

from .models import (
    MFCCKNNModel,
    MFCCSVMModel,
    XVectorModel,
    ECAPATDNNModel,
)
from .data import SpeakerDataset, DataLoader
from .features import FeatureExtractor
from .metrics import SpeakerMetrics
from .train import Trainer
from .eval import Evaluator

__all__ = [
    "MFCCKNNModel",
    "MFCCSVMModel", 
    "XVectorModel",
    "ECAPATDNNModel",
    "SpeakerDataset",
    "DataLoader",
    "FeatureExtractor",
    "SpeakerMetrics",
    "Trainer",
    "Evaluator",
]
