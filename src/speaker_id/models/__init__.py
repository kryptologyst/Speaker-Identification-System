"""Speaker identification models."""

from .base import BaseModel
from .mfcc_knn import MFCCKNNModel
from .mfcc_svm import MFCCSVMModel
from .xvector import XVectorModel
from .ecapa_tdnn import ECAPATDNNModel

__all__ = [
    "BaseModel",
    "MFCCKNNModel",
    "MFCCSVMModel",
    "XVectorModel", 
    "ECAPATDNNModel",
]
