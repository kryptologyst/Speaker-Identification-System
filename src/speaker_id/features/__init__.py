"""Feature extraction for speaker identification."""

from .extractor import FeatureExtractor
from .mfcc import MFCCExtractor
from .mel_spec import MelSpectrogramExtractor

__all__ = [
    "FeatureExtractor",
    "MFCCExtractor", 
    "MelSpectrogramExtractor",
]
