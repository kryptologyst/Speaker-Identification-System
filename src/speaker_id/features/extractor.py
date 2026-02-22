"""Base feature extractor class."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import numpy as np
import torch


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize feature extractor.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.sample_rate = config.get("sample_rate", 16000)
        
    @abstractmethod
    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract features from audio.
        
        Args:
            audio: Input audio array.
            sample_rate: Sample rate of audio.
            
        Returns:
            Extracted features.
        """
        pass
        
    def preprocess(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio before feature extraction.
        
        Args:
            audio: Input audio array.
            
        Returns:
            Preprocessed audio.
        """
        # Apply pre-emphasis if configured
        if self.config.get("preemphasis", 0) > 0:
            audio = self._apply_preemphasis(audio, self.config["preemphasis"])
            
        return audio
        
    def _apply_preemphasis(self, audio: np.ndarray, coeff: float) -> np.ndarray:
        """Apply pre-emphasis filter.
        
        Args:
            audio: Input audio array.
            coeff: Pre-emphasis coefficient.
            
        Returns:
            Pre-emphasized audio.
        """
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])
        
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features.
        
        Args:
            features: Input features.
            
        Returns:
            Normalized features.
        """
        if self.config.get("normalize", True):
            # Z-score normalization
            mean = np.mean(features, axis=0, keepdims=True)
            std = np.std(features, axis=0, keepdims=True)
            features = (features - mean) / (std + 1e-8)
            
        return features
        
    def apply_cmvn(self, features: np.ndarray) -> np.ndarray:
        """Apply Cepstral Mean and Variance Normalization.
        
        Args:
            features: Input features.
            
        Returns:
            CMVN-normalized features.
        """
        if self.config.get("cmvn", False):
            # Mean normalization
            features = features - np.mean(features, axis=0, keepdims=True)
            # Variance normalization
            std = np.std(features, axis=0, keepdims=True)
            features = features / (std + 1e-8)
            
        return features
        
    def extract_statistics(self, features: np.ndarray) -> np.ndarray:
        """Extract statistical features (mean, std, etc.).
        
        Args:
            features: Input features.
            
        Returns:
            Statistical features.
        """
        if features.ndim == 2:  # Time series features
            # Extract mean, std, min, max for each feature dimension
            mean_feat = np.mean(features, axis=0)
            std_feat = np.std(features, axis=0)
            min_feat = np.min(features, axis=0)
            max_feat = np.max(features, axis=0)
            
            # Concatenate all statistics
            stats = np.concatenate([mean_feat, std_feat, min_feat, max_feat])
            return stats
        else:
            return features
