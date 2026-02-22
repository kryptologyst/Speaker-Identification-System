"""MFCC feature extraction."""

from typing import Dict, Any

import librosa
import numpy as np

from .extractor import FeatureExtractor


class MFCCExtractor(FeatureExtractor):
    """MFCC feature extractor."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize MFCC extractor.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
        
        # MFCC parameters
        mfcc_config = config.get("mfcc", {})
        self.n_mfcc = mfcc_config.get("n_mfcc", 13)
        self.n_fft = mfcc_config.get("n_fft", 1024)
        self.hop_length = mfcc_config.get("hop_length", 512)
        self.n_mels = mfcc_config.get("n_mels", 80)
        
    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract MFCC features from audio.
        
        Args:
            audio: Input audio array.
            sample_rate: Sample rate of audio.
            
        Returns:
            MFCC features.
        """
        # Preprocess audio
        audio = self.preprocess(audio)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        
        # Apply CMVN if configured
        mfcc = self.apply_cmvn(mfcc)
        
        # Extract statistics (mean across time for traditional ML)
        if self.config.get("extract_stats", True):
            mfcc_stats = self.extract_statistics(mfcc.T)  # Transpose for time x features
            return mfcc_stats
        else:
            return mfcc.T  # Return time x features
            
    def extract_delta_features(self, mfcc: np.ndarray) -> np.ndarray:
        """Extract delta and delta-delta features.
        
        Args:
            mfcc: MFCC features.
            
        Returns:
            Combined MFCC + delta + delta-delta features.
        """
        # Delta features (first derivative)
        delta = librosa.feature.delta(mfcc.T).T
        
        # Delta-delta features (second derivative)
        delta_delta = librosa.feature.delta(mfcc.T, order=2).T
        
        # Combine all features
        combined = np.concatenate([mfcc.T, delta, delta_delta], axis=1)
        
        return combined
