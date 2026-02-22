"""Mel-spectrogram feature extraction."""

from typing import Dict, Any

import librosa
import numpy as np
import torch
import torchaudio

from .extractor import FeatureExtractor


class MelSpectrogramExtractor(FeatureExtractor):
    """Mel-spectrogram feature extractor for deep learning models."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize mel-spectrogram extractor.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
        
        # Mel-spectrogram parameters
        mel_config = config.get("mel_spec", {})
        self.n_fft = mel_config.get("n_fft", 1024)
        self.hop_length = mel_config.get("hop_length", 512)
        self.n_mels = mel_config.get("n_mels", 80)
        self.f_min = mel_config.get("f_min", 0)
        self.f_max = mel_config.get("f_max", 8000)
        
        # Create mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
        )
        
    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract mel-spectrogram features from audio.
        
        Args:
            audio: Input audio array.
            sample_rate: Sample rate of audio.
            
        Returns:
            Mel-spectrogram features.
        """
        # Preprocess audio
        audio = self.preprocess(audio)
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Extract mel-spectrogram
        mel_spec = self.mel_transform(audio_tensor)
        
        # Convert to log scale
        log_mel_spec = torch.log(mel_spec + 1e-8)
        
        # Convert back to numpy
        log_mel_spec_np = log_mel_spec.numpy()
        
        # Apply CMVN if configured
        if self.config.get("cmvn", False):
            log_mel_spec_np = self.apply_cmvn(log_mel_spec_np)
            
        return log_mel_spec_np
        
    def extract_librosa(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract mel-spectrogram using librosa (alternative implementation).
        
        Args:
            audio: Input audio array.
            sample_rate: Sample rate of audio.
            
        Returns:
            Mel-spectrogram features.
        """
        # Preprocess audio
        audio = self.preprocess(audio)
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Apply CMVN if configured
        if self.config.get("cmvn", False):
            log_mel_spec = self.apply_cmvn(log_mel_spec)
            
        return log_mel_spec
