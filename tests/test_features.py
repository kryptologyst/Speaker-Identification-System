"""Tests for feature extraction."""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from speaker_id.features import MFCCExtractor, MelSpectrogramExtractor


class TestMFCCExtractor:
    """Test MFCC feature extractor."""
    
    def test_initialization(self):
        """Test extractor initialization."""
        config = {
            "mfcc": {
                "n_mfcc": 13,
                "n_fft": 1024,
                "hop_length": 512,
                "n_mels": 80
            },
            "extract_stats": True
        }
        extractor = MFCCExtractor(config)
        assert extractor.n_mfcc == 13
        assert extractor.n_fft == 1024
        
    def test_feature_extraction(self):
        """Test feature extraction."""
        config = {
            "mfcc": {
                "n_mfcc": 13,
                "n_fft": 1024,
                "hop_length": 512,
                "n_mels": 80
            },
            "extract_stats": True
        }
        extractor = MFCCExtractor(config)
        
        # Create synthetic audio
        sample_rate = 16000
        duration = 2.0
        audio = np.random.randn(int(sample_rate * duration))
        
        # Extract features
        features = extractor.extract(audio, sample_rate)
        
        # Check feature dimensions
        expected_dim = 13 * 4  # 13 MFCC * 4 statistics
        assert features.shape == (expected_dim,)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
        
    def test_preprocessing(self):
        """Test audio preprocessing."""
        config = {
            "mfcc": {
                "n_mfcc": 13,
                "n_fft": 1024,
                "hop_length": 512,
                "n_mels": 80
            },
            "preemphasis": 0.97,
            "extract_stats": True
        }
        extractor = MFCCExtractor(config)
        
        # Create synthetic audio
        sample_rate = 16000
        duration = 2.0
        audio = np.random.randn(int(sample_rate * duration))
        
        # Test preprocessing
        preprocessed = extractor.preprocess(audio)
        assert len(preprocessed) == len(audio)
        assert not np.any(np.isnan(preprocessed))


class TestMelSpectrogramExtractor:
    """Test Mel-spectrogram feature extractor."""
    
    def test_initialization(self):
        """Test extractor initialization."""
        config = {
            "mel_spec": {
                "n_fft": 1024,
                "hop_length": 512,
                "n_mels": 80,
                "f_min": 0,
                "f_max": 8000
            },
            "sample_rate": 16000
        }
        extractor = MelSpectrogramExtractor(config)
        assert extractor.n_fft == 1024
        assert extractor.n_mels == 80
        
    def test_feature_extraction(self):
        """Test feature extraction."""
        config = {
            "mel_spec": {
                "n_fft": 1024,
                "hop_length": 512,
                "n_mels": 80,
                "f_min": 0,
                "f_max": 8000
            },
            "sample_rate": 16000
        }
        extractor = MelSpectrogramExtractor(config)
        
        # Create synthetic audio
        sample_rate = 16000
        duration = 2.0
        audio = np.random.randn(int(sample_rate * duration))
        
        # Extract features
        features = extractor.extract(audio, sample_rate)
        
        # Check feature dimensions
        expected_frames = int(duration * sample_rate / 512) + 1
        assert features.shape == (80, expected_frames)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
        
    def test_librosa_extraction(self):
        """Test librosa-based extraction."""
        config = {
            "mel_spec": {
                "n_fft": 1024,
                "hop_length": 512,
                "n_mels": 80,
                "f_min": 0,
                "f_max": 8000
            },
            "sample_rate": 16000
        }
        extractor = MelSpectrogramExtractor(config)
        
        # Create synthetic audio
        sample_rate = 16000
        duration = 2.0
        audio = np.random.randn(int(sample_rate * duration))
        
        # Extract features using librosa
        features = extractor.extract_librosa(audio, sample_rate)
        
        # Check feature dimensions
        expected_frames = int(duration * sample_rate / 512) + 1
        assert features.shape == (80, expected_frames)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
