"""Data loader for speaker identification."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from ..features import FeatureExtractor, MFCCExtractor, MelSpectrogramExtractor
from ..utils.audio import load_audio, add_noise, speed_perturb
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SpeakerDatasetPyTorch(Dataset):
    """PyTorch dataset for speaker identification."""
    
    def __init__(
        self,
        metadata: pd.DataFrame,
        feature_extractor: FeatureExtractor,
        config: Dict,
        is_training: bool = False,
    ) -> None:
        """Initialize PyTorch dataset.
        
        Args:
            metadata: Dataset metadata.
            feature_extractor: Feature extractor.
            config: Configuration.
            is_training: Whether this is training data.
        """
        self.metadata = metadata
        self.feature_extractor = feature_extractor
        self.config = config
        self.is_training = is_training
        
        # Audio parameters
        self.sample_rate = config.get("sample_rate", 16000)
        self.max_duration = config.get("max_duration", 10.0)
        self.min_duration = config.get("min_duration", 1.0)
        
        # Augmentation parameters
        self.augmentation_config = config.get("augmentation", {})
        self.speed_perturb = self.augmentation_config.get("speed_perturb", False)
        self.noise_augment = self.augmentation_config.get("noise_augment", False)
        self.noise_snr_range = self.augmentation_config.get("noise_snr_range", [10, 30])
        
    def __len__(self) -> int:
        """Get dataset length.
        
        Returns:
            Number of samples.
        """
        return len(self.metadata)
        
    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
        """Get dataset item.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple of (features, speaker_id).
        """
        row = self.metadata.iloc[idx]
        file_path = row['file_path']
        speaker_id = row['speaker_id']
        
        try:
            # Load audio
            audio, sr = load_audio(
                file_path,
                sample_rate=self.sample_rate,
                normalize=True,
            )
            
            # Apply augmentations if training
            if self.is_training:
                audio = self._apply_augmentations(audio, sr)
                
            # Extract features
            features = self.feature_extractor.extract(audio, sr)
            
            # Convert to tensor if needed
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float()
                
            return features, speaker_id
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return zero features as fallback
            if isinstance(self.feature_extractor, MFCCExtractor):
                # MFCC statistics
                features = torch.zeros(self.feature_extractor.n_mfcc * 4)  # mean, std, min, max
            else:
                # Mel-spectrogram
                features = torch.zeros(80, 100)  # Default mel-spectrogram size
                
            return features, speaker_id
            
    def _apply_augmentations(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply data augmentations.
        
        Args:
            audio: Input audio.
            sr: Sample rate.
            
        Returns:
            Augmented audio.
        """
        # Speed perturbation
        if self.speed_perturb:
            speed_factor = np.random.uniform(0.9, 1.1)
            audio = speed_perturb(audio, sr, speed_factor)
            
        # Noise augmentation
        if self.noise_augment:
            snr_db = np.random.uniform(*self.noise_snr_range)
            audio = add_noise(audio, snr_db)
            
        return audio


class DataLoader:
    """Data loader for speaker identification."""
    
    def __init__(
        self,
        config: Dict,
        feature_extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        """Initialize data loader.
        
        Args:
            config: Configuration.
            feature_extractor: Feature extractor.
        """
        self.config = config
        
        # Initialize feature extractor
        if feature_extractor is None:
            self.feature_extractor = self._create_feature_extractor()
        else:
            self.feature_extractor = feature_extractor
            
    def _create_feature_extractor(self) -> FeatureExtractor:
        """Create feature extractor based on config.
        
        Returns:
            Feature extractor.
        """
        model_type = self.config.get("model", {}).get("type", "mfcc_knn")
        
        if model_type in ["mfcc_knn", "mfcc_svm"]:
            return MFCCExtractor(self.config.get("features", {}))
        elif model_type in ["xvector", "ecapa_tdnn"]:
            return MelSpectrogramExtractor(self.config.get("features", {}))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def create_dataloaders(
        self,
        splits: Dict[str, pd.DataFrame],
        batch_size: Optional[int] = None,
        num_workers: int = 4,
    ) -> Dict[str, TorchDataLoader]:
        """Create PyTorch data loaders.
        
        Args:
            splits: Dataset splits.
            batch_size: Batch size.
            num_workers: Number of worker processes.
            
        Returns:
            Dictionary of data loaders.
        """
        if batch_size is None:
            batch_size = self.config.get("training", {}).get("batch_size", 32)
            
        dataloaders = {}
        
        for split_name, metadata in splits.items():
            is_training = (split_name == "train")
            
            dataset = SpeakerDatasetPyTorch(
                metadata=metadata,
                feature_extractor=self.feature_extractor,
                config=self.config,
                is_training=is_training,
            )
            
            dataloader = TorchDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=is_training,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=is_training,
            )
            
            dataloaders[split_name] = dataloader
            
        return dataloaders
        
    def extract_features_batch(
        self,
        metadata: pd.DataFrame,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features for traditional ML models.
        
        Args:
            metadata: Dataset metadata.
            batch_size: Batch size for processing.
            
        Returns:
            Tuple of (features, labels).
        """
        features_list = []
        labels_list = []
        
        for idx, row in metadata.iterrows():
            file_path = row['file_path']
            speaker_id = row['speaker_id']
            
            try:
                # Load audio
                audio, sr = load_audio(
                    file_path,
                    sample_rate=self.config.get("sample_rate", 16000),
                    normalize=True,
                )
                
                # Extract features
                features = self.feature_extractor.extract(audio, sr)
                features_list.append(features)
                labels_list.append(speaker_id)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
                
        return np.array(features_list), np.array(labels_list)
