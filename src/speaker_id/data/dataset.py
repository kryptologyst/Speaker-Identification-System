"""Speaker dataset implementation."""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils.audio import load_audio
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SpeakerDataset:
    """Speaker identification dataset."""
    
    def __init__(
        self,
        data_dir: str,
        config: Dict,
        anonymize_filenames: bool = True,
    ) -> None:
        """Initialize speaker dataset.
        
        Args:
            data_dir: Directory containing speaker data.
            config: Dataset configuration.
            anonymize_filenames: Whether to anonymize filenames in logs.
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.anonymize_filenames = anonymize_filenames
        
        # Dataset parameters
        self.sample_rate = config.get("sample_rate", 16000)
        self.max_duration = config.get("max_duration", 10.0)
        self.min_duration = config.get("min_duration", 1.0)
        
        # Split parameters
        self.train_split = config.get("train_split", 0.7)
        self.val_split = config.get("val_split", 0.15)
        self.test_split = config.get("test_split", 0.15)
        self.speaker_wise_split = config.get("speaker_wise_split", True)
        
        # Data storage
        self.metadata: Optional[pd.DataFrame] = None
        self.speakers: List[str] = []
        self.speaker_to_id: Dict[str, int] = {}
        
    def load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata.
        
        Returns:
            DataFrame with metadata.
        """
        if self.metadata is not None:
            return self.metadata
            
        logger.info("Loading dataset metadata...")
        
        # Collect all audio files
        audio_files = []
        speakers = []
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist. Creating synthetic dataset.")
            return self._create_synthetic_dataset()
            
        # Walk through directory structure
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    file_path = Path(root) / file
                    speaker = file_path.parent.name
                    
                    audio_files.append(str(file_path))
                    speakers.append(speaker)
                    
        if not audio_files:
            logger.warning("No audio files found. Creating synthetic dataset.")
            return self._create_synthetic_dataset()
            
        # Create metadata DataFrame
        metadata = pd.DataFrame({
            'file_path': audio_files,
            'speaker': speakers,
            'filename': [Path(f).name for f in audio_files],
        })
        
        # Get unique speakers
        self.speakers = sorted(list(set(speakers)))
        self.speaker_to_id = {speaker: idx for idx, speaker in enumerate(self.speakers)}
        
        logger.info(f"Found {len(audio_files)} audio files from {len(self.speakers)} speakers")
        
        # Add speaker IDs
        metadata['speaker_id'] = metadata['speaker'].map(self.speaker_to_id)
        
        self.metadata = metadata
        return metadata
        
    def _create_synthetic_dataset(self) -> pd.DataFrame:
        """Create a synthetic dataset for demonstration.
        
        Returns:
            Synthetic dataset metadata.
        """
        logger.info("Creating synthetic dataset...")
        
        # Create synthetic speakers
        speakers = [f"speaker_{i:03d}" for i in range(5)]
        self.speakers = speakers
        self.speaker_to_id = {speaker: idx for idx, speaker in enumerate(speakers)}
        
        # Generate synthetic file paths
        audio_files = []
        speaker_list = []
        
        for speaker in speakers:
            for i in range(10):  # 10 files per speaker
                filename = f"{speaker}_utterance_{i:03d}.wav"
                file_path = self.data_dir / speaker / filename
                audio_files.append(str(file_path))
                speaker_list.append(speaker)
                
        metadata = pd.DataFrame({
            'file_path': audio_files,
            'speaker': speaker_list,
            'filename': [Path(f).name for f in audio_files],
            'speaker_id': [self.speaker_to_id[s] for s in speaker_list],
        })
        
        self.metadata = metadata
        logger.info(f"Created synthetic dataset with {len(audio_files)} files from {len(speakers)} speakers")
        
        return metadata
        
    def create_splits(self, random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """Create train/validation/test splits.
        
        Args:
            random_state: Random seed for reproducibility.
            
        Returns:
            Dictionary with split DataFrames.
        """
        metadata = self.load_metadata()
        
        if self.speaker_wise_split:
            # Speaker-wise splitting (no speaker leakage)
            splits = self._create_speaker_wise_splits(metadata, random_state)
        else:
            # Random splitting
            splits = self._create_random_splits(metadata, random_state)
            
        logger.info("Dataset splits created:")
        for split_name, split_data in splits.items():
            logger.info(f"  {split_name}: {len(split_data)} samples")
            
        return splits
        
    def _create_speaker_wise_splits(
        self,
        metadata: pd.DataFrame,
        random_state: int,
    ) -> Dict[str, pd.DataFrame]:
        """Create speaker-wise splits.
        
        Args:
            metadata: Dataset metadata.
            random_state: Random seed.
            
        Returns:
            Dictionary with split DataFrames.
        """
        splits = {}
        
        # Split speakers
        speakers = self.speakers.copy()
        random.seed(random_state)
        random.shuffle(speakers)
        
        n_speakers = len(speakers)
        n_train_speakers = int(n_speakers * self.train_split)
        n_val_speakers = int(n_speakers * self.val_split)
        
        train_speakers = speakers[:n_train_speakers]
        val_speakers = speakers[n_train_speakers:n_train_speakers + n_val_speakers]
        test_speakers = speakers[n_train_speakers + n_val_speakers:]
        
        # Create splits based on speakers
        splits['train'] = metadata[metadata['speaker'].isin(train_speakers)]
        splits['val'] = metadata[metadata['speaker'].isin(val_speakers)]
        splits['test'] = metadata[metadata['speaker'].isin(test_speakers)]
        
        return splits
        
    def _create_random_splits(
        self,
        metadata: pd.DataFrame,
        random_state: int,
    ) -> Dict[str, pd.DataFrame]:
        """Create random splits.
        
        Args:
            metadata: Dataset metadata.
            random_state: Random seed.
            
        Returns:
            Dictionary with split DataFrames.
        """
        # First split: train vs temp
        train_data, temp_data = train_test_split(
            metadata,
            test_size=(1 - self.train_split),
            random_state=random_state,
            stratify=metadata['speaker'],
        )
        
        # Second split: val vs test
        val_size = self.val_split / (self.val_split + self.test_split)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_size),
            random_state=random_state,
            stratify=temp_data['speaker'],
        )
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data,
        }
        
    def get_speaker_info(self) -> Dict[str, int]:
        """Get speaker information.
        
        Returns:
            Dictionary mapping speaker names to IDs.
        """
        return self.speaker_to_id.copy()
        
    def get_num_speakers(self) -> int:
        """Get number of speakers.
        
        Returns:
            Number of unique speakers.
        """
        return len(self.speakers)
        
    def save_metadata(self, filepath: str) -> None:
        """Save metadata to file.
        
        Args:
            filepath: Path to save metadata.
        """
        if self.metadata is None:
            self.load_metadata()
            
        self.metadata.to_csv(filepath, index=False)
        logger.info(f"Metadata saved to {filepath}")
        
    def load_metadata_from_file(self, filepath: str) -> pd.DataFrame:
        """Load metadata from file.
        
        Args:
            filepath: Path to metadata file.
            
        Returns:
            Loaded metadata DataFrame.
        """
        self.metadata = pd.read_csv(filepath)
        
        # Recreate speaker mappings
        self.speakers = sorted(self.metadata['speaker'].unique().tolist())
        self.speaker_to_id = {speaker: idx for idx, speaker in enumerate(self.speakers)}
        
        logger.info(f"Metadata loaded from {filepath}")
        return self.metadata
