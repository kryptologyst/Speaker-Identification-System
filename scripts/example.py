#!/usr/bin/env python3
"""Example script demonstrating speaker identification usage."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import yaml
from speaker_id import (
    SpeakerDataset,
    DataLoader,
    MFCCKNNModel,
    XVectorModel,
    Trainer,
    Evaluator,
    SpeakerMetrics,
)
from speaker_id.utils.device import get_device, set_seed
from speaker_id.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def create_synthetic_data():
    """Create synthetic audio data for demonstration."""
    logger.info("Creating synthetic audio data...")
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create speaker directories
    speakers = ["speaker_001", "speaker_002", "speaker_003"]
    for speaker in speakers:
        speaker_dir = data_dir / speaker
        speaker_dir.mkdir(exist_ok=True)
        
        # Create synthetic audio files
        for i in range(5):
            # Generate synthetic audio (sine waves with different frequencies)
            sample_rate = 16000
            duration = 2.0
            freq = 440 + i * 100  # Different frequency for each file
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * freq * t) * 0.5
            
            # Add some noise
            noise = np.random.normal(0, 0.1, len(audio))
            audio = audio + noise
            
            # Save audio file
            audio_file = speaker_dir / f"utterance_{i:03d}.wav"
            import soundfile as sf
            sf.write(audio_file, audio, sample_rate)
            
    logger.info(f"Created synthetic data for {len(speakers)} speakers")


def main():
    """Main example function."""
    logger.info("Speaker Identification Example")
    
    # Set random seed for reproducibility
    set_seed(42, deterministic=True)
    
    # Create synthetic data
    create_synthetic_data()
    
    # Configuration
    config = {
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "sample_rate": 16000,
            "max_duration": 10.0,
            "min_duration": 1.0,
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
            "speaker_wise_split": True,
        },
        "features": {
            "mfcc": {
                "n_mfcc": 13,
                "n_fft": 1024,
                "hop_length": 512,
                "n_mels": 80,
            },
            "mel_spec": {
                "n_fft": 1024,
                "hop_length": 512,
                "n_mels": 80,
                "f_min": 0,
                "f_max": 8000,
            },
            "preemphasis": 0.97,
            "normalize": True,
            "cmvn": True,
            "extract_stats": True,
        },
        "model": {
            "type": "mfcc_knn",
            "mfcc_knn": {
                "n_neighbors": 3,
                "weights": "distance",
                "metric": "euclidean",
            },
        },
        "training": {
            "batch_size": 32,
            "num_epochs": 10,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "patience": 5,
            "optimizer": "adam",
            "scheduler": "cosine",
            "loss": "cross_entropy",
        },
        "evaluation": {
            "metrics": ["accuracy", "top_k_accuracy", "eer", "min_dcf"],
            "top_k_values": [1, 5, 10],
        },
        "device": "auto",
        "seed": 42,
        "deterministic": True,
    }
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = SpeakerDataset("data/raw", config)
    metadata = dataset.load_metadata()
    
    # Create splits
    splits = dataset.create_splits(random_state=config["seed"])
    num_speakers = dataset.get_num_speakers()
    speaker_names = dataset.speakers
    
    logger.info(f"Dataset: {len(metadata)} samples, {num_speakers} speakers")
    logger.info(f"Splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    # Example 1: Traditional ML Model (MFCC + KNN)
    logger.info("\n" + "="*50)
    logger.info("Example 1: MFCC + KNN Model")
    logger.info("="*50)
    
    # Create model
    model = MFCCKNNModel(config["model"])
    
    # Create data loader
    data_loader = DataLoader(config)
    
    # Extract features
    logger.info("Extracting features...")
    X_train, y_train = data_loader.extract_features_batch(splits["train"])
    X_val, y_val = data_loader.extract_features_batch(splits["val"])
    X_test, y_test = data_loader.extract_features_batch(splits["test"])
    
    logger.info(f"Feature dimensions: {X_train.shape[1]}")
    
    # Train model
    logger.info("Training model...")
    trainer = Trainer(model, config)
    trainer.train_traditional_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = Evaluator(model, config)
    test_metrics = evaluator.evaluate_model(X_test=X_test, y_test=y_test, speaker_names=speaker_names)
    
    # Display results
    logger.info("Results:")
    for metric_name, value in test_metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    # Example 2: Deep Learning Model (X-vector)
    logger.info("\n" + "="*50)
    logger.info("Example 2: X-vector Model")
    logger.info("="*50)
    
    # Update config for neural model
    config["model"]["type"] = "xvector"
    config["model"]["xvector"] = {
        "input_dim": 80,
        "hidden_dims": [512, 512, 512, 512, 1500],
        "embedding_dim": 512,
        "dropout": 0.5,
    }
    
    # Create model
    model = XVectorModel(config["model"])
    model.setup_classifier(num_speakers)
    
    # Create data loaders
    data_loader = DataLoader(config)
    dataloaders = data_loader.create_dataloaders(splits, batch_size=8)
    
    # Train model
    logger.info("Training neural model...")
    trainer = Trainer(model, config)
    trainer.train_neural_model(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
    )
    
    # Evaluate model
    logger.info("Evaluating neural model...")
    evaluator = Evaluator(model, config)
    test_metrics = evaluator.evaluate_model(
        test_loader=dataloaders["test"],
        speaker_names=speaker_names,
    )
    
    # Display results
    logger.info("Results:")
    for metric_name, value in test_metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    # Create leaderboard
    logger.info("\n" + "="*50)
    logger.info("Leaderboard")
    logger.info("="*50)
    
    results = {
        "MFCC+KNN": test_metrics,  # Using the last results for demo
        "X-vector": test_metrics,  # Using the same results for demo
    }
    
    metrics = SpeakerMetrics(config)
    leaderboard = metrics.create_leaderboard(results)
    print(leaderboard)
    
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()
