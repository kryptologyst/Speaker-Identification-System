#!/usr/bin/env python3
"""Training script for speaker identification."""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import numpy as np
import torch

from speaker_id import (
    SpeakerDataset,
    DataLoader,
    MFCCKNNModel,
    MFCCSVMModel,
    XVectorModel,
    ECAPATDNNModel,
    Trainer,
    Evaluator,
    SpeakerMetrics,
)
from speaker_id.utils.device import get_device, set_seed
from speaker_id.utils.logging import setup_logging, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train speaker identification model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Path to output directory"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["mfcc_knn", "mfcc_svm", "xvector", "ecapa_tdnn"],
        help="Model type to train"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, model_type: str, num_speakers: int):
    """Create model instance."""
    model_config = config.get("model", {})
    model_config["type"] = model_type
    
    if model_type == "mfcc_knn":
        return MFCCKNNModel(model_config)
    elif model_type == "mfcc_svm":
        return MFCCSVMModel(model_config)
    elif model_type == "xvector":
        model = XVectorModel(model_config)
        model.setup_classifier(num_speakers)
        return model
    elif model_type == "ecapa_tdnn":
        model = ECAPATDNNModel(model_config)
        model.setup_classifier(num_speakers)
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_type:
        config["model"]["type"] = args.model_type
    if args.device != "auto":
        config["device"] = args.device
    if args.seed != 42:
        config["seed"] = args.seed
        
    # Set random seed
    set_seed(config["seed"], config.get("deterministic", True))
    
    # Setup device
    device = get_device(config.get("device", "auto"))
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = SpeakerDataset(args.data_dir, config)
    metadata = dataset.load_metadata()
    
    # Create splits
    splits = dataset.create_splits(random_state=config["seed"])
    num_speakers = dataset.get_num_speakers()
    speaker_names = dataset.speakers
    
    logger.info(f"Dataset: {len(metadata)} samples, {num_speakers} speakers")
    logger.info(f"Splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    # Create model
    model_type = config["model"]["type"]
    model = create_model(config, model_type, num_speakers)
    logger.info(f"Created model: {model_type}")
    
    # Create trainer
    trainer = Trainer(model, config, device)
    
    # Train model
    if model_type in ["mfcc_knn", "mfcc_svm"]:
        # Traditional ML models
        logger.info("Training traditional ML model...")
        
        # Create data loader for feature extraction
        data_loader = DataLoader(config)
        
        # Extract features
        logger.info("Extracting features...")
        X_train, y_train = data_loader.extract_features_batch(splits["train"])
        X_val, y_val = data_loader.extract_features_batch(splits["val"])
        X_test, y_test = data_loader.extract_features_batch(splits["test"])
        
        # Train model
        trainer.train_traditional_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluator = Evaluator(model, config, device)
        test_metrics = evaluator.evaluate_model(X_test=X_test, y_test=y_test, speaker_names=speaker_names)
        
    else:
        # Neural network models
        logger.info("Training neural network model...")
        
        # Create data loaders
        data_loader = DataLoader(config)
        dataloaders = data_loader.create_dataloaders(splits)
        
        # Train model
        checkpoint_dir = output_dir / "checkpoints"
        trainer.train_neural_model(
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            checkpoint_dir=str(checkpoint_dir)
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluator = Evaluator(model, config, device)
        test_metrics = evaluator.evaluate_model(
            test_loader=dataloaders["test"],
            speaker_names=speaker_names
        )
        
    # Create evaluation report
    logger.info("Creating evaluation report...")
    metrics = SpeakerMetrics(config)
    
    # Create leaderboard
    results = {model_type: test_metrics}
    leaderboard = metrics.create_leaderboard(results)
    
    # Save results
    results_file = output_dir / "results.txt"
    with open(results_file, 'w') as f:
        f.write(leaderboard)
        
    # Save model
    model_file = output_dir / f"{model_type}_model.pth"
    model.save(str(model_file))
    
    # Save configuration
    config_file = output_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    logger.info(f"Training completed. Results saved to {output_dir}")
    logger.info("Final metrics:")
    for metric_name, value in test_metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
