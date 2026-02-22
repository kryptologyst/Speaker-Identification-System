"""Base model class for speaker identification."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class BaseModel(ABC):
    """Abstract base class for speaker identification models."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize model.
        
        Args:
            config: Model configuration.
        """
        self.config = config
        self.device = torch.device("cpu")
        self.is_trained = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model.
        
        Args:
            X: Training features.
            y: Training labels.
        """
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features.
            
        Returns:
            Predictions.
        """
        pass
        
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input features.
            
        Returns:
            Class probabilities.
        """
        pass
        
    def save(self, filepath: str) -> None:
        """Save model to file.
        
        Args:
            filepath: Path to save model.
        """
        raise NotImplementedError("Save method not implemented")
        
    def load(self, filepath: str) -> None:
        """Load model from file.
        
        Args:
            filepath: Path to load model from.
        """
        raise NotImplementedError("Load method not implemented")
        
    def to_device(self, device: torch.device) -> None:
        """Move model to device.
        
        Args:
            device: Target device.
        """
        self.device = device


class BaseNeuralModel(BaseModel):
    """Base class for neural network models."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize neural model.
        
        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[nn.Module] = None
        
    def to_device(self, device: torch.device) -> None:
        """Move model to device.
        
        Args:
            device: Target device.
        """
        super().to_device(device)
        if self.model is not None:
            self.model = self.model.to(device)
            
    def save(self, filepath: str) -> None:
        """Save model to file.
        
        Args:
            filepath: Path to save model.
        """
        if self.model is None:
            raise ValueError("Model not initialized")
            
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "is_trained": self.is_trained,
        }, filepath)
        
    def load(self, filepath: str) -> None:
        """Load model from file.
        
        Args:
            filepath: Path to load model from.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.model is None:
            raise ValueError("Model not initialized")
            
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.is_trained = checkpoint.get("is_trained", False)
        
    def train_mode(self) -> None:
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()
            
    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
