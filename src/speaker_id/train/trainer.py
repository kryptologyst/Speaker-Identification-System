"""Trainer for speaker identification models."""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models import BaseModel, BaseNeuralModel
from ..utils.device import get_device, set_seed
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """Trainer for speaker identification models."""
    
    def __init__(
        self,
        model: BaseModel,
        config: Dict,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: Model to train.
            config: Training configuration.
            device: Device to use for training.
        """
        self.model = model
        self.config = config
        self.device = device or get_device()
        
        # Training parameters
        self.training_config = config.get("training", {})
        self.batch_size = self.training_config.get("batch_size", 32)
        self.num_epochs = self.training_config.get("num_epochs", 100)
        self.learning_rate = self.training_config.get("learning_rate", 0.001)
        self.weight_decay = self.training_config.get("weight_decay", 1e-4)
        self.patience = self.training_config.get("patience", 10)
        
        # Optimizer and scheduler
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[nn.Module] = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
        
        # Setup model
        self._setup_model()
        
    def _setup_model(self) -> None:
        """Setup model for training."""
        if isinstance(self.model, BaseNeuralModel):
            self.model.to_device(self.device)
            self._setup_neural_training()
        else:
            logger.info("Training traditional ML model")
            
    def _setup_neural_training(self) -> None:
        """Setup neural network training components."""
        if not isinstance(self.model, BaseNeuralModel):
            return
            
        # Setup optimizer
        optimizer_name = self.training_config.get("optimizer", "adam")
        if optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
        # Setup scheduler
        scheduler_name = self.training_config.get("scheduler", "cosine")
        if scheduler_name.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs
            )
        elif scheduler_name.lower() == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif scheduler_name.lower() == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
            
        # Setup loss function
        loss_name = self.training_config.get("loss", "cross_entropy")
        if loss_name.lower() == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
            
    def train_traditional_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train traditional ML model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
        """
        logger.info("Training traditional ML model...")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = np.mean(val_pred == y_val)
            logger.info(f"Validation accuracy: {val_acc:.4f}")
            
        self.model.is_trained = True
        logger.info("Traditional ML model training completed")
        
    def train_neural_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        """Train neural network model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            checkpoint_dir: Directory to save checkpoints.
        """
        if not isinstance(self.model, BaseNeuralModel):
            raise ValueError("Model is not a neural network model")
            
        logger.info("Training neural network model...")
        
        # Setup checkpoint directory
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
        # Training loop
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader)
            else:
                val_loss, val_acc = float('inf'), 0.0
                
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                if checkpoint_dir:
                    self._save_checkpoint(checkpoint_dir, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                    
        self.model.is_trained = True
        logger.info("Neural network model training completed")
        
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Tuple of (average_loss, accuracy).
        """
        if not isinstance(self.model, BaseNeuralModel):
            raise ValueError("Model is not a neural network model")
            
        self.model.train_mode()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            embeddings, logits = self.model.forward_with_classification(data)
            loss = self.criterion(logits, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
            
        return total_loss / len(train_loader), correct / total
        
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Tuple of (average_loss, accuracy).
        """
        if not isinstance(self.model, BaseNeuralModel):
            raise ValueError("Model is not a neural network model")
            
        self.model.eval_mode()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                embeddings, logits = self.model.forward_with_classification(data)
                loss = self.criterion(logits, target)
                
                # Statistics
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
        return total_loss / len(val_loader), correct / total
        
    def _save_checkpoint(self, checkpoint_dir: str, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            checkpoint_dir: Checkpoint directory.
            is_best: Whether this is the best model.
        """
        if not isinstance(self.model, BaseNeuralModel):
            return
            
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config,
        }
        
        # Save current checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint.
        """
        if not isinstance(self.model, BaseNeuralModel):
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history.
        
        Args:
            save_path: Path to save plot.
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.training_history['train_loss'], label='Train Loss')
        ax1.plot(self.training_history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.training_history['train_acc'], label='Train Acc')
        ax2.plot(self.training_history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
            
        plt.show()
