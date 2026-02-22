"""Evaluator for speaker identification models."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..models import BaseModel, BaseNeuralModel
from ..metrics import SpeakerMetrics
from ..utils.device import get_device
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Evaluator for speaker identification models."""
    
    def __init__(
        self,
        model: BaseModel,
        config: Dict,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate.
            config: Configuration.
            device: Device to use for evaluation.
        """
        self.model = model
        self.config = config
        self.device = device or get_device()
        
        # Setup metrics
        self.metrics = SpeakerMetrics(config)
        
        # Evaluation results
        self.results: Dict[str, Dict[str, float]] = {}
        
    def evaluate_model(
        self,
        test_loader: Optional[DataLoader] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        speaker_names: Optional[List[str]] = None,
        save_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            test_loader: Test data loader (for neural models).
            X_test: Test features (for traditional models).
            y_test: Test labels.
            speaker_names: Speaker names for reporting.
            save_dir: Directory to save results.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if not self.model.is_trained:
            raise ValueError("Model is not trained")
            
        logger.info("Evaluating model...")
        
        if isinstance(self.model, BaseNeuralModel) and test_loader is not None:
            # Evaluate neural model
            metrics = self._evaluate_neural_model(test_loader, speaker_names)
        elif X_test is not None and y_test is not None:
            # Evaluate traditional model
            metrics = self._evaluate_traditional_model(X_test, y_test, speaker_names)
        else:
            raise ValueError("Invalid evaluation data provided")
            
        # Save results
        if save_dir:
            self._save_results(metrics, save_dir)
            
        logger.info("Model evaluation completed")
        return metrics
        
    def _evaluate_neural_model(
        self,
        test_loader: DataLoader,
        speaker_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate neural network model.
        
        Args:
            test_loader: Test data loader.
            speaker_names: Speaker names for reporting.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if not isinstance(self.model, BaseNeuralModel):
            raise ValueError("Model is not a neural network model")
            
        self.model.eval_mode()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                embeddings, logits = self.model.forward_with_classification(data)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
        # Convert to numpy arrays
        y_pred = np.array(all_predictions)
        y_true = np.array(all_labels)
        y_proba = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self.metrics.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            speaker_names=speaker_names,
        )
        
        return metrics
        
    def _evaluate_traditional_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        speaker_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate traditional ML model.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            speaker_names: Speaker names for reporting.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Get probabilities if available
        try:
            y_proba = self.model.predict_proba(X_test)
        except:
            y_proba = None
            
        # Calculate metrics
        metrics = self.metrics.evaluate(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            speaker_names=speaker_names,
        )
        
        return metrics
        
    def _save_results(self, metrics: Dict[str, float], save_dir: str) -> None:
        """Save evaluation results.
        
        Args:
            metrics: Evaluation metrics.
            save_dir: Directory to save results.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Save metrics to file
        metrics_file = os.path.join(save_dir, "evaluation_metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write("Speaker Identification Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
                
        logger.info(f"Evaluation results saved to {save_dir}")
        
    def create_evaluation_report(
        self,
        test_loader: Optional[DataLoader] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        speaker_names: Optional[List[str]] = None,
        save_dir: Optional[str] = None,
    ) -> str:
        """Create comprehensive evaluation report.
        
        Args:
            test_loader: Test data loader (for neural models).
            X_test: Test features (for traditional models).
            y_test: Test labels.
            speaker_names: Speaker names for reporting.
            save_dir: Directory to save report.
            
        Returns:
            Evaluation report string.
        """
        # Get predictions and probabilities
        if isinstance(self.model, BaseNeuralModel) and test_loader is not None:
            y_pred, y_true, y_proba = self._get_neural_predictions(test_loader)
        elif X_test is not None and y_test is not None:
            y_pred = self.model.predict(X_test)
            y_true = y_test
            try:
                y_proba = self.model.predict_proba(X_test)
            except:
                y_proba = None
        else:
            raise ValueError("Invalid evaluation data provided")
            
        # Create report
        report = "Speaker Identification Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Basic metrics
        metrics = self.metrics.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            speaker_names=speaker_names,
        )
        
        report += "Performance Metrics:\n"
        report += "-" * 20 + "\n"
        for metric_name, value in metrics.items():
            report += f"{metric_name}: {value:.4f}\n"
        report += "\n"
        
        # Confusion matrix
        if speaker_names is not None:
            report += "Confusion Matrix:\n"
            report += "-" * 20 + "\n"
            cm = self._create_confusion_matrix_text(y_true, y_pred, speaker_names)
            report += cm + "\n"
            
        # Per-speaker performance
        if speaker_names is not None:
            report += "Per-Speaker Performance:\n"
            report += "-" * 25 + "\n"
            per_speaker_metrics = self._calculate_per_speaker_metrics(
                y_true, y_pred, speaker_names
            )
            for speaker, metrics_dict in per_speaker_metrics.items():
                report += f"{speaker}:\n"
                for metric_name, value in metrics_dict.items():
                    report += f"  {metric_name}: {value:.4f}\n"
                report += "\n"
                
        # Save report
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            report_file = os.path.join(save_dir, "evaluation_report.txt")
            with open(report_file, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {report_file}")
            
        return report
        
    def _get_neural_predictions(
        self,
        test_loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get predictions from neural model.
        
        Args:
            test_loader: Test data loader.
            
        Returns:
            Tuple of (predictions, labels, probabilities).
        """
        if not isinstance(self.model, BaseNeuralModel):
            raise ValueError("Model is not a neural network model")
            
        self.model.eval_mode()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                embeddings, logits = self.model.forward_with_classification(data)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
        return (
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probabilities)
        )
        
    def _create_confusion_matrix_text(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        speaker_names: List[str],
    ) -> str:
        """Create confusion matrix as text.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            speaker_names: Speaker names.
            
        Returns:
            Confusion matrix as text.
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Create header
        header = "    " + " ".join([f"{name[:8]:>8}" for name in speaker_names])
        
        # Create rows
        rows = []
        for i, speaker in enumerate(speaker_names):
            row = f"{speaker[:8]:>8} " + " ".join([f"{cm[i,j]:>8}" for j in range(len(speaker_names))])
            rows.append(row)
            
        return header + "\n" + "\n".join(rows)
        
    def _calculate_per_speaker_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        speaker_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Calculate per-speaker metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            speaker_names: Speaker names.
            
        Returns:
            Dictionary of per-speaker metrics.
        """
        per_speaker_metrics = {}
        
        for i, speaker in enumerate(speaker_names):
            # Get samples for this speaker
            speaker_mask = (y_true == i)
            speaker_true = y_true[speaker_mask]
            speaker_pred = y_pred[speaker_mask]
            
            if len(speaker_true) == 0:
                continue
                
            # Calculate metrics
            accuracy = np.mean(speaker_pred == speaker_true)
            precision = np.mean(speaker_pred[speaker_pred == i] == i) if np.sum(speaker_pred == i) > 0 else 0
            recall = np.mean(speaker_pred[speaker_true == i] == i) if np.sum(speaker_true == i) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_speaker_metrics[speaker] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
            }
            
        return per_speaker_metrics
