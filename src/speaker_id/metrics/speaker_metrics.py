"""Speaker identification metrics."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    classification_report,
    confusion_matrix,
)

from .eer import calculate_eer, calculate_min_dcf
from .det_curve import plot_det_curve
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SpeakerMetrics:
    """Speaker identification evaluation metrics."""
    
    def __init__(self, config: Dict) -> None:
        """Initialize speaker metrics.
        
        Args:
            config: Configuration.
        """
        self.config = config
        self.metrics_config = config.get("evaluation", {})
        self.top_k_values = self.metrics_config.get("top_k_values", [1, 5, 10])
        
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        speaker_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate speaker identification performance.
        
        Args:
            y_true: True speaker labels.
            y_pred: Predicted speaker labels.
            y_proba: Predicted probabilities.
            speaker_names: Speaker names for reporting.
            
        Returns:
            Dictionary of metrics.
        """
        metrics = {}
        
        # Basic accuracy
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        
        # Top-k accuracy
        if y_proba is not None:
            for k in self.top_k_values:
                top_k_acc = top_k_accuracy_score(
                    y_true, y_proba, k=k, labels=np.arange(y_proba.shape[1])
                )
                metrics[f"top_{k}_accuracy"] = top_k_acc
                
        # Classification report
        if speaker_names is not None:
            report = classification_report(
                y_true, y_pred, target_names=speaker_names, output_dict=True
            )
            metrics["macro_f1"] = report["macro avg"]["f1-score"]
            metrics["weighted_f1"] = report["weighted avg"]["f1-score"]
            
        # Speaker verification metrics (if probabilities available)
        if y_proba is not None:
            eer_metrics = self._calculate_verification_metrics(y_true, y_proba)
            metrics.update(eer_metrics)
            
        return metrics
        
    def _calculate_verification_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate speaker verification metrics.
        
        Args:
            y_true: True speaker labels.
            y_proba: Predicted probabilities.
            
        Returns:
            Dictionary of verification metrics.
        """
        metrics = {}
        
        # Convert to binary verification problem
        # For each speaker, create positive/negative pairs
        unique_speakers = np.unique(y_true)
        
        scores = []
        labels = []
        
        for speaker in unique_speakers:
            # Positive pairs (same speaker)
            speaker_indices = np.where(y_true == speaker)[0]
            speaker_proba = y_proba[speaker_indices, speaker]
            scores.extend(speaker_proba)
            labels.extend([1] * len(speaker_proba))
            
            # Negative pairs (different speakers)
            other_indices = np.where(y_true != speaker)[0]
            other_proba = y_proba[other_indices, speaker]
            scores.extend(other_proba)
            labels.extend([0] * len(other_proba))
            
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Calculate EER and minDCF
        try:
            eer = calculate_eer(labels, scores)
            min_dcf = calculate_min_dcf(labels, scores)
            
            metrics["eer"] = eer
            metrics["min_dcf"] = min_dcf
            
        except Exception as e:
            logger.warning(f"Could not calculate verification metrics: {e}")
            
        return metrics
        
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        speaker_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True speaker labels.
            y_pred: Predicted speaker labels.
            speaker_names: Speaker names.
            save_path: Path to save plot.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Speaker Identification Confusion Matrix')
        plt.colorbar()
        
        if speaker_names is not None:
            tick_marks = np.arange(len(speaker_names))
            plt.xticks(tick_marks, speaker_names, rotation=45)
            plt.yticks(tick_marks, speaker_names)
            
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
                        
        plt.ylabel('True Speaker')
        plt.xlabel('Predicted Speaker')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
            
        plt.show()
        
    def plot_det_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot Detection Error Trade-off (DET) curve.
        
        Args:
            y_true: True speaker labels.
            y_proba: Predicted probabilities.
            save_path: Path to save plot.
        """
        # Convert to binary verification problem
        unique_speakers = np.unique(y_true)
        
        scores = []
        labels = []
        
        for speaker in unique_speakers:
            # Positive pairs (same speaker)
            speaker_indices = np.where(y_true == speaker)[0]
            speaker_proba = y_proba[speaker_indices, speaker]
            scores.extend(speaker_proba)
            labels.extend([1] * len(speaker_proba))
            
            # Negative pairs (different speakers)
            other_indices = np.where(y_true != speaker)[0]
            other_proba = y_proba[other_indices, speaker]
            scores.extend(other_proba)
            labels.extend([0] * len(other_proba))
            
        scores = np.array(scores)
        labels = np.array(labels)
        
        plot_det_curve(labels, scores, save_path=save_path)
        
    def create_leaderboard(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> str:
        """Create evaluation leaderboard.
        
        Args:
            results: Dictionary of model results.
            save_path: Path to save leaderboard.
            
        Returns:
            Formatted leaderboard string.
        """
        # Create leaderboard
        leaderboard = "Speaker Identification Leaderboard\n"
        leaderboard += "=" * 50 + "\n\n"
        
        # Sort models by accuracy
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1].get("accuracy", 0),
            reverse=True
        )
        
        # Header
        leaderboard += f"{'Model':<20} {'Accuracy':<10} {'Top-5 Acc':<12} {'EER':<8} {'MinDCF':<8}\n"
        leaderboard += "-" * 60 + "\n"
        
        # Results
        for model_name, metrics in sorted_models:
            accuracy = metrics.get("accuracy", 0)
            top_5_acc = metrics.get("top_5_accuracy", 0)
            eer = metrics.get("eer", 0)
            min_dcf = metrics.get("min_dcf", 0)
            
            leaderboard += f"{model_name:<20} {accuracy:<10.3f} {top_5_acc:<12.3f} {eer:<8.3f} {min_dcf:<8.3f}\n"
            
        leaderboard += "\n"
        
        # Detailed metrics for each model
        for model_name, metrics in sorted_models:
            leaderboard += f"{model_name} Detailed Metrics:\n"
            leaderboard += "-" * 30 + "\n"
            
            for metric_name, value in metrics.items():
                leaderboard += f"  {metric_name}: {value:.4f}\n"
                
            leaderboard += "\n"
            
        if save_path:
            with open(save_path, 'w') as f:
                f.write(leaderboard)
            logger.info(f"Leaderboard saved to {save_path}")
            
        return leaderboard
