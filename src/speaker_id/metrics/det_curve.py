"""Detection Error Trade-off (DET) curve plotting."""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from ..utils.logging import get_logger

logger = get_logger(__name__)


def plot_det_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Detection Error Trade-off (DET) Curve",
) -> None:
    """Plot Detection Error Trade-off (DET) curve.
    
    Args:
        y_true: Binary labels (0 or 1).
        y_scores: Prediction scores.
        save_path: Path to save plot.
        title: Plot title.
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Calculate FNR
    fnr = 1 - tpr
    
    # Convert to percentages
    fpr_pct = fpr * 100
    fnr_pct = fnr * 100
    
    # Create DET curve plot
    plt.figure(figsize=(8, 8))
    
    # Plot DET curve
    plt.plot(fpr_pct, fnr_pct, 'b-', linewidth=2, label='DET Curve')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Random Classifier')
    
    # Find EER point
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer_fpr = fpr_pct[eer_idx]
    eer_fnr = fnr_pct[eer_idx]
    
    # Mark EER point
    plt.plot(eer_fpr, eer_fnr, 'ro', markersize=8, label=f'EER = {eer_fpr:.1f}%')
    
    # Formatting
    plt.xlabel('False Positive Rate (%)')
    plt.ylabel('False Negative Rate (%)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    
    # Use log scale for better visualization
    plt.xscale('log')
    plt.yscale('log')
    
    # Set log scale limits
    plt.xlim(0.1, 100)
    plt.ylim(0.1, 100)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"DET curve saved to {save_path}")
        
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "ROC Curve",
) -> None:
    """Plot Receiver Operating Characteristic (ROC) curve.
    
    Args:
        y_true: Binary labels (0 or 1).
        y_scores: Prediction scores.
        save_path: Path to save plot.
        title: Plot title.
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Calculate AUC
    auc = np.trapz(tpr, fpr)
    
    # Create ROC curve plot
    plt.figure(figsize=(8, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    # Find EER point
    eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
    eer_fpr = fpr[eer_idx]
    eer_tpr = tpr[eer_idx]
    
    # Mark EER point
    plt.plot(eer_fpr, eer_tpr, 'ro', markersize=8, label=f'EER = {eer_fpr:.3f}')
    
    # Formatting
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
        
    plt.show()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Precision-Recall Curve",
) -> None:
    """Plot Precision-Recall curve.
    
    Args:
        y_true: Binary labels (0 or 1).
        y_scores: Prediction scores.
        save_path: Path to save plot.
        title: Plot title.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calculate average precision
    ap = average_precision_score(y_true, y_scores)
    
    # Create precision-recall curve plot
    plt.figure(figsize=(8, 8))
    
    # Plot precision-recall curve
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AP = {ap:.3f})')
    
    # Plot baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Random Classifier (AP = {baseline:.3f})')
    
    # Formatting
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-Recall curve saved to {save_path}")
        
    plt.show()
