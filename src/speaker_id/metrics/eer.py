"""Equal Error Rate (EER) calculation."""

from typing import Tuple

import numpy as np
from sklearn.metrics import roc_curve


def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Calculate Equal Error Rate (EER).
    
    Args:
        y_true: Binary labels (0 or 1).
        y_scores: Prediction scores.
        
    Returns:
        Equal Error Rate.
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find threshold where FPR = 1 - TPR (EER point)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]
    
    return eer


def calculate_min_dcf(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """Calculate minimum Detection Cost Function (minDCF).
    
    Args:
        y_true: Binary labels (0 or 1).
        y_scores: Prediction scores.
        p_target: Prior probability of target.
        c_miss: Cost of miss.
        c_fa: Cost of false alarm.
        
    Returns:
        Minimum Detection Cost Function.
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Calculate DCF for each threshold
    fnr = 1 - tpr
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    
    # Find minimum DCF
    min_dcf = np.min(dcf)
    
    return min_dcf


def calculate_dcf(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """Calculate Detection Cost Function (DCF) for a specific threshold.
    
    Args:
        y_true: Binary labels (0 or 1).
        y_scores: Prediction scores.
        threshold: Decision threshold.
        p_target: Prior probability of target.
        c_miss: Cost of miss.
        c_fa: Cost of false alarm.
        
    Returns:
        Detection Cost Function.
    """
    # Apply threshold
    y_pred = (y_scores >= threshold).astype(int)
    
    # Calculate FPR and FNR
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Calculate DCF
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    
    return dcf
