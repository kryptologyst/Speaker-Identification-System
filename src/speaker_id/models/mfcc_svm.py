"""MFCC + SVM model for speaker identification."""

from typing import Dict, Any, Optional

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multiclass import OneVsRestClassifier

from .base import BaseModel


class MFCCSVMModel(BaseModel):
    """MFCC features with SVM classifier."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize MFCC SVM model.
        
        Args:
            config: Model configuration.
        """
        super().__init__(config)
        
        # SVM parameters
        svm_config = config.get("mfcc_svm", {})
        self.kernel = svm_config.get("kernel", "rbf")
        self.C = svm_config.get("C", 1.0)
        self.gamma = svm_config.get("gamma", "scale")
        
        # Initialize components
        self.svm = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=True,  # Enable probability estimates
            random_state=42,
        )
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the SVM model.
        
        Args:
            X: Training features (MFCC statistics).
            y: Training labels (speaker IDs).
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train SVM
        self.svm.fit(X_scaled, y_encoded)
        self.is_trained = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features.
            
        Returns:
            Predicted speaker labels.
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        X_scaled = self.scaler.transform(X)
        y_pred_encoded = self.svm.predict(X_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        return y_pred
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input features.
            
        Returns:
            Class probabilities.
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        X_scaled = self.scaler.transform(X)
        return self.svm.predict_proba(X_scaled)
        
    def predict_with_confidence(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence scores.
        
        Args:
            X: Input features.
            
        Returns:
            Tuple of (predictions, confidence_scores).
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        # Get probabilities
        probabilities = self.predict_proba(X)
        
        # Get predictions
        predictions = self.predict(X)
        
        # Calculate confidence as max probability
        confidence_scores = np.max(probabilities, axis=1)
        
        return predictions, confidence_scores
        
    def get_decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get decision function values.
        
        Args:
            X: Input features.
            
        Returns:
            Decision function values.
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        X_scaled = self.scaler.transform(X)
        return self.svm.decision_function(X_scaled)
        
    def get_support_vectors(self) -> np.ndarray:
        """Get support vectors.
        
        Returns:
            Support vectors.
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        return self.scaler.inverse_transform(self.svm.support_vectors_)
        
    def save(self, filepath: str) -> None:
        """Save model to file.
        
        Args:
            filepath: Path to save model.
        """
        model_data = {
            "svm": self.svm,
            "label_encoder": self.label_encoder,
            "scaler": self.scaler,
            "config": self.config,
            "is_trained": self.is_trained,
        }
        joblib.dump(model_data, filepath)
        
    def load(self, filepath: str) -> None:
        """Load model from file.
        
        Args:
            filepath: Path to load model from.
        """
        model_data = joblib.load(filepath)
        
        self.svm = model_data["svm"]
        self.label_encoder = model_data["label_encoder"]
        self.scaler = model_data["scaler"]
        self.config = model_data["config"]
        self.is_trained = model_data["is_trained"]
