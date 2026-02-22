"""MFCC + KNN model for speaker identification."""

from typing import Dict, Any, Optional

import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from .base import BaseModel


class MFCCKNNModel(BaseModel):
    """MFCC features with KNN classifier."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize MFCC KNN model.
        
        Args:
            config: Model configuration.
        """
        super().__init__(config)
        
        # KNN parameters
        knn_config = config.get("mfcc_knn", {})
        self.n_neighbors = knn_config.get("n_neighbors", 3)
        self.weights = knn_config.get("weights", "distance")
        self.metric = knn_config.get("metric", "euclidean")
        
        # Initialize components
        self.knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
        )
        self.label_encoder = LabelEncoder()
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the KNN model.
        
        Args:
            X: Training features (MFCC statistics).
            y: Training labels (speaker IDs).
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train KNN
        self.knn.fit(X, y_encoded)
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
            
        y_pred_encoded = self.knn.predict(X)
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
            
        return self.knn.predict_proba(X)
        
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
        
    def get_neighbors(self, X: np.ndarray, n_neighbors: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """Get nearest neighbors for input samples.
        
        Args:
            X: Input features.
            n_neighbors: Number of neighbors to return.
            
        Returns:
            Tuple of (distances, indices).
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
            
        distances, indices = self.knn.kneighbors(X, n_neighbors=n_neighbors)
        return distances, indices
        
    def save(self, filepath: str) -> None:
        """Save model to file.
        
        Args:
            filepath: Path to save model.
        """
        model_data = {
            "knn": self.knn,
            "label_encoder": self.label_encoder,
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
        
        self.knn = model_data["knn"]
        self.label_encoder = model_data["label_encoder"]
        self.config = model_data["config"]
        self.is_trained = model_data["is_trained"]
