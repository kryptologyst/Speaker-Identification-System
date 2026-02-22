"""Tests for speaker identification models."""

import pytest
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from speaker_id.models import MFCCKNNModel, MFCCSVMModel, XVectorModel, ECAPATDNNModel


class TestMFCCKNNModel:
    """Test MFCC KNN model."""
    
    def test_initialization(self):
        """Test model initialization."""
        config = {"mfcc_knn": {"n_neighbors": 3}}
        model = MFCCKNNModel(config)
        assert model.n_neighbors == 3
        assert not model.is_trained
        
    def test_training_and_prediction(self):
        """Test model training and prediction."""
        config = {"mfcc_knn": {"n_neighbors": 3}}
        model = MFCCKNNModel(config)
        
        # Create synthetic data
        X_train = np.random.randn(20, 52)  # 13 MFCC * 4 stats
        y_train = np.array([0, 1, 0, 1] * 5)  # 2 speakers
        
        # Train model
        model.fit(X_train, y_train)
        assert model.is_trained
        
        # Make predictions
        X_test = np.random.randn(5, 52)
        predictions = model.predict(X_test)
        assert len(predictions) == 5
        
        # Test probabilities
        probabilities = model.predict_proba(X_test)
        assert probabilities.shape == (5, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestMFCCSVMModel:
    """Test MFCC SVM model."""
    
    def test_initialization(self):
        """Test model initialization."""
        config = {"mfcc_svm": {"kernel": "rbf", "C": 1.0}}
        model = MFCCSVMModel(config)
        assert model.kernel == "rbf"
        assert model.C == 1.0
        assert not model.is_trained
        
    def test_training_and_prediction(self):
        """Test model training and prediction."""
        config = {"mfcc_svm": {"kernel": "rbf", "C": 1.0}}
        model = MFCCSVMModel(config)
        
        # Create synthetic data
        X_train = np.random.randn(20, 52)  # 13 MFCC * 4 stats
        y_train = np.array([0, 1, 0, 1] * 5)  # 2 speakers
        
        # Train model
        model.fit(X_train, y_train)
        assert model.is_trained
        
        # Make predictions
        X_test = np.random.randn(5, 52)
        predictions = model.predict(X_test)
        assert len(predictions) == 5
        
        # Test probabilities
        probabilities = model.predict_proba(X_test)
        assert probabilities.shape == (5, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestXVectorModel:
    """Test X-vector model."""
    
    def test_initialization(self):
        """Test model initialization."""
        config = {
            "xvector": {
                "input_dim": 80,
                "hidden_dims": [512, 512],
                "embedding_dim": 512,
                "dropout": 0.5
            }
        }
        model = XVectorModel(config)
        assert model.input_dim == 80
        assert model.embedding_dim == 512
        assert not model.is_trained
        
    def test_classifier_setup(self):
        """Test classifier setup."""
        config = {
            "xvector": {
                "input_dim": 80,
                "hidden_dims": [512, 512],
                "embedding_dim": 512,
                "dropout": 0.5
            }
        }
        model = XVectorModel(config)
        model.setup_classifier(num_classes=5)
        assert model.num_classes == 5
        assert model.classifier is not None
        
    def test_forward_pass(self):
        """Test forward pass."""
        config = {
            "xvector": {
                "input_dim": 80,
                "hidden_dims": [512, 512],
                "embedding_dim": 512,
                "dropout": 0.5
            }
        }
        model = XVectorModel(config)
        model.setup_classifier(num_classes=5)
        
        # Create synthetic input
        batch_size, seq_len, input_dim = 2, 100, 80
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        embeddings = model.forward(x)
        assert embeddings.shape == (batch_size, 512)
        
        # Forward pass with classification
        embeddings, logits = model.forward_with_classification(x)
        assert embeddings.shape == (batch_size, 512)
        assert logits.shape == (batch_size, 5)


class TestECAPATDNNModel:
    """Test ECAPA-TDNN model."""
    
    def test_initialization(self):
        """Test model initialization."""
        config = {
            "ecapa_tdnn": {
                "input_dim": 80,
                "channels": [512, 512, 512],
                "kernel_sizes": [5, 3, 3],
                "dilations": [1, 2, 3],
                "attention_channels": 128,
                "embedding_dim": 192
            }
        }
        model = ECAPATDNNModel(config)
        assert model.input_dim == 80
        assert model.embedding_dim == 192
        assert not model.is_trained
        
    def test_classifier_setup(self):
        """Test classifier setup."""
        config = {
            "ecapa_tdnn": {
                "input_dim": 80,
                "channels": [512, 512, 512],
                "kernel_sizes": [5, 3, 3],
                "dilations": [1, 2, 3],
                "attention_channels": 128,
                "embedding_dim": 192
            }
        }
        model = ECAPATDNNModel(config)
        model.setup_classifier(num_classes=5)
        assert model.num_classes == 5
        assert model.classifier is not None
        
    def test_forward_pass(self):
        """Test forward pass."""
        config = {
            "ecapa_tdnn": {
                "input_dim": 80,
                "channels": [512, 512, 512],
                "kernel_sizes": [5, 3, 3],
                "dilations": [1, 2, 3],
                "attention_channels": 128,
                "embedding_dim": 192
            }
        }
        model = ECAPATDNNModel(config)
        model.setup_classifier(num_classes=5)
        
        # Create synthetic input
        batch_size, seq_len, input_dim = 2, 100, 80
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        embeddings = model.forward(x)
        assert embeddings.shape == (batch_size, 192)
        
        # Forward pass with classification
        embeddings, logits = model.forward_with_classification(x)
        assert embeddings.shape == (batch_size, 192)
        assert logits.shape == (batch_size, 5)
