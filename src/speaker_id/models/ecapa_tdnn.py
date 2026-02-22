"""ECAPA-TDNN model for speaker identification."""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import BaseNeuralModel


class ECAPATDNNModel(BaseNeuralModel):
    """ECAPA-TDNN speaker embedding model."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize ECAPA-TDNN model.
        
        Args:
            config: Model configuration.
        """
        super().__init__(config)
        
        # Model parameters
        ecapa_config = config.get("ecapa_tdnn", {})
        self.input_dim = ecapa_config.get("input_dim", 80)
        self.channels = ecapa_config.get("channels", [512, 512, 512, 512, 1536])
        self.kernel_sizes = ecapa_config.get("kernel_sizes", [5, 3, 3, 3, 1])
        self.dilations = ecapa_config.get("dilations", [1, 2, 3, 4, 1])
        self.attention_channels = ecapa_config.get("attention_channels", 128)
        self.embedding_dim = ecapa_config.get("embedding_dim", 192)
        
        # Initialize model
        self.model = ECAPATDNNNetwork(
            input_dim=self.input_dim,
            channels=self.channels,
            kernel_sizes=self.kernel_sizes,
            dilations=self.dilations,
            attention_channels=self.attention_channels,
            embedding_dim=self.embedding_dim,
        )
        
        # Training components
        self.num_classes: Optional[int] = None
        self.classifier: Optional[nn.Linear] = None
        
    def setup_classifier(self, num_classes: int) -> None:
        """Setup classification head.
        
        Args:
            num_classes: Number of speaker classes.
        """
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.embedding_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input mel-spectrograms.
            
        Returns:
            Speaker embeddings.
        """
        return self.model(x)
        
    def forward_with_classification(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with classification.
        
        Args:
            x: Input mel-spectrograms.
            
        Returns:
            Tuple of (embeddings, logits).
        """
        embeddings = self.forward(x)
        
        if self.classifier is None:
            raise ValueError("Classifier not initialized")
            
        logits = self.classifier(embeddings)
        return embeddings, logits
        
    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract speaker embeddings.
        
        Args:
            x: Input mel-spectrograms.
            
        Returns:
            Speaker embeddings.
        """
        self.eval_mode()
        with torch.no_grad():
            embeddings = self.forward(x)
        return embeddings
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model (placeholder - use Trainer class).
        
        Args:
            X: Training features.
            y: Training labels.
        """
        raise NotImplementedError("Use Trainer class for training neural models")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features.
            
        Returns:
            Predictions.
        """
        if not self.is_trained or self.classifier is None:
            raise ValueError("Model not trained")
            
        self.eval_mode()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            _, logits = self.forward_with_classification(X_tensor)
            predictions = torch.argmax(logits, dim=1)
            return predictions.cpu().numpy()
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input features.
            
        Returns:
            Class probabilities.
        """
        if not self.is_trained or self.classifier is None:
            raise ValueError("Model not trained")
            
        self.eval_mode()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            _, logits = self.forward_with_classification(X_tensor)
            probabilities = F.softmax(logits, dim=1)
            return probabilities.cpu().numpy()


class ECAPATDNNNetwork(nn.Module):
    """ECAPA-TDNN neural network architecture."""
    
    def __init__(
        self,
        input_dim: int = 80,
        channels: list[int] = [512, 512, 512, 512, 1536],
        kernel_sizes: list[int] = [5, 3, 3, 3, 1],
        dilations: list[int] = [1, 2, 3, 4, 1],
        attention_channels: int = 128,
        embedding_dim: int = 192,
    ) -> None:
        """Initialize ECAPA-TDNN network.
        
        Args:
            input_dim: Input feature dimension.
            channels: Channel dimensions for each layer.
            kernel_sizes: Kernel sizes for each layer.
            dilations: Dilation rates for each layer.
            attention_channels: Attention channel dimension.
            embedding_dim: Final embedding dimension.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        
        # Input projection
        self.input_projection = nn.Conv1d(input_dim, channels[0], kernel_size=1)
        
        # TDNN layers
        self.tdnn_layers = nn.ModuleList()
        prev_channels = channels[0]
        
        for i in range(len(channels) - 1):
            self.tdnn_layers.append(
                TDNNBlock(
                    in_channels=prev_channels,
                    out_channels=channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                )
            )
            prev_channels = channels[i + 1]
            
        # SE-Res2Block
        self.se_res2block = SERes2Block(
            channels=channels[-1],
            scale=8,
        )
        
        # Attentive statistics pooling
        self.attention = AttentiveStatisticsPooling(
            channels=channels[-1],
            attention_channels=attention_channels,
        )
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.BatchNorm1d(channels[-1] * 2),
            nn.Linear(channels[-1] * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input mel-spectrograms (batch_size, seq_len, input_dim).
            
        Returns:
            Speaker embeddings (batch_size, embedding_dim).
        """
        # Transpose for Conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_projection(x)
        
        # TDNN layers
        for tdnn_layer in self.tdnn_layers:
            x = tdnn_layer(x)
            
        # SE-Res2Block
        x = self.se_res2block(x)
        
        # Attentive statistics pooling
        pooled = self.attention(x)  # (batch_size, channels * 2)
        
        # Final layers
        embeddings = self.final_layers(pooled)
        
        return embeddings


class TDNNBlock(nn.Module):
    """Time Delay Neural Network block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        """Initialize TDNN block.
        
        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Convolution kernel size.
            dilation: Dilation rate.
        """
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features.
            
        Returns:
            Output features.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SERes2Block(nn.Module):
    """Squeeze-and-Excitation Res2Block."""
    
    def __init__(self, channels: int, scale: int = 8) -> None:
        """Initialize SE-Res2Block.
        
        Args:
            channels: Number of channels.
            scale: Scale factor for Res2Net.
        """
        super().__init__()
        
        self.channels = channels
        self.scale = scale
        
        # Res2Net layers
        self.res2net_layers = nn.ModuleList()
        for i in range(scale):
            self.res2net_layers.append(
                nn.Conv1d(
                    channels // scale,
                    channels // scale,
                    kernel_size=3,
                    padding=1,
                )
            )
            
        # SE module
        self.se_module = SEModule(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features.
            
        Returns:
            Output features.
        """
        # Split channels
        chunks = torch.chunk(x, self.scale, dim=1)
        
        # Apply Res2Net layers
        for i, chunk in enumerate(chunks):
            if i == 0:
                out = self.res2net_layers[i](chunk)
            else:
                out = out + self.res2net_layers[i](chunk)
                
        # Concatenate
        out = torch.cat([out] + chunks[1:], dim=1)
        
        # Apply SE module
        out = self.se_module(out)
        
        return out


class SEModule(nn.Module):
    """Squeeze-and-Excitation module."""
    
    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize SE module.
        
        Args:
            channels: Number of channels.
            reduction: Reduction factor.
        """
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features.
            
        Returns:
            Output features.
        """
        b, c, _ = x.size()
        
        # Global average pooling
        y = self.avg_pool(x).view(b, c)
        
        # FC layers
        y = self.fc(y).view(b, c, 1)
        
        # Scale
        return x * y


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling."""
    
    def __init__(self, channels: int, attention_channels: int = 128) -> None:
        """Initialize attentive statistics pooling.
        
        Args:
            channels: Number of input channels.
            attention_channels: Number of attention channels.
        """
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(attention_channels, channels, kernel_size=1),
            nn.Softmax(dim=2),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features (batch_size, channels, seq_len).
            
        Returns:
            Pooled features (batch_size, channels * 2).
        """
        # Calculate attention weights
        attention_weights = self.attention(x)  # (batch_size, channels, seq_len)
        
        # Apply attention
        attended = x * attention_weights  # (batch_size, channels, seq_len)
        
        # Calculate statistics
        mean = torch.sum(attended, dim=2)  # (batch_size, channels)
        std = torch.sqrt(
            torch.sum(attention_weights * (x - mean.unsqueeze(2)) ** 2, dim=2)
        )  # (batch_size, channels)
        
        # Concatenate mean and std
        pooled = torch.cat([mean, std], dim=1)  # (batch_size, channels * 2)
        
        return pooled
