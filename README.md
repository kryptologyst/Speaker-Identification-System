# Speaker Identification System

Privacy-preserving speaker identification system for research and educational purposes. This project implements both traditional machine learning approaches (MFCC + KNN/SVM) and advanced deep learning models (x-vector, ECAPA-TDNN) for speaker identification.

## ⚠️ PRIVACY NOTICE

**This system is designed for research and educational purposes only.** It should NOT be used for biometric identification in production environments without proper privacy safeguards and user consent. Voice cloning and unauthorized biometric identification are prohibited.

## Features

- **Multiple Model Architectures**: Traditional ML (MFCC+KNN/SVM) and Deep Learning (x-vector, ECAPA-TDNN)
- **Comprehensive Evaluation**: Accuracy, EER, minDCF, DET curves, confusion matrices
- **Privacy-Preserving**: Anonymized logging, clear disclaimers, research-focused design
- **Modern Stack**: PyTorch 2.x, torchaudio, librosa, scikit-learn
- **Interactive Demo**: Streamlit web application for real-time testing
- **Reproducible**: Deterministic seeding, comprehensive configuration
- **Production-Ready**: Clean code, type hints, comprehensive documentation

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA/MPS support (optional)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Speaker-Identification-System.git
   cd Speaker-Identification-System
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or for development:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Verify installation**:
   ```bash
   python -c "import speaker_id; print('Installation successful!')"
   ```

## Quick Start

### 1. Prepare Data

Place your audio data in the following structure:
```
data/raw/
├── speaker_001/
│   ├── utterance_001.wav
│   ├── utterance_002.wav
│   └── ...
├── speaker_002/
│   ├── utterance_001.wav
│   └── ...
└── ...
```

### 2. Train a Model

```bash
# Train MFCC + KNN model
python scripts/train.py --model_type mfcc_knn --data_dir data/raw

# Train x-vector model
python scripts/train.py --model_type xvector --data_dir data/raw

# Train ECAPA-TDNN model
python scripts/train.py --model_type ecapa_tdnn --data_dir data/raw
```

### 3. Run the Demo

```bash
streamlit run demo/app.py
```

## Model Architectures

### Traditional Machine Learning

- **MFCC + KNN**: Mel-frequency cepstral coefficients with k-nearest neighbors
- **MFCC + SVM**: Mel-frequency cepstral coefficients with support vector machine

### Deep Learning

- **X-vector**: Time-delay neural network with statistics pooling
- **ECAPA-TDNN**: Enhanced channel attention and temporal context aggregation

## Configuration

The system uses YAML configuration files. Key parameters:

```yaml
# Model configuration
model:
  type: "mfcc_knn"  # or "mfcc_svm", "xvector", "ecapa_tdnn"

# Feature extraction
features:
  mfcc:
    n_mfcc: 13
    n_fft: 1024
    hop_length: 512

# Training
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001

# Data
data:
  sample_rate: 16000
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Top-k Accuracy**: Accuracy for top-k predictions
- **EER**: Equal Error Rate for speaker verification
- **minDCF**: Minimum Detection Cost Function
- **DET Curves**: Detection Error Trade-off curves
- **Confusion Matrices**: Per-speaker performance analysis

## Project Structure

```
speaker-identification/
├── src/speaker_id/           # Main package
│   ├── models/              # Model implementations
│   ├── data/                # Data handling
│   ├── features/            # Feature extraction
│   ├── metrics/             # Evaluation metrics
│   ├── train/               # Training utilities
│   ├── eval/                # Evaluation utilities
│   └── utils/               # Utility functions
├── configs/                 # Configuration files
├── data/                    # Data directory
├── scripts/                 # Training scripts
├── demo/                    # Demo application
├── tests/                   # Unit tests
├── assets/                  # Generated assets
└── outputs/                 # Model outputs
```

## API Usage

### Basic Usage

```python
from speaker_id import SpeakerDataset, MFCCKNNModel, Trainer

# Load dataset
dataset = SpeakerDataset("data/raw", config)
splits = dataset.create_splits()

# Create model
model = MFCCKNNModel(config["model"])

# Train model
trainer = Trainer(model, config)
trainer.train_traditional_model(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### Advanced Usage

```python
from speaker_id import XVectorModel, DataLoader, Evaluator

# Create neural model
model = XVectorModel(config["model"])
model.setup_classifier(num_speakers=10)

# Create data loaders
data_loader = DataLoader(config)
dataloaders = data_loader.create_dataloaders(splits)

# Train model
trainer = Trainer(model, config)
trainer.train_neural_model(dataloaders["train"], dataloaders["val"])

# Evaluate model
evaluator = Evaluator(model, config)
metrics = evaluator.evaluate_model(dataloaders["test"])
```

## Demo Application

The Streamlit demo provides an interactive interface for:

- **Model Loading**: Load trained models
- **Audio Upload**: Upload audio files for identification
- **Real-time Prediction**: Get speaker predictions with confidence scores
- **Visualizations**: View waveforms, spectrograms, and confidence plots
- **Model Information**: Display model details and configuration

### Running the Demo

```bash
streamlit run demo/app.py
```

## Privacy and Ethics

This system includes several privacy-preserving features:

- **Anonymized Logging**: Filenames and metadata are anonymized in logs
- **Clear Disclaimers**: Prominent warnings about research-only usage
- **No PII Storage**: No personally identifiable information is stored
- **Educational Focus**: Designed for learning and research, not production

## Limitations

- **Research Purpose**: Not suitable for production biometric identification
- **Limited Dataset**: Demo uses synthetic data
- **No Real-time Processing**: Batch processing only
- **No Voice Cloning**: Speaker identification only, no voice synthesis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{speaker_identification_2026,
  title={Speaker Identification System for Research and Education},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Speaker-Identification-System}
}
```

## Support

For questions and support:

- Create an issue on GitHub
- Check the documentation
- Review the demo application

## Acknowledgments

- PyTorch team for the deep learning framework
- SpeechBrain for speaker recognition research
- Librosa for audio processing
- Scikit-learn for machine learning utilities

---

**Remember**: This system is for research and educational purposes only. Always respect privacy and obtain proper consent when working with voice data.
# Speaker-Identification-System
