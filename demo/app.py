#!/usr/bin/env python3
"""Streamlit demo application for speaker identification."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from speaker_id import (
    SpeakerDataset,
    DataLoader,
    MFCCKNNModel,
    MFCCSVMModel,
    XVectorModel,
    ECAPATDNNModel,
    SpeakerMetrics,
)
from speaker_id.utils.device import get_device
from speaker_id.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Speaker Identification Demo",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Privacy disclaimer
PRIVACY_DISCLAIMER = """
**PRIVACY NOTICE**: This is a research and educational demonstration. 
This system should NOT be used for biometric identification in production 
environments without proper privacy safeguards and user consent. 
Voice cloning and unauthorized biometric identification are prohibited.
"""

def load_model(model_type: str, model_path: str, config: dict):
    """Load trained model."""
    try:
        if model_type == "mfcc_knn":
            model = MFCCKNNModel(config.get("model", {}))
        elif model_type == "mfcc_svm":
            model = MFCCSVMModel(config.get("model", {}))
        elif model_type == "xvector":
            model = XVectorModel(config.get("model", {}))
            model.setup_classifier(config.get("num_speakers", 5))
        elif model_type == "ecapa_tdnn":
            model = ECAPATDNNModel(config.get("model", {}))
            model.setup_classifier(config.get("num_speakers", 5))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def extract_features(audio_data: np.ndarray, sample_rate: int, config: dict):
    """Extract features from audio."""
    try:
        # Create feature extractor
        data_loader = DataLoader(config)
        feature_extractor = data_loader.feature_extractor
        
        # Extract features
        features = feature_extractor.extract(audio_data, sample_rate)
        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def plot_audio_waveform(audio_data: np.ndarray, sample_rate: int):
    """Plot audio waveform."""
    time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio_data,
        mode='lines',
        name='Waveform',
        line=dict(color='blue', width=1)
    ))
    
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=300
    )
    
    return fig

def plot_spectrogram(audio_data: np.ndarray, sample_rate: int):
    """Plot mel-spectrogram."""
    # Extract mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_mels=80,
        n_fft=1024,
        hop_length=512
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Create time and frequency axes
    time_axis = librosa.frames_to_time(
        np.arange(log_mel_spec.shape[1]),
        sr=sample_rate,
        hop_length=512
    )
    freq_axis = librosa.mel_frequencies(n_mels=80, fmin=0, fmax=sample_rate//2)
    
    fig = go.Figure(data=go.Heatmap(
        z=log_mel_spec,
        x=time_axis,
        y=freq_axis,
        colorscale='Viridis',
        colorbar=dict(title="dB")
    ))
    
    fig.update_layout(
        title="Mel-Spectrogram",
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
        height=400
    )
    
    return fig

def plot_prediction_confidence(probabilities: np.ndarray, speaker_names: list):
    """Plot prediction confidence."""
    fig = go.Figure(data=go.Bar(
        x=speaker_names,
        y=probabilities,
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title="Speaker Prediction Confidence",
        xaxis_title="Speaker",
        yaxis_title="Probability",
        height=400
    )
    
    return fig

def main():
    """Main demo application."""
    # Header
    st.title("🎤 Speaker Identification Demo")
    st.markdown(PRIVACY_DISCLAIMER)
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["mfcc_knn", "mfcc_svm", "xvector", "ecapa_tdnn"],
        help="Select the speaker identification model to use"
    )
    
    # Model path
    model_path = st.sidebar.text_input(
        "Model Path",
        value=f"outputs/{model_type}_model.pth",
        help="Path to the trained model file"
    )
    
    # Configuration
    config = {
        "model": {"type": model_type},
        "features": {
            "mfcc": {"n_mfcc": 13, "n_fft": 1024, "hop_length": 512, "n_mels": 80},
            "mel_spec": {"n_fft": 1024, "hop_length": 512, "n_mels": 80}
        },
        "sample_rate": 16000,
        "num_speakers": 5
    }
    
    # Load model
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            model = load_model(model_type, model_path, config)
            if model is not None:
                st.sidebar.success("Model loaded successfully!")
                st.session_state.model = model
            else:
                st.sidebar.error("Failed to load model")
    
    # Main content
    if "model" not in st.session_state:
        st.warning("Please load a model first using the sidebar.")
        return
        
    model = st.session_state.model
    
    # Audio input
    st.header("Audio Input")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Upload an audio file for speaker identification"
    )
    
    # Audio recording
    st.subheader("Or Record Audio")
    audio_bytes = st.audio(
        "Record audio using your microphone",
        format="audio/wav"
    )
    
    if uploaded_file is not None:
        # Process uploaded file
        try:
            # Load audio
            audio_data, sample_rate = librosa.load(uploaded_file, sr=16000)
            
            # Display audio player
            st.audio(uploaded_file, format="audio/wav")
            
            # Extract features
            features = extract_features(audio_data, sample_rate, config)
            
            if features is not None:
                # Make prediction
                if model_type in ["mfcc_knn", "mfcc_svm"]:
                    prediction = model.predict([features])[0]
                    probabilities = model.predict_proba([features])[0]
                else:
                    # Neural model
                    features_tensor = torch.from_numpy(features).float().unsqueeze(0)
                    with torch.no_grad():
                        _, logits = model.forward_with_classification(features_tensor)
                        probabilities = torch.softmax(logits, dim=1).numpy()[0]
                        prediction = np.argmax(probabilities)
                
                # Display results
                st.header("Results")
                
                # Speaker names (placeholder)
                speaker_names = [f"Speaker_{i:03d}" for i in range(len(probabilities))]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Prediction")
                    predicted_speaker = speaker_names[prediction]
                    confidence = probabilities[prediction]
                    
                    st.success(f"**Predicted Speaker**: {predicted_speaker}")
                    st.info(f"**Confidence**: {confidence:.3f}")
                    
                with col2:
                    st.subheader("All Probabilities")
                    for i, (speaker, prob) in enumerate(zip(speaker_names, probabilities)):
                        st.write(f"{speaker}: {prob:.3f}")
                
                # Visualizations
                st.header("Visualizations")
                
                # Audio waveform
                st.subheader("Audio Waveform")
                waveform_fig = plot_audio_waveform(audio_data, sample_rate)
                st.plotly_chart(waveform_fig, use_container_width=True)
                
                # Spectrogram
                st.subheader("Mel-Spectrogram")
                spectrogram_fig = plot_spectrogram(audio_data, sample_rate)
                st.plotly_chart(spectrogram_fig, use_container_width=True)
                
                # Prediction confidence
                st.subheader("Prediction Confidence")
                confidence_fig = plot_prediction_confidence(probabilities, speaker_names)
                st.plotly_chart(confidence_fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing audio file: {e}")
    
    # Model information
    st.header("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Details")
        st.write(f"**Model Type**: {model_type}")
        st.write(f"**Model Path**: {model_path}")
        st.write(f"**Number of Speakers**: {config['num_speakers']}")
        
    with col2:
        st.subheader("Feature Extraction")
        if model_type in ["mfcc_knn", "mfcc_svm"]:
            st.write("**Features**: MFCC (Mel-Frequency Cepstral Coefficients)")
            st.write("**Dimensions**: 13 MFCC coefficients")
        else:
            st.write("**Features**: Mel-Spectrogram")
            st.write("**Dimensions**: 80 mel-frequency bins")
    
    # Instructions
    st.header("Instructions")
    st.markdown("""
    1. **Load Model**: Use the sidebar to select and load a trained model
    2. **Upload Audio**: Upload an audio file or record audio using your microphone
    3. **View Results**: The system will identify the speaker and show confidence scores
    4. **Visualizations**: Explore the audio waveform, spectrogram, and prediction confidence
    
    **Note**: This demo uses synthetic data for demonstration purposes. 
    In a real application, you would need to train the model on your specific dataset.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("**Speaker Identification System** - Research and Educational Demo")
    st.markdown(PRIVACY_DISCLAIMER)

if __name__ == "__main__":
    main()
