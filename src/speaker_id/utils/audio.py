"""Audio processing utilities."""

from typing import Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio(
    file_path: str,
    sample_rate: int = 16000,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load audio file with librosa.
    
    Args:
        file_path: Path to audio file.
        sample_rate: Target sample rate.
        mono: Whether to convert to mono.
        normalize: Whether to normalize audio.
        
    Returns:
        Tuple of (audio_array, sample_rate).
    """
    try:
        audio, sr = librosa.load(
            file_path,
            sr=sample_rate,
            mono=mono,
            dtype=np.float32,
        )
        
        if normalize:
            audio = normalize_audio(audio)
            
        return audio, sr
        
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}: {e}")


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio to target sample rate.
    
    Args:
        audio: Input audio array.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.
        
    Returns:
        Resampled audio array.
    """
    if orig_sr == target_sr:
        return audio
        
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range.
    
    Args:
        audio: Input audio array.
        
    Returns:
        Normalized audio array.
    """
    if np.max(np.abs(audio)) > 0:
        return audio / np.max(np.abs(audio))
    return audio


def apply_preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter to audio.
    
    Args:
        audio: Input audio array.
        coeff: Pre-emphasis coefficient.
        
    Returns:
        Pre-emphasized audio array.
    """
    return np.append(audio[0], audio[1:] - coeff * audio[:-1])


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    top_db: float = 20,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Trim silence from audio.
    
    Args:
        audio: Input audio array.
        sample_rate: Sample rate.
        top_db: Silence threshold in dB.
        frame_length: Frame length for analysis.
        hop_length: Hop length for analysis.
        
    Returns:
        Trimmed audio array.
    """
    return librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]


def add_noise(
    audio: np.ndarray,
    snr_db: float = 20,
    noise_type: str = "white",
) -> np.ndarray:
    """Add noise to audio.
    
    Args:
        audio: Input audio array.
        snr_db: Signal-to-noise ratio in dB.
        noise_type: Type of noise ("white", "pink", "brown").
        
    Returns:
        Noisy audio array.
    """
    # Calculate signal power
    signal_power = np.mean(audio ** 2)
    
    # Calculate noise power based on SNR
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Generate noise
    if noise_type == "white":
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    elif noise_type == "pink":
        # Simplified pink noise (1/f)
        white_noise = np.random.normal(0, 1, len(audio))
        freqs = np.fft.fftfreq(len(audio))
        pink_filter = 1 / np.sqrt(np.abs(freqs) + 1e-8)
        pink_filter[0] = 1  # DC component
        noise = np.real(np.fft.ifft(np.fft.fft(white_noise) * pink_filter))
        noise = noise * np.sqrt(noise_power) / np.std(noise)
    else:  # brown noise
        white_noise = np.random.normal(0, 1, len(audio))
        brown_filter = 1 / (np.abs(np.fft.fftfreq(len(audio))) + 1e-8)
        brown_filter[0] = 1
        noise = np.real(np.fft.ifft(np.fft.fft(white_noise) * brown_filter))
        noise = noise * np.sqrt(noise_power) / np.std(noise)
    
    return audio + noise


def speed_perturb(audio: np.ndarray, sample_rate: int, speed_factor: float) -> np.ndarray:
    """Apply speed perturbation to audio.
    
    Args:
        audio: Input audio array.
        sample_rate: Sample rate.
        speed_factor: Speed factor (>1 for faster, <1 for slower).
        
    Returns:
        Speed-perturbed audio array.
    """
    return librosa.effects.time_stretch(audio, rate=speed_factor)
