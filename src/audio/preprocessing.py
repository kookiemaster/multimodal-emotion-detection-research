"""
Audio preprocessing utilities for emotion recognition

This module provides functions for preprocessing audio data for emotion recognition,
including loading audio files, extracting features, and preparing data for model training.
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
from tqdm import tqdm

def load_audio_file(file_path, sr=22050):
    """
    Load an audio file and return the signal
    
    Args:
        file_path: Path to the audio file
        sr: Sample rate to resample audio
        
    Returns:
        Audio signal as numpy array
    """
    try:
        signal, _ = librosa.load(file_path, sr=sr)
        return signal
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_melspectrogram(signal, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    """
    Extract mel-spectrogram from audio signal
    
    Args:
        signal: Audio signal
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
        
    Returns:
        Mel-spectrogram
    """
    if signal is None or len(signal) == 0:
        return None
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=signal, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels
    )
    
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec

def pad_or_truncate(array, target_length):
    """
    Pad or truncate array to target length
    
    Args:
        array: Input array
        target_length: Target length
        
    Returns:
        Padded or truncated array
    """
    if len(array) > target_length:
        return array[:target_length]
    else:
        return np.pad(array, (0, target_length - len(array)), 'constant')

def create_time_windows(spectrogram, window_size=128, time_step=64):
    """
    Create time windows from spectrogram for time distributed processing
    
    Args:
        spectrogram: Mel-spectrogram
        window_size: Size of each window
        time_step: Step size between windows
        
    Returns:
        List of time windows
    """
    # Get spectrogram dimensions
    n_mels, n_frames = spectrogram.shape
    
    # Create time windows
    windows = []
    for i in range(0, max(1, n_frames - window_size + 1), time_step):
        if i + window_size <= n_frames:
            window = spectrogram[:, i:i+window_size]
            windows.append(window)
    
    # If no windows were created (audio too short), create at least one
    if len(windows) == 0:
        # Pad spectrogram to window_size
        padded_spec = np.zeros((n_mels, window_size))
        padded_spec[:, :min(n_frames, window_size)] = spectrogram[:, :min(n_frames, window_size)]
        windows.append(padded_spec)
    
    return windows

def prepare_audio_data(metadata_path, max_samples_per_class=None, window_size=128, time_step=64):
    """
    Prepare audio data for model training
    
    Args:
        metadata_path: Path to metadata CSV file
        max_samples_per_class: Maximum number of samples per class (for balancing)
        window_size: Size of each time window
        time_step: Step size between windows
        
    Returns:
        X_train, y_train, X_test, y_test
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    # Map emotions to indices
    emotion_to_index = {
        'neutral': 0,
        'calm': 1,
        'happy': 2,
        'sad': 3,
        'angry': 4,
        'fearful': 5,
        'disgust': 6,
        'surprised': 7
    }
    
    # Balance classes if needed
    if max_samples_per_class is not None:
        balanced_metadata = []
        for emotion in emotion_to_index.keys():
            emotion_samples = metadata[metadata['emotion'] == emotion]
            if len(emotion_samples) > max_samples_per_class:
                emotion_samples = emotion_samples.sample(max_samples_per_class, random_state=42)
            balanced_metadata.append(emotion_samples)
        metadata = pd.concat(balanced_metadata)
    
    # Shuffle data
    metadata = metadata.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train and test (80/20)
    train_size = int(0.8 * len(metadata))
    train_metadata = metadata[:train_size]
    test_metadata = metadata[train_size:]
    
    # Process training data
    X_train = []
    y_train = []
    
    print("Processing training data...")
    for _, row in tqdm(train_metadata.iterrows(), total=len(train_metadata)):
        # Load audio file
        signal = load_audio_file(row['file_path'])
        if signal is None:
            continue
        
        # Extract mel-spectrogram
        mel_spec = extract_melspectrogram(signal)
        if mel_spec is None:
            continue
        
        # Create time windows
        windows = create_time_windows(mel_spec, window_size, time_step)
        
        # Add to training data
        for window in windows:
            # Add channel dimension
            window = np.expand_dims(window, axis=-1)
            X_train.append(window)
            y_train.append(emotion_to_index[row['emotion']])
    
    # Process test data
    X_test = []
    y_test = []
    
    print("Processing test data...")
    for _, row in tqdm(test_metadata.iterrows(), total=len(test_metadata)):
        # Load audio file
        signal = load_audio_file(row['file_path'])
        if signal is None:
            continue
        
        # Extract mel-spectrogram
        mel_spec = extract_melspectrogram(signal)
        if mel_spec is None:
            continue
        
        # Create time windows
        windows = create_time_windows(mel_spec, window_size, time_step)
        
        # Add to test data
        for window in windows:
            # Add channel dimension
            window = np.expand_dims(window, axis=-1)
            X_test.append(window)
            y_test.append(emotion_to_index[row['emotion']])
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

def prepare_audio_batch(X, y, batch_size=32, time_steps=1):
    """
    Prepare audio batch for time distributed model
    
    Args:
        X: Input features
        y: Target labels
        batch_size: Batch size
        time_steps: Number of time steps
        
    Returns:
        X_batch, y_batch
    """
    # Reshape X to (batch_size, time_steps, mel_bins, time_frames, channels)
    X_batch = X.reshape(batch_size, time_steps, X.shape[1], X.shape[2], X.shape[3])
    
    # Convert to torch tensors
    X_batch = torch.tensor(X_batch, dtype=torch.float32)
    y_batch = torch.tensor(y, dtype=torch.long)
    
    return X_batch, y_batch
