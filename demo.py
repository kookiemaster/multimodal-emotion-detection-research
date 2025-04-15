"""
Usage examples for multimodal emotion detection models

This script demonstrates how to use the trained models for emotion detection
on new audio and text inputs.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display

from src.audio.preprocessing import extract_melspectrogram, create_time_windows
from src.text.preprocessing import TextPreprocessor
from src.utils.simplified_models import create_simplified_text_model, create_simplified_audio_model

def predict_emotion_from_text(text, model_path, preprocessor, device='cpu'):
    """
    Predict emotion from text input
    
    Args:
        text: Input text
        model_path: Path to trained model
        preprocessor: Text preprocessor
        device: Device to use for inference
        
    Returns:
        Predicted emotion and probabilities
    """
    # Tokenize and convert to sequence
    sequence = preprocessor.text_to_sequence(text)
    
    # Convert to tensor
    inputs = torch.tensor([sequence], dtype=torch.long).to(device)
    
    # Load model
    model = create_simplified_text_model(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=100,
        hidden_dim=64,
        num_classes=7,
        max_seq_length=preprocessor.max_seq_length
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = outputs[0].cpu().numpy()
        predicted_class = np.argmax(probabilities)
    
    return predicted_class, probabilities

def predict_emotion_from_audio(audio_path, model_path, emotion_map, device='cpu', window_size=64):
    """
    Predict emotion from audio file
    
    Args:
        audio_path: Path to audio file
        model_path: Path to trained model
        emotion_map: Dictionary mapping class indices to emotion labels
        device: Device to use for inference
        window_size: Window size for spectrograms
        
    Returns:
        Predicted emotion and probabilities
    """
    # Load audio file
    signal, sr = librosa.load(audio_path, sr=22050)
    
    # Extract mel-spectrogram
    mel_spec = extract_melspectrogram(signal, sr=sr, n_mels=window_size)
    
    # Create time windows
    windows = create_time_windows(mel_spec, window_size=window_size, time_step=window_size//2)
    
    # Add channel dimension
    windows = [np.expand_dims(window, axis=-1) for window in windows]
    
    # Convert to tensor
    inputs = torch.tensor(windows, dtype=torch.float32).to(device)
    
    # Reshape for CNN input (batch_size, channels, height, width)
    inputs = inputs.permute(0, 3, 1, 2)
    
    # Load model
    model = create_simplified_audio_model(
        num_classes=len(emotion_map),
        input_shape=(window_size, window_size, 1)
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(inputs)
        # Average predictions across windows
        avg_output = torch.mean(outputs, dim=0)
        probabilities = avg_output.cpu().numpy()
        predicted_class = np.argmax(probabilities)
    
    return predicted_class, probabilities

def visualize_audio_prediction(audio_path, predicted_class, probabilities, emotion_map, save_path=None):
    """
    Visualize audio prediction with waveform and mel-spectrogram
    
    Args:
        audio_path: Path to audio file
        predicted_class: Predicted emotion class
        probabilities: Prediction probabilities
        emotion_map: Dictionary mapping class indices to emotion labels
        save_path: Path to save visualization
    """
    # Load audio file
    signal, sr = librosa.load(audio_path, sr=22050)
    
    # Extract mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(signal, sr=sr)
    plt.title('Waveform')
    
    # Plot mel-spectrogram
    plt.subplot(3, 1, 2)
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    
    # Plot prediction probabilities
    plt.subplot(3, 1, 3)
    emotions = [emotion_map.get(i, f"Class {i}") for i in range(len(probabilities))]
    plt.bar(emotions, probabilities)
    plt.title(f'Prediction: {emotion_map.get(predicted_class, f"Class {predicted_class}")}')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.close()

def visualize_text_prediction(text, predicted_class, probabilities, emotion_map, save_path=None):
    """
    Visualize text prediction
    
    Args:
        text: Input text
        predicted_class: Predicted emotion class
        probabilities: Prediction probabilities
        emotion_map: Dictionary mapping class indices to emotion labels
        save_path: Path to save visualization
    """
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot prediction probabilities
    emotions = [emotion_map.get(i, f"Class {i}") for i in range(len(probabilities))]
    plt.bar(emotions, probabilities)
    plt.title(f'Prediction: {emotion_map.get(predicted_class, f"Class {predicted_class}")}')
    plt.xlabel('Emotion')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    
    # Add text as annotation
    plt.figtext(0.5, 0.01, f'Text: "{text}"', wrap=True, horizontalalignment='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Multimodal Emotion Detection Demo')
    
    # General parameters
    parser.add_argument('--model_dir', type=str, required=True, 
                        help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='demo_results',
                        help='Directory to save demo results')
    parser.add_argument('--mode', type=str, choices=['audio', 'text', 'both'], default='both', 
                        help='Which model(s) to demonstrate')
    
    # Text parameters
    parser.add_argument('--text', type=str, default="I'm feeling really happy today, everything is going well!",
                        help='Text input for emotion detection')
    
    # Audio parameters
    parser.add_argument('--audio_file', type=str, default=None,
                        help='Path to audio file for emotion detection')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Text emotion detection
    if args.mode in ['text', 'both']:
        print("\n=== Text Emotion Detection ===\n")
        
        # Load emotion mapping
        emotion_mapping_path = os.path.join(args.model_dir, 'text_emotion_mapping.csv')
        if os.path.exists(emotion_mapping_path):
            import pandas as pd
            emotion_mapping = pd.read_csv(emotion_mapping_path)
            emotion_map = dict(zip(emotion_mapping['index'], emotion_mapping['emotion']))
        else:
            print("Warning: Emotion mapping file not found. Using generic class names.")
            emotion_map = {0: 'angry', 1: 'disgust', 2: 'fearful', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprised'}
        
        # Create text preprocessor
        preprocessor = TextPreprocessor(max_seq_length=50, min_word_freq=1)
        
        # Build vocabulary from sample data
        sample_texts = [
            "I'm feeling really happy today, everything is going well!",
            "I'm so sad and disappointed about what happened.",
            "That makes me so angry, I can't believe they did that!",
            "I'm really scared about the upcoming presentation.",
            "It's just a normal day, nothing special happening.",
            "The surprise party they threw for me was amazing!",
            "That smell is disgusting, I can't stand it."
        ]
        
        preprocessor.build_vocabulary(sample_texts)
        
        # Load model
        model_path = os.path.join(args.model_dir, 'simplified_text_model_best.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(args.model_dir, 'simplified_text_model_final.pth')
        
        # Make prediction
        predicted_class, probabilities = predict_emotion_from_text(
            args.text, model_path, preprocessor
        )
        
        # Print results
        print(f"Text: \"{args.text}\"")
        print(f"Predicted emotion: {emotion_map.get(predicted_class, f'Class {predicted_class}')}")
        print(f"Probabilities: {probabilities}")
        
        # Visualize prediction
        visualize_text_prediction(
            args.text, predicted_class, probabilities, emotion_map,
            save_path=os.path.join(args.output_dir, 'text_prediction.png')
        )
    
    # Audio emotion detection
    if args.mode in ['audio', 'both'] and args.audio_file:
        print("\n=== Audio Emotion Detection ===\n")
        
        # Define emotion mapping
        emotion_map = {
            0: 'neutral',
            1: 'calm',
            2: 'happy',
            3: 'sad',
            4: 'angry',
            5: 'fearful',
            6: 'disgust',
            7: 'surprised'
        }
        
        # Load model
        model_path = os.path.join(args.model_dir, 'simplified_audio_model_best.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(args.model_dir, 'simplified_audio_model_final.pth')
        
        # Make prediction
        predicted_class, probabilities = predict_emotion_from_audio(
            args.audio_file, model_path, emotion_map
        )
        
        # Print results
        print(f"Audio file: {args.audio_file}")
        print(f"Predicted emotion: {emotion_map.get(predicted_class, f'Class {predicted_class}')}")
        print(f"Probabilities: {probabilities}")
        
        # Visualize prediction
        visualize_audio_prediction(
            args.audio_file, predicted_class, probabilities, emotion_map,
            save_path=os.path.join(args.output_dir, 'audio_prediction.png')
        )
    
    print("\nDemo completed successfully!")

if __name__ == '__main__':
    main()
