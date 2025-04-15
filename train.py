"""
Main training script for multimodal emotion detection models

This script trains both audio and text emotion recognition models.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from src.audio.preprocessing import prepare_audio_data
from src.text.preprocessing import TextPreprocessor
from src.utils.trainer import ModelTrainer

def train_audio_model(args):
    """
    Train audio emotion recognition model
    
    Args:
        args: Command line arguments
    """
    print("\n=== Training Audio Emotion Recognition Model ===\n")
    
    # Prepare data
    X_train, y_train, X_test, y_test = prepare_audio_data(
        metadata_path=args.audio_metadata,
        max_samples_per_class=args.max_samples_per_class,
        window_size=args.window_size,
        time_step=args.time_step
    )
    
    # Create model trainer
    model_params = {
        'num_classes': len(np.unique(y_train)),
        'input_shape': (args.window_size, args.window_size, 1)
    }
    
    trainer = ModelTrainer('audio', model_params)
    trainer.build_model()
    
    # Create dataloaders
    train_loader, val_loader = trainer.create_dataloaders(
        X_train, y_train, X_test, y_test, batch_size=args.batch_size
    )
    
    # Train model
    history = trainer.train(
        train_loader, val_loader, 
        epochs=args.epochs, 
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )
    
    # Plot training history
    trainer.plot_history(save_path=os.path.join(args.save_dir, 'audio_training_history.png'))
    
    # Evaluate model
    test_loss, test_acc, predictions, targets = trainer.evaluate(val_loader)
    
    # Save evaluation results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'predictions': predictions.tolist(),
        'targets': targets.tolist()
    }
    
    pd.DataFrame(results).to_csv(os.path.join(args.save_dir, 'audio_evaluation_results.csv'), index=False)
    
    print(f"\nAudio model training completed. Model saved to {args.save_dir}")

def train_text_model(args):
    """
    Train text emotion recognition model
    
    Args:
        args: Command line arguments
    """
    print("\n=== Training Text Emotion Recognition Model ===\n")
    
    # Create text preprocessor
    preprocessor = TextPreprocessor(
        max_seq_length=args.max_seq_length,
        min_word_freq=args.min_word_freq
    )
    
    # Prepare data
    X_train, y_train, X_test, y_test, emotion_to_index = preprocessor.prepare_data(
        train_path=args.text_train_path,
        test_path=args.text_test_path
    )
    
    # Save emotion mapping
    emotion_mapping = {v: k for k, v in emotion_to_index.items()}
    pd.DataFrame(list(emotion_mapping.items()), columns=['index', 'emotion']).to_csv(
        os.path.join(args.save_dir, 'text_emotion_mapping.csv'), index=False
    )
    
    # Create model trainer
    model_params = {
        'vocab_size': preprocessor.vocab_size,
        'embedding_dim': args.embedding_dim,
        'num_classes': len(emotion_to_index),
        'max_seq_length': args.max_seq_length
    }
    
    trainer = ModelTrainer('text', model_params)
    trainer.build_model()
    
    # Create dataloaders
    train_loader, val_loader = trainer.create_dataloaders(
        X_train, y_train, X_test, y_test, batch_size=args.batch_size
    )
    
    # Train model
    history = trainer.train(
        train_loader, val_loader, 
        epochs=args.epochs, 
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )
    
    # Plot training history
    trainer.plot_history(save_path=os.path.join(args.save_dir, 'text_training_history.png'))
    
    # Evaluate model
    test_loss, test_acc, predictions, targets = trainer.evaluate(val_loader)
    
    # Save evaluation results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'predictions': predictions.tolist(),
        'targets': targets.tolist()
    }
    
    pd.DataFrame(results).to_csv(os.path.join(args.save_dir, 'text_evaluation_results.csv'), index=False)
    
    print(f"\nText model training completed. Model saved to {args.save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train multimodal emotion recognition models')
    
    # General parameters
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--mode', type=str, choices=['audio', 'text', 'both'], default='both', 
                        help='Which model(s) to train')
    
    # Audio model parameters
    parser.add_argument('--audio_metadata', type=str, default='data/audio/ravdess_metadata.csv',
                        help='Path to audio metadata CSV')
    parser.add_argument('--max_samples_per_class', type=int, default=None, 
                        help='Maximum number of samples per class for audio')
    parser.add_argument('--window_size', type=int, default=128, 
                        help='Window size for audio spectrograms')
    parser.add_argument('--time_step', type=int, default=64, 
                        help='Time step for audio spectrograms')
    
    # Text model parameters
    parser.add_argument('--text_train_path', type=str, default='data/text/train.csv',
                        help='Path to text training data CSV')
    parser.add_argument('--text_test_path', type=str, default='data/text/test.csv',
                        help='Path to text test data CSV')
    parser.add_argument('--max_seq_length', type=int, default=100, 
                        help='Maximum sequence length for text')
    parser.add_argument('--min_word_freq', type=int, default=2, 
                        help='Minimum word frequency for vocabulary')
    parser.add_argument('--embedding_dim', type=int, default=300, 
                        help='Word embedding dimension')
    
    args = parser.parse_args()
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(args.save_dir, f"run_{timestamp}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.save_dir, 'training_args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Train models
    if args.mode in ['audio', 'both']:
        train_audio_model(args)
    
    if args.mode in ['text', 'both']:
        train_text_model(args)
    
    print("\nTraining completed successfully!")

if __name__ == '__main__':
    main()
