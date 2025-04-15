"""
Simplified training script for resource-constrained environments

This script trains simplified versions of the emotion detection models
that can run in environments with limited computational resources.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from src.audio.preprocessing import prepare_audio_data
from src.text.preprocessing import TextPreprocessor
from src.utils.simplified_models import create_simplified_text_model, create_simplified_audio_model

class SimplifiedTrainer:
    """
    Simplified trainer class for emotion recognition models
    """
    def __init__(self, model_type, model, device=None):
        """
        Initialize trainer
        
        Args:
            model_type: 'audio' or 'text'
            model: Model instance
            device: Device to use for training (cpu or cuda)
        """
        self.model_type = model_type
        self.model = model
        self.device = device if device is not None else torch.device('cpu')
        self.model.to(self.device)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        print(f"Using device: {self.device}")
    
    def train_epoch(self, X_train, y_train, batch_size, criterion, optimizer):
        """
        Train for one epoch
        
        Args:
            X_train: Training features
            y_train: Training labels
            batch_size: Batch size
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Training loss and accuracy
        """
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Process data in batches
        num_samples = len(X_train)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            if self.model_type == 'audio':
                inputs = torch.tensor(X_train[batch_indices], dtype=torch.float32).to(self.device)
                # Reshape for CNN input (batch_size, channels, height, width)
                inputs = inputs.permute(0, 3, 1, 2)
            else:  # text
                inputs = torch.tensor(X_train[batch_indices], dtype=torch.long).to(self.device)
                
            targets = torch.tensor(y_train[batch_indices], dtype=torch.long).to(self.device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * len(batch_indices)
            _, predicted = torch.max(outputs, 1)
            train_total += len(batch_indices)
            train_correct += (predicted == targets).sum().item()
        
        # Calculate average training loss and accuracy
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        return train_loss, train_acc
    
    def validate(self, X_val, y_val, batch_size, criterion):
        """
        Validate model
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            batch_size: Batch size
            criterion: Loss function
            
        Returns:
            Validation loss and accuracy
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Process data in batches
        num_samples = len(X_val)
        
        with torch.no_grad():
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                
                # Get batch data
                if self.model_type == 'audio':
                    inputs = torch.tensor(X_val[start_idx:end_idx], dtype=torch.float32).to(self.device)
                    # Reshape for CNN input (batch_size, channels, height, width)
                    inputs = inputs.permute(0, 3, 1, 2)
                else:  # text
                    inputs = torch.tensor(X_val[start_idx:end_idx], dtype=torch.long).to(self.device)
                    
                targets = torch.tensor(y_val[start_idx:end_idx], dtype=torch.long).to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                val_loss += loss.item() * (end_idx - start_idx)
                _, predicted = torch.max(outputs, 1)
                val_total += (end_idx - start_idx)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate average validation loss and accuracy
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        return val_loss, val_acc
    
    def train(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=8, learning_rate=0.001, save_dir='models'):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            save_dir: Directory to save model checkpoints
            
        Returns:
            Training history
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize best validation loss
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self.train_epoch(X_train, y_train, batch_size, criterion, optimizer)
            
            # Validation phase
            val_loss, val_acc = self.validate(X_val, y_val, batch_size, criterion)
            
            # Save model if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f"{save_dir}/simplified_{self.model_type}_model_best.pth")
                print(f"Model saved with validation loss: {val_loss:.4f}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch results
            print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
        
        # Save final model
        torch.save(self.model.state_dict(), f"{save_dir}/simplified_{self.model_type}_model_final.pth")
        
        # Save training history
        history_df = pd.DataFrame({
            'epoch': range(1, epochs + 1),
            'train_loss': self.history['train_loss'],
            'train_acc': self.history['train_acc'],
            'val_loss': self.history['val_loss'],
            'val_acc': self.history['val_acc']
        })
        history_df.to_csv(f"{save_dir}/simplified_{self.model_type}_training_history.csv", index=False)
        
        return self.history

def train_simplified_text_model(args):
    """
    Train simplified text emotion recognition model
    
    Args:
        args: Command line arguments
    """
    print("\n=== Training Simplified Text Emotion Recognition Model ===\n")
    
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
    
    # Create simplified model
    model = create_simplified_text_model(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(emotion_to_index),
        max_seq_length=args.max_seq_length
    )
    
    # Create trainer
    trainer = SimplifiedTrainer('text', model)
    
    # Train model
    history = trainer.train(
        X_train, y_train, X_test, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )
    
    print(f"\nText model training completed. Model saved to {args.save_dir}")

def train_simplified_audio_model(args):
    """
    Train simplified audio emotion recognition model
    
    Args:
        args: Command line arguments
    """
    print("\n=== Training Simplified Audio Emotion Recognition Model ===\n")
    
    # Prepare data
    X_train, y_train, X_test, y_test = prepare_audio_data(
        metadata_path=args.audio_metadata,
        max_samples_per_class=args.max_samples_per_class,
        window_size=args.window_size,
        time_step=args.time_step
    )
    
    # Create simplified model
    model = create_simplified_audio_model(
        num_classes=len(np.unique(y_train)),
        input_shape=(args.window_size, args.window_size, 1)
    )
    
    # Create trainer
    trainer = SimplifiedTrainer('audio', model)
    
    # Train model
    history = trainer.train(
        X_train, y_train, X_test, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )
    
    print(f"\nAudio model training completed. Model saved to {args.save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train simplified multimodal emotion recognition models')
    
    # General parameters
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--mode', type=str, choices=['audio', 'text', 'both'], default='both', 
                        help='Which model(s) to train')
    
    # Audio model parameters
    parser.add_argument('--audio_metadata', type=str, default='data/audio/ravdess_metadata.csv',
                        help='Path to audio metadata CSV')
    parser.add_argument('--max_samples_per_class', type=int, default=50, 
                        help='Maximum number of samples per class for audio')
    parser.add_argument('--window_size', type=int, default=64, 
                        help='Window size for audio spectrograms')
    parser.add_argument('--time_step', type=int, default=32, 
                        help='Time step for audio spectrograms')
    
    # Text model parameters
    parser.add_argument('--text_train_path', type=str, default='data/text/train.csv',
                        help='Path to text training data CSV')
    parser.add_argument('--text_test_path', type=str, default='data/text/test.csv',
                        help='Path to text test data CSV')
    parser.add_argument('--max_seq_length', type=int, default=50, 
                        help='Maximum sequence length for text')
    parser.add_argument('--min_word_freq', type=int, default=1, 
                        help='Minimum word frequency for vocabulary')
    parser.add_argument('--embedding_dim', type=int, default=100, 
                        help='Word embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, 
                        help='LSTM hidden dimension')
    
    args = parser.parse_args()
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(args.save_dir, f"simplified_run_{timestamp}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.save_dir, 'training_args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Train models
    if args.mode in ['text', 'both']:
        train_simplified_text_model(args)
    
    if args.mode in ['audio', 'both']:
        train_simplified_audio_model(args)
    
    print("\nTraining completed successfully!")

if __name__ == '__main__':
    main()
