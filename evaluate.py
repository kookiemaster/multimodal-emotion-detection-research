"""
Evaluation script for simplified multimodal emotion detection models

This script evaluates the performance of the trained simplified models
and generates visualizations of the results.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from src.audio.preprocessing import prepare_audio_data
from src.text.preprocessing import TextPreprocessor
from src.utils.simplified_models import create_simplified_text_model, create_simplified_audio_model

def evaluate_model(model, X_test, y_test, batch_size=4, device='cpu', model_type='text'):
    """
    Evaluate model on test data
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        batch_size: Batch size
        device: Device to use for evaluation
        model_type: 'audio' or 'text'
        
    Returns:
        Test loss, accuracy, predictions, and true labels
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []
    
    # Process data in batches
    num_samples = len(X_test)
    
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            
            # Get batch data
            if model_type == 'audio':
                inputs = torch.tensor(X_test[start_idx:end_idx], dtype=torch.float32).to(device)
                # Reshape for CNN input (batch_size, channels, height, width)
                inputs = inputs.permute(0, 3, 1, 2)
            else:  # text
                inputs = torch.tensor(X_test[start_idx:end_idx], dtype=torch.long).to(device)
                
            targets = torch.tensor(y_test[start_idx:end_idx], dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            test_loss += loss.item() * (end_idx - start_idx)
            _, predicted = torch.max(outputs, 1)
            test_total += (end_idx - start_idx)
            test_correct += (predicted == targets).sum().item()
            
            # Save predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate average test loss and accuracy
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    
    return test_loss, test_acc, np.array(all_predictions), np.array(all_targets)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()

def evaluate_text_model(args):
    """
    Evaluate text emotion recognition model
    
    Args:
        args: Command line arguments
    """
    print("\n=== Evaluating Text Emotion Recognition Model ===\n")
    
    # Load emotion mapping
    emotion_mapping_path = os.path.join(args.model_dir, 'text_emotion_mapping.csv')
    if os.path.exists(emotion_mapping_path):
        emotion_mapping = pd.read_csv(emotion_mapping_path)
        emotion_mapping_dict = dict(zip(emotion_mapping['index'], emotion_mapping['emotion']))
        class_names = [emotion_mapping_dict[i] for i in range(len(emotion_mapping_dict))]
    else:
        print("Warning: Emotion mapping file not found. Using generic class names.")
        class_names = [f"Class {i}" for i in range(7)]  # Default to 7 classes
    
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
    
    # Create model
    model = create_simplified_text_model(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(emotion_to_index),
        max_seq_length=args.max_seq_length
    )
    
    # Load model weights
    model_path = os.path.join(args.model_dir, 'simplified_text_model_best.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(args.model_dir, 'simplified_text_model_final.pth')
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Evaluate model
    test_loss, test_acc, predictions, targets = evaluate_model(
        model, X_test, y_test, batch_size=args.batch_size, device='cpu', model_type='text'
    )
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Generate classification report
    report = classification_report(targets, predictions, target_names=class_names, digits=3)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(args.output_dir, 'text_classification_report.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        targets, predictions, class_names,
        save_path=os.path.join(args.output_dir, 'text_confusion_matrix.png')
    )
    
    # Save evaluation results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'predictions': predictions.tolist(),
        'targets': targets.tolist()
    }
    
    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, 'text_evaluation_results.csv'), index=False)
    
    print(f"\nText model evaluation completed. Results saved to {args.output_dir}")

def evaluate_audio_model(args):
    """
    Evaluate audio emotion recognition model
    
    Args:
        args: Command line arguments
    """
    print("\n=== Evaluating Audio Emotion Recognition Model ===\n")
    
    # Prepare data
    X_train, y_train, X_test, y_test = prepare_audio_data(
        metadata_path=args.audio_metadata,
        max_samples_per_class=args.max_samples_per_class,
        window_size=args.window_size,
        time_step=args.time_step
    )
    
    # Get class names
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
    
    class_names = [emotion_map.get(i, f"Class {i}") for i in range(len(np.unique(y_train)))]
    
    # Create model
    model = create_simplified_audio_model(
        num_classes=len(np.unique(y_train)),
        input_shape=(args.window_size, args.window_size, 1)
    )
    
    # Load model weights
    model_path = os.path.join(args.model_dir, 'simplified_audio_model_best.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(args.model_dir, 'simplified_audio_model_final.pth')
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Evaluate model
    test_loss, test_acc, predictions, targets = evaluate_model(
        model, X_test, y_test, batch_size=args.batch_size, device='cpu', model_type='audio'
    )
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Generate classification report
    report = classification_report(targets, predictions, target_names=class_names, digits=3)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(args.output_dir, 'audio_classification_report.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        targets, predictions, class_names,
        save_path=os.path.join(args.output_dir, 'audio_confusion_matrix.png')
    )
    
    # Save evaluation results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'predictions': predictions.tolist(),
        'targets': targets.tolist()
    }
    
    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, 'audio_evaluation_results.csv'), index=False)
    
    print(f"\nAudio model evaluation completed. Results saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate simplified multimodal emotion recognition models')
    
    # General parameters
    parser.add_argument('--model_dir', type=str, required=True, 
                        help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--mode', type=str, choices=['audio', 'text', 'both'], default='both', 
                        help='Which model(s) to evaluate')
    
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate models
    if args.mode in ['text', 'both']:
        evaluate_text_model(args)
    
    if args.mode in ['audio', 'both']:
        evaluate_audio_model(args)
    
    print("\nEvaluation completed successfully!")

if __name__ == '__main__':
    main()
