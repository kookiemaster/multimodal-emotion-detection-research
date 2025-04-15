"""
Simplified model implementations for resource-constrained environments

This module provides lightweight versions of the emotion detection models
that can run in environments with limited computational resources.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedTextModel(nn.Module):
    """
    Simplified CNN-LSTM model for text emotion recognition
    """
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64, num_classes=7, max_seq_length=100):
        super(SimplifiedTextModel, self).__init__()
        
        # Embedding layer with reduced dimensions
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Single CNN block instead of three
        self.conv = nn.Conv1d(embedding_dim, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate the size after convolution
        self.lstm_input_size = 64 * (max_seq_length // 2)
        
        # Single LSTM layer instead of three
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
        """
        # Embedding
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # Transpose for 1D convolution (batch_size, embedding_dim, seq_length)
        x = x.transpose(1, 2)
        
        # Apply CNN block
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Transpose back for LSTM (batch_size, seq_length, features)
        x = x.transpose(1, 2)
        
        # Apply LSTM layer
        x, _ = self.lstm(x)
        
        # Take only the last output
        x = x[:, -1, :]
        
        # Apply fully connected layer
        x = self.fc(x)
        
        # Apply softmax
        x = F.softmax(x, dim=-1)
        
        return x

class SimplifiedAudioModel(nn.Module):
    """
    Simplified CNN model for audio emotion recognition
    """
    def __init__(self, num_classes=8, input_shape=(128, 128, 1)):
        super(SimplifiedAudioModel, self).__init__()
        
        # Input shape: (batch_size, channels, height, width)
        self.input_shape = input_shape
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.3)
        
        # Calculate the flattened size after convolutions
        self._calculate_flatten_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def _calculate_flatten_size(self):
        """Calculate the size of the flattened features after convolutions"""
        # Assuming input shape is (128, 128, 1)
        h, w = self.input_shape[0], self.input_shape[1]
        
        # Apply 2 pooling layers with pool_size=2
        h = h // 4
        w = w // 4
            
        # Calculate flattened size
        self.flatten_size = 32 * h * w
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        """
        # Apply convolutional layers
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        
        # Apply softmax
        x = F.softmax(x, dim=-1)
        
        return x

def create_simplified_text_model(vocab_size, embedding_dim=100, hidden_dim=64, num_classes=7, max_seq_length=100):
    """
    Create and initialize the simplified text emotion recognition model
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of LSTM hidden state
        num_classes: Number of emotion classes
        max_seq_length: Maximum sequence length
        
    Returns:
        Initialized SimplifiedTextModel
    """
    model = SimplifiedTextModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        max_seq_length=max_seq_length
    )
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    return model

def create_simplified_audio_model(num_classes=8, input_shape=(128, 128, 1)):
    """
    Create and initialize the simplified audio emotion recognition model
    
    Args:
        num_classes: Number of emotion classes
        input_shape: Shape of input spectrograms (mel_bins, time_frames, channels)
        
    Returns:
        Initialized SimplifiedAudioModel
    """
    model = SimplifiedAudioModel(num_classes=num_classes, input_shape=input_shape)
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    return model
