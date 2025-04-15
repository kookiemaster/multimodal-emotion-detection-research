"""
Text Emotion Recognition Model

This module implements a CNN-LSTM architecture for text emotion recognition
based on Maelfabien's approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEmotionModel(nn.Module):
    """
    CNN-LSTM model for text emotion recognition
    """
    def __init__(self, vocab_size, embedding_dim=300, num_classes=7, max_seq_length=100):
        super(TextEmotionModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # CNN blocks
        # Block 1
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=8, padding=3)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(128)
        
        # Block 2
        self.conv2 = nn.Conv1d(128, 256, kernel_size=8, padding=3)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.3)
        self.bn2 = nn.BatchNorm1d(256)
        
        # Block 3
        self.conv3 = nn.Conv1d(256, 512, kernel_size=8, padding=3)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout3 = nn.Dropout(0.3)
        self.bn3 = nn.BatchNorm1d(512)
        
        # Calculate the size after convolutions
        self.lstm_input_size = self._calculate_lstm_input_size(max_seq_length)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(self.lstm_input_size, 180, batch_first=True, return_sequences=True)
        self.lstm2 = nn.LSTM(180, 180, batch_first=True, return_sequences=True)
        self.lstm3 = nn.LSTM(180, 180, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(180, 128)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def _calculate_lstm_input_size(self, seq_length):
        """Calculate the size of the features after convolutions"""
        # After each pooling layer, the sequence length is halved
        for _ in range(3):  # 3 pooling layers
            seq_length = seq_length // 2
            
        # The feature size is 512 (from the last conv layer)
        return 512
        
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
        
        # Apply CNN blocks
        # Block 1
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Transpose back for LSTM (batch_size, seq_length, features)
        x = x.transpose(1, 2)
        
        # Apply LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        # Apply softmax
        x = F.softmax(x, dim=-1)
        
        return x

def create_text_model(vocab_size, embedding_dim=300, num_classes=7, max_seq_length=100):
    """
    Create and initialize the text emotion recognition model
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        num_classes: Number of emotion classes
        max_seq_length: Maximum sequence length
        
    Returns:
        Initialized TextEmotionModel
    """
    model = TextEmotionModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        max_seq_length=max_seq_length
    )
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    return model
