"""
Audio Emotion Recognition Model

This module implements a Time Distributed CNN-LSTM architecture for audio emotion recognition
based on Maelfabien's approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalFeatureLearningBlock(nn.Module):
    """
    Local Feature Learning Block (LFLB) for audio feature extraction
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2, dropout_rate=0.3):
        super(LocalFeatureLearningBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        x = self.pool(x)
        x = self.dropout(x)
        return x

class AudioEmotionModel(nn.Module):
    """
    Time Distributed CNN-LSTM model for audio emotion recognition
    """
    def __init__(self, num_classes=8, input_shape=(128, 128, 1)):
        super(AudioEmotionModel, self).__init__()
        
        # Input shape: (batch_size, time_steps, mel_bins, time_frames, channels)
        self.input_shape = input_shape
        
        # Local Feature Learning Blocks
        self.lflb1 = LocalFeatureLearningBlock(1, 64)
        self.lflb2 = LocalFeatureLearningBlock(64, 128)
        self.lflb3 = LocalFeatureLearningBlock(128, 256)
        self.lflb4 = LocalFeatureLearningBlock(256, 512)
        
        # Calculate the flattened size after convolutions
        self._calculate_flatten_size()
        
        # LSTM layers
        self.lstm1 = nn.LSTM(self.flatten_size, 256, batch_first=True, return_sequences=True)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def _calculate_flatten_size(self):
        """Calculate the size of the flattened features after convolutions"""
        # Assuming input shape is (128, 128, 1)
        h, w = self.input_shape[0], self.input_shape[1]
        
        # Apply 4 LFLBs with pool_size=2
        for _ in range(4):
            h = h // 2
            w = w // 2
            
        # Calculate flattened size
        self.flatten_size = 512 * h * w
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, mel_bins, time_frames, channels)
        """
        batch_size, time_steps, mel_bins, time_frames, channels = x.shape
        
        # Reshape for time distributed processing
        x = x.view(batch_size * time_steps, channels, mel_bins, time_frames)
        
        # Apply LFLBs
        x = self.lflb1(x)
        x = self.lflb2(x)
        x = self.lflb3(x)
        x = self.lflb4(x)
        
        # Flatten
        x = x.view(batch_size * time_steps, -1)
        
        # Reshape back to sequence
        x = x.view(batch_size, time_steps, -1)
        
        # Apply LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Apply softmax
        x = F.softmax(x, dim=-1)
        
        return x

def create_audio_model(num_classes=8, input_shape=(128, 128, 1)):
    """
    Create and initialize the audio emotion recognition model
    
    Args:
        num_classes: Number of emotion classes
        input_shape: Shape of input spectrograms (mel_bins, time_frames, channels)
        
    Returns:
        Initialized AudioEmotionModel
    """
    model = AudioEmotionModel(num_classes=num_classes, input_shape=input_shape)
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    return model
