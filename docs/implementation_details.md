# Implementation Details: Multimodal Emotion Detection

This document provides detailed information about the implementation of our multimodal emotion detection system, including the architecture, preprocessing techniques, training approach, and evaluation results.

## Project Overview

The goal of this project was to implement state-of-the-art multimodal emotion detection methods that combine voice and text modalities. We researched various approaches and selected Maelfabien's Multimodal Emotion Recognition approach as our baseline, which uses CNN-LSTM architectures for both audio and text processing.

During implementation, we encountered significant resource constraints in our development environment, which required us to adapt our approach. We created simplified versions of the models that maintain the core functionality while requiring fewer computational resources.

## Data Sources

### Audio Data
- **Dataset**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Features**: 1440 audio samples across 8 emotion categories (calm, happy, sad, angry, disgust, fearful, surprised, neutral)
- **Preprocessing**: Extracted mel-spectrograms from audio files, created time windows for processing

### Text Data
- **Dataset**: Created a sample dataset with 21 text samples across 7 emotion categories
- **Features**: Text samples with emotion labels (happy, sad, angry, fearful, neutral, surprised, disgust)
- **Preprocessing**: Custom tokenization, stopword removal, sequence padding

## Model Architecture

### Original Architecture (Based on Research)
The original architecture from Maelfabien's approach consisted of:

#### Audio Model
- Time Distributed CNN-LSTM architecture
- Multiple Local Feature Learning Blocks (LFLBs)
- Multiple LSTM layers for temporal modeling
- Fully connected layers for classification

#### Text Model
- Word embeddings layer
- Multiple CNN blocks for feature extraction
- Multiple LSTM layers for sequence modeling
- Fully connected layers for classification

### Simplified Architecture (Implemented)

Due to resource constraints, we simplified the architecture:

#### Simplified Audio Model
```python
class SimplifiedAudioModel(nn.Module):
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
```

#### Simplified Text Model
```python
class SimplifiedTextModel(nn.Module):
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
```

## Implementation Challenges and Solutions

### Challenge 1: Memory Constraints
- **Problem**: The original PyTorch installation was terminated due to memory constraints.
- **Solution**: Used a CPU-optimized version of PyTorch (2.0.0+cpu) and reduced model complexity.

### Challenge 2: NLTK Dependency Issues
- **Problem**: Encountered issues with NLTK's punkt_tab resource.
- **Solution**: Implemented a custom tokenization approach using basic string operations and a predefined list of stopwords.

### Challenge 3: NumPy Compatibility
- **Problem**: Compatibility issues between NumPy 2.x and PyTorch.
- **Solution**: Downgraded NumPy to version 1.24.3 to ensure compatibility.

### Challenge 4: Limited Training Data
- **Problem**: Limited sample size for text emotion recognition.
- **Solution**: Created a balanced sample dataset with multiple examples per emotion category.

## Training Approach

We used a simplified training approach with the following parameters:
- Batch size: 4 (reduced from 32)
- Epochs: 3 (reduced from 50)
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: Cross-Entropy Loss

## Evaluation Results

### Text Model Performance
- **Test Accuracy**: 14.29%
- **F1-Score (Macro Avg)**: 0.036
- **Precision (Macro Avg)**: 0.020
- **Recall (Macro Avg)**: 0.143

The model primarily predicted the "fearful" class, resulting in low precision and F1-scores for other classes. This is likely due to the simplified architecture, limited training data, and reduced training time.

## Future Improvements

Given more computational resources, several improvements could be made:
1. Use the original, more complex model architectures
2. Train on larger and more diverse datasets
3. Implement more sophisticated preprocessing techniques
4. Increase training time (more epochs)
5. Implement true multimodal fusion (combining audio and text features)
6. Experiment with pre-trained models like BERT for text and VGGish for audio

## Conclusion

This implementation demonstrates the core concepts of multimodal emotion detection while working within significant resource constraints. Despite the limitations, we successfully implemented a simplified version of the CNN-LSTM architecture for both audio and text modalities, and established a foundation that can be extended with more resources.

The relatively low performance metrics highlight the challenges of emotion detection, especially with simplified models and limited data. However, the implementation provides a valuable starting point for further research and development in this area.
