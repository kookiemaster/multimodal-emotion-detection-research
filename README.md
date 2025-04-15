# Multimodal Emotion Detection Research

This repository contains an implementation of multimodal emotion detection using voice and text modalities. The project explores state-of-the-art approaches for emotion recognition and implements a simplified version based on CNN-LSTM architectures.

## Project Overview

Emotion detection is a challenging task in artificial intelligence that aims to recognize human emotions from various input modalities. This project focuses on:

1. Researching state-of-the-art multimodal emotion detection methods
2. Implementing a simplified version of the CNN-LSTM architecture for both audio and text
3. Training and evaluating the models on emotion recognition datasets
4. Providing tools for using the trained models on new inputs

## Repository Structure

```
multimodal-emotion-detection-research/
├── data/                       # Data directory
│   ├── audio/                  # Audio data and metadata
│   └── text/                   # Text data for emotion recognition
├── docs/                       # Documentation
│   └── implementation_details.md  # Detailed implementation documentation
├── models/                     # Trained model checkpoints
├── src/                        # Source code
│   ├── audio/                  # Audio processing and models
│   │   ├── model.py            # Audio emotion model implementation
│   │   └── preprocessing.py    # Audio preprocessing utilities
│   ├── text/                   # Text processing and models
│   │   ├── model.py            # Text emotion model implementation
│   │   └── preprocessing.py    # Text preprocessing utilities
│   └── utils/                  # Utility functions
│       ├── simplified_models.py # Resource-efficient model implementations
│       ├── trainer.py          # Model training utilities
│       └── download_datasets.py # Dataset download and preprocessing
├── demo.py                     # Usage examples and demonstration
├── evaluate.py                 # Model evaluation script
├── train.py                    # Original training script
├── train_simplified.py         # Resource-efficient training script
└── README.md                   # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kookiemaster/multimodal-emotion-detection-research.git
cd multimodal-emotion-detection-research
```

2. Install dependencies:
```bash
pip install torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.24.3 librosa scikit-learn matplotlib seaborn pandas tqdm
```

## Data Preparation

The project uses the following datasets:

1. **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) for audio emotion recognition
2. A sample text dataset for text emotion recognition

To prepare the datasets:

```bash
python src/utils/download_datasets.py
```

This script will:
- Download and extract the RAVDESS dataset
- Create metadata for audio files
- Generate a sample text dataset for development

## Model Architecture

### Audio Emotion Recognition

The audio emotion recognition model uses a simplified CNN architecture:
- 2 convolutional layers with max pooling and dropout
- Fully connected layers for classification

### Text Emotion Recognition

The text emotion recognition model uses a simplified CNN-LSTM architecture:
- Word embedding layer
- 1D convolutional layer with max pooling
- Single LSTM layer
- Fully connected layer for classification

## Training

To train the simplified models:

```bash
python train_simplified.py --mode both --epochs 5 --batch_size 8
```

Options:
- `--mode`: Choose from `audio`, `text`, or `both`
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--save_dir`: Directory to save models

## Evaluation

To evaluate the trained models:

```bash
python evaluate.py --model_dir models/simplified_run_YYYYMMDD_HHMMSS --mode both
```

Options:
- `--model_dir`: Directory containing trained models
- `--mode`: Choose from `audio`, `text`, or `both`
- `--output_dir`: Directory to save evaluation results

## Usage Examples

The `demo.py` script provides examples of how to use the trained models for emotion detection:

```bash
python demo.py --model_dir models/simplified_run_YYYYMMDD_HHMMSS --text "I'm feeling really happy today!"
```

For audio emotion detection:

```bash
python demo.py --model_dir models/simplified_run_YYYYMMDD_HHMMSS --mode audio --audio_file path/to/audio.wav
```

## Implementation Details

For detailed information about the implementation, challenges faced, and results obtained, see the [Implementation Details](docs/implementation_details.md) document.

## Results

The simplified text model achieved a test accuracy of 14.29% on the sample dataset. This relatively low performance is expected given the significant simplifications made to accommodate resource constraints, limited training data, and reduced training time.

## Future Improvements

Given more computational resources, several improvements could be made:
1. Use the original, more complex model architectures
2. Train on larger and more diverse datasets
3. Implement more sophisticated preprocessing techniques
4. Increase training time (more epochs)
5. Implement true multimodal fusion (combining audio and text features)
6. Experiment with pre-trained models like BERT for text and VGGish for audio

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The implementation is based on Maelfabien's Multimodal Emotion Recognition approach
- RAVDESS dataset for audio emotion samples
