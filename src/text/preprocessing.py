"""
Text preprocessing utilities for emotion recognition

This module provides functions for preprocessing text data for emotion recognition,
including tokenization, cleaning, and preparing data for model training.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import Counter
import re
import string

# Simple tokenizer without NLTK dependency
def simple_tokenize(text):
    """Simple tokenizer that splits on whitespace and punctuation"""
    # Convert to lowercase
    text = text.lower()
    # Replace punctuation with spaces
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    # Split on whitespace
    tokens = text.split()
    return tokens

# Simple list of English stopwords
STOPWORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
             "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
             'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
             'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
             'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
             'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
             'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
             'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
             'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
             'about', 'against', 'between', 'into', 'through', 'during', 'before', 
             'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
             'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
             'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
             'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
             'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
             'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 
             'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
             "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 
             'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', 
             "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
             'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', 
             "won't", 'wouldn', "wouldn't"}

class TextPreprocessor:
    """
    Text preprocessor for emotion recognition
    """
    def __init__(self, max_seq_length=100, min_word_freq=2):
        self.max_seq_length = max_seq_length
        self.min_word_freq = min_word_freq
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_counts = Counter()
        self.vocab_size = 0
        
    def clean_text(self, text):
        """
        Clean and standardize text
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = simple_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in STOPWORDS]
        
        return tokens
    
    def build_vocabulary(self, texts):
        """
        Build vocabulary from texts
        
        Args:
            texts: List of texts
            
        Returns:
            Vocabulary size
        """
        # Count word frequencies
        for text in texts:
            tokens = self.tokenize(text)
            self.word_counts.update(tokens)
        
        # Filter words by frequency
        filtered_words = [word for word, count in self.word_counts.items() if count >= self.min_word_freq]
        
        # Create word-to-index mapping
        self.word_to_index = {'<PAD>': 0, '<UNK>': 1}
        self.index_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        for i, word in enumerate(filtered_words, start=2):
            self.word_to_index[word] = i
            self.index_to_word[i] = word
        
        self.vocab_size = len(self.word_to_index)
        
        return self.vocab_size
    
    def text_to_sequence(self, text):
        """
        Convert text to sequence of indices
        
        Args:
            text: Input text
            
        Returns:
            Sequence of indices
        """
        tokens = self.tokenize(text)
        sequence = [self.word_to_index.get(token, 1) for token in tokens]  # 1 is <UNK>
        
        # Pad or truncate sequence
        if len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]
        else:
            sequence = sequence + [0] * (self.max_seq_length - len(sequence))  # 0 is <PAD>
        
        return sequence
    
    def prepare_data(self, train_path, test_path=None, test_split=0.2):
        """
        Prepare data for model training
        
        Args:
            train_path: Path to training data CSV
            test_path: Path to test data CSV (optional)
            test_split: Test split ratio if test_path is None
            
        Returns:
            X_train, y_train, X_test, y_test, emotion_to_index
        """
        # Load training data
        train_df = pd.read_csv(train_path)
        
        # Map emotions to indices
        emotions = sorted(train_df['emotion'].unique())
        emotion_to_index = {emotion: i for i, emotion in enumerate(emotions)}
        
        # Build vocabulary from training data
        print("Building vocabulary...")
        self.build_vocabulary(train_df['text'])
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Process training data
        X_train = []
        y_train = []
        
        print("Processing training data...")
        for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
            sequence = self.text_to_sequence(row['text'])
            X_train.append(sequence)
            y_train.append(emotion_to_index[row['emotion']])
        
        # Load or split test data
        if test_path is not None:
            test_df = pd.read_csv(test_path)
            
            # Process test data
            X_test = []
            y_test = []
            
            print("Processing test data...")
            for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
                sequence = self.text_to_sequence(row['text'])
                X_test.append(sequence)
                y_test.append(emotion_to_index[row['emotion']])
        else:
            # Split training data
            train_size = int((1 - test_split) * len(X_train))
            X_test = X_train[train_size:]
            y_test = y_train[train_size:]
            X_train = X_train[:train_size]
            y_train = y_train[:train_size]
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"Training data shape: {X_train.shape}, {y_train.shape}")
        print(f"Test data shape: {X_test.shape}, {y_test.shape}")
        
        return X_train, y_train, X_test, y_test, emotion_to_index
    
    def prepare_batch(self, X, y, batch_size=32):
        """
        Prepare batch for model training
        
        Args:
            X: Input sequences
            y: Target labels
            batch_size: Batch size
            
        Returns:
            X_batch, y_batch
        """
        # Convert to torch tensors
        X_batch = torch.tensor(X[:batch_size], dtype=torch.long)
        y_batch = torch.tensor(y[:batch_size], dtype=torch.long)
        
        return X_batch, y_batch
