# Dataset Download and Preprocessing Script

import os
import requests
import zipfile
import io
import pandas as pd
import numpy as np
from tqdm import tqdm

# Create directories if they don't exist
os.makedirs('data/audio', exist_ok=True)
os.makedirs('data/text', exist_ok=True)

# Function to download and extract RAVDESS dataset
def download_ravdess():
    print("Downloading RAVDESS dataset...")
    
    # RAVDESS dataset URL
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    
    # Download the file
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Create a progress bar
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
    
    # Download and extract the dataset
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall("data/audio/ravdess")
    
    print("RAVDESS dataset downloaded and extracted successfully!")
    
    # Create a metadata file for the RAVDESS dataset
    create_ravdess_metadata()

# Function to create metadata for RAVDESS dataset
def create_ravdess_metadata():
    print("Creating metadata for RAVDESS dataset...")
    
    # Get all audio files
    audio_files = []
    for root, dirs, files in os.walk("data/audio/ravdess"):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    
    # Create a dataframe to store metadata
    metadata = []
    
    # RAVDESS filename format: modality-vocal_channel-emotion-emotional_intensity-statement-repetition-actor.wav
    # Emotions: 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    for file_path in audio_files:
        file_name = os.path.basename(file_path)
        parts = file_name.split('-')
        
        if len(parts) == 7:
            emotion_code = parts[2]
            emotion = emotion_map.get(emotion_code, 'unknown')
            intensity = 'normal' if parts[3] == '01' else 'strong'
            actor_id = parts[6].split('.')[0]
            gender = 'female' if int(actor_id) % 2 == 0 else 'male'
            
            metadata.append({
                'file_path': file_path,
                'emotion': emotion,
                'intensity': intensity,
                'actor_id': actor_id,
                'gender': gender
            })
    
    # Create a dataframe and save it
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv("data/audio/ravdess_metadata.csv", index=False)
    
    print(f"Metadata created with {len(metadata_df)} entries!")
    print(f"Emotion distribution:\n{metadata_df['emotion'].value_counts()}")

# Function to download Stream-of-consciousness dataset
def download_soc_dataset():
    print("Note: The Stream-of-consciousness dataset requires manual download due to licensing restrictions.")
    print("Creating a sample text dataset for development purposes...")
    
    # Create a sample dataset for development
    emotions = ['happy', 'sad', 'angry', 'fearful', 'neutral', 'surprised', 'disgust']
    
    # Sample texts for each emotion
    sample_texts = {
        'happy': [
            "Today was an amazing day! I got a promotion at work and my friends threw me a surprise party.",
            "I'm feeling so joyful right now. Everything seems to be going right in my life.",
            "The sun is shining, birds are singing, and I just feel so content with everything."
        ],
        'sad': [
            "I've been feeling down lately. Nothing seems to bring me joy anymore.",
            "I miss how things used to be. Everything feels so different now.",
            "It's been a tough week. I just can't seem to shake this feeling of emptiness."
        ],
        'angry': [
            "I can't believe they would do this to me! After everything I've done for them!",
            "This is absolutely infuriating. I've never been so mad in my life.",
            "The way they treated me was completely unacceptable. I'm still fuming about it."
        ],
        'fearful': [
            "I'm really worried about the upcoming presentation. What if I mess up?",
            "The strange noises outside my window are making me anxious. I can't sleep.",
            "I'm terrified of what might happen next. The uncertainty is overwhelming."
        ],
        'neutral': [
            "I went to the store today and bought some groceries. Then I came home and cooked dinner.",
            "The weather today is mild. Not too hot, not too cold.",
            "I read a book for about an hour. It was informative."
        ],
        'surprised': [
            "I couldn't believe my eyes when I saw what happened! It was completely unexpected.",
            "The plot twist in that movie caught me completely off guard. I was shocked!",
            "When they announced the winner, my jaw dropped. I never saw that coming."
        ],
        'disgust': [
            "The smell in that room was absolutely revolting. I had to leave immediately.",
            "I found something moldy in the back of my fridge and it made me feel sick.",
            "The way they behaved at the party was completely distasteful. I was repulsed."
        ]
    }
    
    # Create a dataframe
    data = []
    for emotion, texts in sample_texts.items():
        for text in texts:
            data.append({
                'text': text,
                'emotion': emotion
            })
    
    # Create a dataframe and save it
    df = pd.DataFrame(data)
    
    # Create more samples to ensure enough data for stratification
    expanded_data = []
    for emotion, texts in sample_texts.items():
        # Add each text multiple times with slight variations
        for text in texts:
            for i in range(5):  # Create 5 variations of each text
                variation = text + f" {i+1}."
                expanded_data.append({
                    'text': variation,
                    'emotion': emotion
                })
    
    # Create expanded dataframe
    expanded_df = pd.DataFrame(expanded_data)
    
    # Split into train and test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(expanded_df, test_size=0.2, random_state=42, stratify=expanded_df['emotion'])
    
    # Save the dataframes
    train_df.to_csv("data/text/train.csv", index=False)
    test_df.to_csv("data/text/test.csv", index=False)
    
    print(f"Sample text dataset created with {len(df)} entries!")
    print(f"Emotion distribution:\n{df['emotion'].value_counts()}")

if __name__ == "__main__":
    print("Starting dataset download and preprocessing...")
    
    # Download and preprocess RAVDESS dataset
    download_ravdess()
    
    # Download and preprocess Stream-of-consciousness dataset
    download_soc_dataset()
    
    print("Dataset download and preprocessing completed!")
