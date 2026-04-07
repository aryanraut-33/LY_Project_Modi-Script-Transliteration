import os

# Paths
# Based on your directory structure: data/raw/train and data/raw/test
RAW_DATA_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
MODEL_SAVE_DIR = 'outputs/models'
LOG_DIR = 'outputs/logs'

# Image Settings
# Upscaling to 75x75 to meet InceptionResNetV2 minimum requirements
IMAGE_SIZE = (75, 75)
CHANNELS = 1  # Binarized images are grayscale

# Training Hyperparameters 
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.20 # Using 20% of 'train' folder for internal validation
DROPOUT_RATES = [0.3] # Drop-out induced approach
LEARNING_RATE = 0.001

# Dataset Specifics
NUM_CLASSES = 46  # 46 folders in Training & Testing