"""
Global Configuration for Road Damage Detection Project
========================================================
This file contains all project-wide settings and hyperparameters.
"""

import os
from pathlib import Path
import torch as _torch

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent.absolute()
DATASET_PATH = Path("C:/Users/hp/Desktop/archive")
TRAIN_IMG_DIR = DATASET_PATH / "RDD_SPLIT" / "train" / "images"
VAL_IMG_DIR = DATASET_PATH / "RDD_SPLIT" / "val" / "images"
TEST_IMG_DIR = DATASET_PATH / "RDD_SPLIT" / "test" / "images"

TRAIN_LABEL_DIR = DATASET_PATH / "RDD_SPLIT" / "train" / "labels"
VAL_LABEL_DIR = DATASET_PATH / "RDD_SPLIT" / "val" / "labels"
TEST_LABEL_DIR = DATASET_PATH / "RDD_SPLIT" / "test" / "labels"

DATA_YAML = DATASET_PATH / "data.yaml"

# Checkpoint and output directories
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ==================== DATA CONFIGURATION ====================
# Class labels (must match data.yaml)
CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
    4: "Repair"  # Additional class found in dataset labels
}

NUM_CLASSES = len(CLASS_NAMES)

# Image settings
IMAGE_SIZE = 416  # Reduced from 512 for faster CPU training
IMAGE_EXTENSION = ".jpg"

# Dataset split ratios (if rebuilding)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==================== PREPROCESSING ====================
# Median filter kernel size
MEDIAN_FILTER_KERNEL = 5

# CLAHE parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)

# Brightness correction range
BRIGHTNESS_CORRECTION_RANGE = (-30, 30)

# Image normalization (Min-Max)
IMG_MIN = 0.0
IMG_MAX = 255.0

# ==================== MODEL CONFIGURATION ====================
# Backbone
BACKBONE = "MobileNetV3-Small"  # Can be replaced with other lightweight architectures
USE_PRETRAINED = True

# Input/output dimensions
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = NUM_CLASSES

# ==================== TRAINING ====================
BATCH_SIZE = 8  # Reduced from 16 for faster CPU training
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# Data augmentation
USE_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    "flip_horizontal": 0.5,
    "flip_vertical": 0.3,
    "rotation_degrees": 15,
    "brightness_range": 0.2,
    "contrast_range": 0.2,
}

# Training device (auto-detect)
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"

# Loss function weights (if using weighted loss)
LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0]  # Equal weight for all classes

# ==================== EPOCH CONTROL ====================
NUM_EPOCHS = 5

# ==================== EVALUATION ====================
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
EVAL_METRICS = ["mAP@0.5", "Precision", "Recall", "F1-Score"]

# ==================== INFERENCE ====================
INFERENCE_DEVICE = "cpu"
INFERENCE_CONFIDENCE = 0.5
INFERENCE_IOU_THRESHOLD = 0.4
DRAW_BBOX = True
SAVE_PREDICTIONS = True

# ==================== LOGGING & DEBUGGING ====================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
VERBOSE = True
SEED = 42  # For reproducibility

# ==================== FLASK APP ====================
FLASK_DEBUG = True
FLASK_PORT = 5000
FLASK_HOST = "0.0.0.0"
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

# ==================== UTILITY FUNCTIONS ====================
def print_config():
    """Print all configuration settings."""
    print("=" * 60)
    print("PROJECT CONFIGURATION")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Checkpoint Dir: {CHECKPOINT_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"\nModel: {BACKBONE}")
    print(f"Input Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Num Epochs: {NUM_EPOCHS}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

if __name__ == "__main__":
    print_config()
