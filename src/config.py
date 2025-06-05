"""
Configuration file with optimized parameters for improved Ing F1 score.
"""

import os
# --- Directory Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULTS_SAVE_DIR = os.path.join(OUTPUT_DIR, "results")
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")

# --- Data Files ---
DOCRED_TRAIN_FILE = os.path.join(DATA_DIR, "train_annotated.json")
DOCRED_DEV_FILE = os.path.join(DATA_DIR, "dev.json")
DOCRED_TEST_FILE = os.path.join(DATA_DIR, "test.json")
DOCRED_REL_INFO_FILE = os.path.join(DATA_DIR, "rel_info.json")

# --- Output Files ---
MODEL_SAVE_PATH = os.path.join(RESULTS_SAVE_DIR, "model_enhanced.pt")

# --- Model Hyperparameters ---
BASE_MODEL_NAME = "bert-base-uncased"  # Placeholder, paper uses Llama-3.1-8B
MAX_SEQ_LENGTH = 512  # Reduced from 512 to avoid memory issues
BATCH_SIZE = 32  # Reduced to avoid memory issues
LEARNING_RATE = 3e-5  # Slightly increased for better convergence
NUM_EPOCHS = 3  # Paper curriculum implies more complex training schedule
WEIGHT_DECAY = 0.01
CONFIDENCE_LOSS_LAMBDA = 0.5  # Weight for confidence loss term
# --- Early Stopping ---
PATIENCE = 2  # Number of validation checks with no improvement before stopping
# --- Model Hyperparameters ---
DROPOUT = 0.1  # Common default for transformers like BERT

# --- Curriculum Learning Parameters ---
# Stage difficulty thresholds (sentence distance between entities)
STAGE_1_MAX_DISTANCE = 1  # Same or adjacent sentences
STAGE_2_MAX_DISTANCE = 4  # Within paragraph
# Stage 3 is anything beyond Stage 2

# --- Data Quality Parameters ---
CONFIDENCE_THRESHOLD = 0.3  # Threshold for filtering low-quality instances

# --- Data Augmentation Parameters ---
ENTITY_SUBSTITUTION_PROB = 0.3  # Probability of substituting an entity
EVIDENCE_MASKING_PROB = 0.2  # Probability of masking evidence sentences
RELATION_TRANSFER_PROB = 0.2  # Probability of transferring relations

# --- Evaluation Parameters ---
PREDICTION_THRESHOLD = {
    1: 0.4,  # Lower threshold for simple relations (Stage 1)
    2: 0.45,  # Medium threshold for multi-hop relations (Stage 2)
    3: 0.5   # Higher threshold for complex relations (Stage 3)
}

# --- Random Seed ---
SEED = 43
