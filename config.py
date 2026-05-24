# config.py - Configuration file for CurriculumDocRE

import os

# -------------------- Directory Paths --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_SAVE_DIR = os.path.join(OUTPUT_DIR, "results")

# Default dataset file names (adjust per dataset)
TRAIN_FILE = "train.json"
DEV_FILE = "dev.json"
TEST_FILE = "test.json"
REL_INFO_FILE = "rel_info.json"

# -------------------- Model Hyperparameters --------------------
BASE_MODEL_NAME = "roberta-large"       # or "bert-base-uncased"
MAX_SEQ_LENGTH = 1024
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 3e-5                    # 5e-5 for BERT
LEARNING_RATE_ADDED = 1e-4
NUM_TRAIN_EPOCHS = 30
WARMUP_RATIO = 0.06
MAX_GRAD_NORM = 1.0
DROPOUT = 0.1
POS_WEIGHT = 20.0                       # positive class weight for ATLoss

# Evidence loss weight
EVI_LAMBDA = 0.5
EVI_THRESH = 0.2
MAX_SENT_NUM = 25

# -------------------- Curriculum Learning Parameters --------------------
# Sentence distance thresholds
STAGE1_MAX_DIST = 1        # distance ≤ 1 -> easy
STAGE2_MAX_DIST = 4        # 1 < distance ≤ 4 -> medium
# Stage 3: distance > 4 -> hard

MAX_ALPHA = 2.0            # exponent for curriculum weighting (final value)
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 10
PHASE3_EPOCHS = 10

# -------------------- Data Augmentation Parameters --------------------
ENTITY_SUBSTITUTION_PROB = 0.3
EVIDENCE_MASKING_PROB = 0.2
RELATION_TRANSFER_PROB = 0.2
AUGMENT_FACTOR = 1

# -------------------- Evaluation --------------------
PREDICTION_THRESHOLD = 0.5

# -------------------- Random Seed --------------------
SEED = 66

# -------------------- WandB --------------------
WANDB_PROJECT = "CurriculumDocRE"
