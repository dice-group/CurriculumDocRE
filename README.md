# CurriculumDocRE: A Curriculum Learning Approach for DocRE

This repository contains an implementation of the CurriculumDocRE approach for document-level relation extraction with curriculum learning. The code has been structured to be memory-efficient and properly integrates all components of the curriculum learning approach.

## Overview

The CurriculumDocRE approach implements a curriculum learning strategy for document-level relation extraction, where the model progressively learns from:
1. Simple relations (entities in same/adjacent sentences)
2. Multi-hop relations (entities in different sentences requiring connecting information)
3. Complex relations (entities across paragraphs requiring advanced coreference)

## Key Features

1. **Curriculum-Aware Attention**:
   - Dynamic attention patterns based on curriculum stage
   - Entity-focused attention mechanisms
   - Stage-specific adapters for different relation complexities

2. **Data Quality Enhancement**:
   - Annotation consistency checking
   - Confidence scoring based on multiple factors
   - Active learning for uncertain instances

3. **Data Augmentation**:
   - Entity substitution with contextual awareness
   - Evidence masking for improved robustness
   - Cross-document relation transfer

4. **Optimized Training Pipeline**:
   - Integrated components in a cohesive flow
   - Curriculum-aware loss functions
   - Confidence-weighted evaluation metrics

5. **Optimized Evaluation**:
   - Stage-specific thresholds for improved Ing F1 score
   - Entity type compatibility checking
   - Confidence-based filtering and ranking

## Requirements

```
torch>=1.10.0
transformers>=4.18.0
numpy>=1.20.0
scikit-learn>=1.0.0
tqdm>=4.62.0
```

Install with: `pip install -r requirements.txt`

## Directory Structure

- `src/`: Source code
  - `model.py`: Enhanced model with curriculum-aware attention
  - `data_quality.py`: Data quality enhancement module
  - `data_augmentation.py`: Data augmentation techniques
  - `data_loader.py`: Data loading and preprocessing
  - `train.py`: Training pipeline with curriculum learning
  - `evaluate.py`: Memory-efficient evaluation script
  - `config.py`: Configuration parameters
  - `utils.py`: Utility functions
- `data/`: DocRED dataset files
  - `train_annotated.json`: Training data
  - `dev.json`: Development data
  - `test.json`: Test data
  - `rel_info.json`: Relation type information

## Running the Code

### Training

```bash
python -m src.train
```

This will:
1. Enhance data quality (annotation consistency, confidence scoring)
2. Apply data augmentation (entity substitution, evidence masking, relation transfer)
3. Train the model using curriculum learning (3 stages)

### Evaluation

```bash
python -m src.evaluate
```

This will generate predictions in `output/results/predictions.json`.

### Official Evaluation

To evaluate using the official DocRED metrics:

1. Clone the DocRED repository:
```bash
git clone https://github.com/thunlp/DocRED.git
```

2. Run the evaluation script:
```bash
cd DocRED/code
python eval.py -g ../../data/dev.json -p ../../output/results/predictions.json
```
