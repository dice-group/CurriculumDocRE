# CurriculumDocRE: A Curriculum Learning Approach for DocRE

This repository contains the implementation of **CurriculumDocRE**, a curriculum learning framework for document-level relation extraction (DocRE). The model progressively learns from easy to hard relations based on a multi‑faceted complexity score that considers sentence distance, coreference resolution, and relation rarity.


## Requirements
``` torch>=1.11.0
transformers>=4.14.1
numpy>=1.20.0
scikit-learn>=1.0.0
tqdm>=4.62.0
ujson
wandb
opt-einsum
```

Install with: `pip install -r requirements.txt`

## Directory Structure
```
CurriculumDocRE/
├── README.md
├── requirements.txt
├── run.py
├── train.py
├── config.py
├── utils.py
├── prepro.py
├── model.py
├── losses.py
├── long_seq.py
├── data_augmentation.py
├── evaluation.py
├── args.py
├── scripts/
│   ├── run_roberta_cur.sh
│   └── run_bert_cur.sh
└── data/ (you need to download Re‑DocRED, CDR, GDA yourself)
```

## Data Preparation

1. Download the required datasets:
   - **Re‑DocRED**: [https://github.com/tonytan48/Re-DocRED](https://github.com/tonytan48/Re-DocRED)
   - **RE2‑DocRED**: [https://github.com/.../RE2-DocRED](https://github.com/.../RE2-DocRED)
   - **CDR**: [BioCreative V CDR task](https://biocreative.sourceforge.net/resources/cdr.html)
   - **GDA**: [DFKI-SLT/GDA on Hugging Face](https://huggingface.co/DFKI-SLT/GDA)

2. Place the files in the `data/` directory with the following structure:

```
data/
├── redocred/
│ ├── train.json
│ ├── dev.json
│ └── test.json
├── re2docred/ (optional)
├── cdr/
│ ├── train.json
│ ├── dev.json
│ └── test.json
└── gda/
├── train.json
├── dev.json
└── test.json
```

3. Also place `rel2id.json` and `rel_info.json` (provided in the DocRED repository) in a `meta/` folder.

## Running the Code

### Training

#### Fully‑supervised baseline (no curriculum)

```bash
python -m src.run --do_train --data_dir data/redocred --train_file train.json --dev_file dev.json --test_file test.json --num_train_epochs 30
```
#### Curriculum learning (three stages)
```bash
python -m src.run --do_train --curriculum --phase1_epochs 10 --phase2_epochs 10 --phase3_epochs 10 --max_alpha 2.0 --data_dir data/redocred
```
#### With augmentation for rare relations
```bash
python -m src.run --do_train --curriculum --augment --augment_factor 1 --data_dir data/redocred
```
#### Training on biomedical datasets (binary)

#### CDR
```bash
python -m src.run --do_train --curriculum --num_labels 2 --num_class 2 --evi_lambda 0 --data_dir data/cdr --transformer_type bert --model_name_or_path bert-base-uncased
```

#### GDA

```bash
python -m src.run --do_train --curriculum --num_labels 2 --num_class 2 --evi_lambda 0 --data_dir data/gda --train_batch_size 2 --gradient_accumulation_steps 4 --transformer_type bert --model_name_or_path bert-base-uncased
```
### Evaluation

#### On development set
```bash
python -m src.run --load_path ./output/curriculum_redocred/best.ckpt --data_dir data/redocred --dev_file dev.json --evaluate
```
#### On test set 
```bash
python -m src.run --load_path ./output/curriculum_redocred/best.ckpt --data_dir data/redocred --test_file test.json --evaluate --test_only
```

#### Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--curriculum` | `False` | Enable curriculum learning |
| `--phase1_epochs` | `10` | Epochs for Stage 1 (easy relations, distance ≤1) |
| `--phase2_epochs` | `10` | Epochs for Stage 2 (medium relations, distance 2‑4) |
| `--phase3_epochs` | `10` | Epochs for Stage 3 (hard relations, distance >4) |
| `--max_alpha` | `2.0` | Exponent for curriculum weighting (higher = more focus on hard pairs) |
| `--augment` | `False` | Enable data augmentation for rare relations |
| `--augment_factor` | `1` | Number of augmented copies per positive document |
| `--pos_weight` | `20` | Positive class weight for ATLoss |
| `--evi_lambda` | `0.5` | Weight for evidence loss (if evidence annotations are available) |
| `--learning_rate` | `3e-5` (RoBERTa) / `5e-5` (BERT) | Learning rate |
| `--train_batch_size` | `4` | Batch size per GPU |
| `--gradient_accumulation_steps` | `2` | Gradient accumulation steps |
| `--max_seq_length` | `1024` | Maximum input tokens |
| `--warmup_ratio` | `0.06` | Linear warmup proportion |

#### Citation
If you use this code in your research, please cite:
```
@inproceedings{ali2026curriculum,
  title={CurriculumDocRE: A Curriculum Learning Approach for Document-Level Relation Extraction},
  author={Ali, Manzoor and Saleem, Muhammad and Ngonga Ngomo, Axel-Cyrille},
  year={2026}
}
```
