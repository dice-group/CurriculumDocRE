#!/bin/bash
# run_bert_cur.sh – CurriculumDocRE training with BERT-base

export CUDA_VISIBLE_DEVICES=${1:-0}

DATA_DIR="data/redocred"
TRAIN_FILE="train.json"
DEV_FILE="dev.json"
TEST_FILE="test.json"
SAVE_PATH="output/bert_curriculum"

PHASE1_EPOCHS=10
PHASE2_EPOCHS=10
PHASE3_EPOCHS=10
MAX_ALPHA=2.0

# BERT uses slightly different learning rate
BATCH_SIZE=4
GRAD_ACC=2
LR=5e-5
WARMUP_RATIO=0.06
EPOCHS=30
EVAL_STEPS=500
SEED=66
POS_WEIGHT=20.0
EVI_LAMBDA=0.5

AUGMENT=0
AUGMENT_FACTOR=1
USE_WANDB=0
WANDB_PROJECT="CurriculumDocRE"
WANDB_NAME="bert_curriculum"

CMD="python run.py --do_train --curriculum \
    --data_dir ${DATA_DIR} \
    --train_file ${TRAIN_FILE} \
    --dev_file ${DEV_FILE} \
    --test_file ${TEST_FILE} \
    --save_path ${SAVE_PATH} \
    --transformer_type bert \
    --model_name_or_path bert-base-uncased \
    --train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --learning_rate ${LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --num_train_epochs ${EPOCHS} \
    --evaluation_steps ${EVAL_STEPS} \
    --seed ${SEED} \
    --pos_weight ${POS_WEIGHT} \
    --evi_lambda ${EVI_LAMBDA} \
    --phase1_epochs ${PHASE1_EPOCHS} \
    --phase2_epochs ${PHASE2_EPOCHS} \
    --phase3_epochs ${PHASE3_EPOCHS} \
    --max_alpha ${MAX_ALPHA}"

if [ ${AUGMENT} -eq 1 ]; then
    CMD="${CMD} --augment --augment_factor ${AUGMENT_FACTOR}"
fi

if [ ${USE_WANDB} -eq 1 ]; then
    CMD="${CMD} --wandb --wandb_project ${WANDB_PROJECT} --wandb_name ${WANDB_NAME}"
fi

echo "Running: ${CMD}"
eval ${CMD}
