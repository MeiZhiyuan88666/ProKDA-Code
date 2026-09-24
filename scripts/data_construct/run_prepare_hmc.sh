#!/bin/bash

set -e
set -x

DATASET_DIR=""
DATASET_NAME="hmc"
SPLIT="train.json"
IMAGE_DIR=""
OUTPUT_DIR=""
COMMENT="hmc local train preprocess"

python prepare_data.py \
  --dataset_dir "$DATASET_DIR" \
  --dataset_name "$DATASET_NAME" \
  --split "$SPLIT" \
  --image_dir "$IMAGE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --comment "$COMMENT"