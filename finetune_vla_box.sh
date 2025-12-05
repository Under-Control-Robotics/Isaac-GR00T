#!/bin/bash

# GR00T Finetuning Script for VLA Box Dataset
# Usage: ./finetune_vla_box.sh
# This script uses all folders under /data/vla_data/dataset/vla_box/

# Parent dataset directory
PARENT_DIR="/data/vla_data/dataset/vla_box"

# Check if parent directory exists
if [ ! -d "$PARENT_DIR" ]; then
    echo "Error: Parent dataset directory does not exist: $PARENT_DIR"
    exit 1
fi

# Get all subdirectories into an array
mapfile -t DATASET_PATHS < <(find "$PARENT_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

# Check if any datasets found
if [ ${#DATASET_PATHS[@]} -eq 0 ]; then
    echo "Error: No datasets found in $PARENT_DIR"
    exit 1
fi

echo "Starting finetuning with all datasets under: $PARENT_DIR"
echo "Dataset paths: ${DATASET_PATHS[*]}"

CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune.py \
    --dataset-path "${DATASET_PATHS[@]}" \
    --num-gpus 1 \
    --batch-size 64 \
    --output-dir "checkpoints/vla_box_all_wblm" \
    --data-config ucr_wblm_moby_history \
    --max-steps 5000 \
    --no-tune-visual \
    --save-steps 5000
