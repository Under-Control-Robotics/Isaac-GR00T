#!/bin/bash
export NUM_GPUS=1

# Base directory containing all move tray datasets
BASE_PATH="/data/anthony/ucr_vla/output/1217_move_tray"

# Automatically find all subdirectories
echo "Discovering datasets in: $BASE_PATH"
DATASETS=()
for dir in "$BASE_PATH"/*/ ; do
    if [ -d "$dir" ]; then
        # Get just the directory name (not full path)
        dirname=$(basename "$dir")
        DATASETS+=("$dirname")
        echo "  Found: $dirname"
    fi
done

echo ""
echo "Total datasets found: ${#DATASETS[@]}"
echo ""

# Build dataset path arguments (tyro expects: --dataset-path path1 path2 path3)
DATASET_ARGS="--dataset-path"
for dataset in "${DATASETS[@]}"; do
  DATASET_ARGS="$DATASET_ARGS $BASE_PATH/$dataset"
done

# Output directory for this finetune run
OUTPUT_DIR="./checkpoints/1218_move_tray_finetune_1.6"

echo "Starting finetune with:"
echo "  Base model: nvidia/GR00T-N1.6-3B"
echo "  Datasets: ${#DATASETS[@]} directories"
echo "  Output: $OUTPUT_DIR"
echo "  Max steps: 25000"
echo "  Batch size: 64"
echo "  Learning rate: 1e-4"
echo ""

CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  $DATASET_ARGS \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path ./ucr_modality_config.py \
  --tune-diffusion-model \
  --tune-projector \
  --num-gpus 1 \
  --output-dir "$OUTPUT_DIR" \
  --global-batch-size 64 \
  --learning-rate 1e-4 \
  --max-steps 10000 \
  --save-steps 10000
