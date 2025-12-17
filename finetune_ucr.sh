#!/bin/bash
export NUM_GPUS=1

# List of all datasets
DATASETS=(
  "moby1020_1"
  "moby1020_2"
  "moby1020_3"
  "moby_1027_1"
  "moby_1027_2"
  "moby_1027_3"
  "moby_move_box"
  # "moby_pick1"
  # "moby_pick2"
  # "moby_pick3"
  # "moby_pick5"
  # "moby_pick7"
  # "moby_pick8"
  "moby_recover_1021_1"
  "moby_test4"
  "moby_test5"
  "moby_test6"
  "move_box_2"
  "new_tab_test"
  "pick_up_the_box_1110_0"
  "pick_up_the_box_1110_1"
  "pick_up_the_box_1110_2"
  "pick_up_the_box_1110_3"
  "pick_up_the_box_1110_4"
  "pick_up_the_box_1110_5"
  "pick_up_the_box_1110_6"
)

# Build dataset path arguments (tyro expects: --dataset-path path1 path2 path3)
DATASET_ARGS="--dataset-path"
BASE_PATH="/mnt/ucr_drive_ml/vla_data/dataset/real/vla_box"
for dataset in "${DATASETS[@]}"; do
  DATASET_ARGS="$DATASET_ARGS $BASE_PATH/$dataset"
done

CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  $DATASET_ARGS \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path ./ucr_modality_config.py \
  --tune-diffusion-model \
  --tune-projector \
  --num-gpus 1 \
  --output-dir ./checkpoints/ucr_17_finetune \
  --global-batch-size 64 \
  --learning-rate 1e-4 \
  --max-steps 25000 \
  --save-steps 25000
