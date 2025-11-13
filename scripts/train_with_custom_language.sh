#!/bin/bash

# Training script with custom language prompts per dataset
# Last 7 datasets (1110_*): "pick up the box and put on the table"
# Other datasets: "pick up the box and put on the table and lift up the robot arm"

CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune.py \
  --dataset-path \
    /data/anthony/ucr_vla/output/1016/moby_vla \
    /data/anthony/ucr_vla/output/1017/moby_vla \
    /data/anthony/ucr_vla/output/1020/moby_vla/moby1020_1 \
    /data/anthony/ucr_vla/output/1020/moby_vla/moby1020_2 \
    /data/anthony/ucr_vla/output/1020/moby_vla/moby1020_3 \
    /data/anthony/ucr_vla/output/1020/moby_vla/moby_test4 \
    /data/anthony/ucr_vla/output/1020/moby_vla/moby_test5 \
    /data/anthony/ucr_vla/output/1020/moby_vla/moby_test6 \
    /data/anthony/ucr_vla/output/1021/moby_vla/moby_move_box \
    /data/anthony/ucr_vla/output/1021/moby_vla/moby_recover_1021_1 \
    /data/anthony/ucr_vla/output/1021/moby_vla/move_box_2 \
    /data/anthony/ucr_vla/output/1027/moby_vla/moby_1027_1 \
    /data/anthony/ucr_vla/output/1027/moby_vla/moby_1027_2 \
    /data/anthony/ucr_vla/output/1027/moby_vla/moby_1027_3 \
    /data/anthony/ucr_vla/output/1110/moby_vla/pick_up_the_box_1110_0 \
    /data/anthony/ucr_vla/output/1110/moby_vla/pick_up_the_box_1110_1 \
    /data/anthony/ucr_vla/output/1110/moby_vla/pick_up_the_box_1110_2 \
    /data/anthony/ucr_vla/output/1110/moby_vla/pick_up_the_box_1110_4 \
    /data/anthony/ucr_vla/output/1110/moby_vla/pick_up_the_box_1110_5 \
    /data/anthony/ucr_vla/output/1110/moby_vla/pick_up_the_box_1110_6 \
  --dataset-language-prompts \
    "pick up the box and put on the table and lift up the robot arm" \
    "pick up the box and put on the table and lift up the robot arm" \
    "pick up the box and put on the table and lift up the robot arm" \
    "pick up the box and put on the table and lift up the robot arm" \
    "pick up the box and put on the table and lift up the robot arm" \
    "pick up the box and put on the table and lift up the robot arm" \
    "pick up the box and put on the table and lift up the robot arm" \
    "pick up the box and put on the table and lift up the robot arm" \
    "pick up the box and put on the table and lift up the robot arm" \
    "pick up the box and put on the table and lift up the robot arm" \
    "pick up the box and put on the table and lift up the robot arm" \
    "pick up the box and put on the table and lift up the robot arm" \
    "pick up the box and put on the table and lift up the robot arm" \
    "pick up the box and put on the table and lift up the robot arm" \
    "pick up the box and put on the table" \
    "pick up the box and put on the table" \
    "pick up the box and put on the table" \
    "pick up the box and put on the table" \
    "pick up the box and put on the table" \
    "pick up the box and put on the table" \
  --num-gpus 1 \
  --batch-size 64 \
  --output-dir checkpoints/1112_250_demos_multi_language_prompt_action_horizon16_two_frame_one_joint_filter_data \
  --data-config ucr_wblm_moby_history \
  --max-steps 20000 \
  --no-tune-visual \
  --save-steps 20000
