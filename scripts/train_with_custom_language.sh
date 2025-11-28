#!/bin/bash

# Training script with custom language prompts per dataset
# Last 7 datasets (1110_*): "pick up the box and put on the table"
# Other datasets: "pick up the box and put on the table and lift up the robot arm"

  CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune.py \
    --dataset-path \
      /data/anthony/ucr_vla/output/1120/vla_box/moby1020_1 \
      /data/anthony/ucr_vla/output/1120/vla_box/moby1020_2 \
      /data/anthony/ucr_vla/output/1120/vla_box/moby1020_3 \
      /data/anthony/ucr_vla/output/1120/vla_box/moby_1027_1 \
      /data/anthony/ucr_vla/output/1120/vla_box/moby_1027_2 \
      /data/anthony/ucr_vla/output/1120/vla_box/moby_1027_3 \
      /data/anthony/ucr_vla/output/1120/vla_box/moby_move_box \
      /data/anthony/ucr_vla/output/1120/vla_box/moby_recover_1021_1 \
      /data/anthony/ucr_vla/output/1120/vla_box/moby_test4 \
      /data/anthony/ucr_vla/output/1120/vla_box/moby_test5 \
      /data/anthony/ucr_vla/output/1120/vla_box/moby_test6 \
      /data/anthony/ucr_vla/output/1120/vla_box/move_box_2 \
      /data/anthony/ucr_vla/output/1120/vla_box/new_tab_test \
      /data/anthony/ucr_vla/output/1120/vla_box/pick_up_the_box_1110_0 \
      /data/anthony/ucr_vla/output/1120/vla_box/pick_up_the_box_1110_1 \
      /data/anthony/ucr_vla/output/1120/vla_box/pick_up_the_box_1110_2 \
      /data/anthony/ucr_vla/output/1120/vla_box/pick_up_the_box_1110_3 \
      /data/anthony/ucr_vla/output/1120/vla_box/pick_up_the_box_1110_4 \
      /data/anthony/ucr_vla/output/1120/vla_box/pick_up_the_box_1110_5 \
      /data/anthony/ucr_vla/output/1120/vla_box/pick_up_the_box_1110_6 \
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
      "pick up the box and put on the table" \
      "pick up the box and put on the table" \
      "pick up the box and put on the table" \
      "pick up the box and put on the table" \
      "pick up the box and put on the table" \
      "pick up the box and put on the table" \
      "pick up the box and put on the table" \
    --num-gpus 1 \
    --batch-size 64 \
    --output-dir checkpoints/1120_filter_data_short_image_history_32action \
    --data-config ucr_wblm_moby_history \
    --max-steps 24000 \
    --no-tune-visual \
    --save-steps 24000