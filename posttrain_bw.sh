CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune_bw.py  \
    --dataset-path /data/anthony/ucr_ros/data_files/golden_data_lerobot /data/anthony/ucr_ros/data_files/vla_dataset_output/golden_lerobot/2026-02-27_20:51:48.765331/episode_00007 \
    /data/anthony/ucr_ros/data_files/vla_dataset_output/golden_lerobot/2026-02-27_20:51:48.765331/episode_00008 /data/anthony/ucr_ros/data_files/vla_dataset_output/golden_lerobot/2026-02-27_20:51:48.765331/episode_00012   \
    --num-gpus 1   --batch-size 64   --output-dir checkpoints/0309_golden_post_train_bw_03012   --data-config ucr_wblm_moby_history   --max-steps 3000   --save-steps 3000 --learning_rate 5e-6 --base_model_path checkpoints/0309_pretrain_dagger_10x_data_aug_bw_2026-3-11/checkpoint-60000/
