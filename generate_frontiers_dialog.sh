#!/bin/bash
#SBATCH -J generate_dialog
#SBATCH -p <partition_name>
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=8
#SBATCH --time=4-00:00:00
#SBATCH --output=instance_job_%j.out
#SBATCH --error=instance_job_%j.err

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5

srun python -u generate_frontiers_dialog.py \
    --task instance \
    --vocabulary hm3d \
    --scene_ids all \
    --shortest_path_threshold 0.1 \
    --target_detected_threshold 5 \
    --episodes_file_path VL-LN-Bench/raw_data/mp3d/train/train_iign.json.gz \
    --habitat_config_path dialog_generation/config/tasks/dialog_mp3d.yaml \
    --baseline_config_path dialog_generation/config/expertiments/gen_videos.yaml \
    --normal_category_path dialog_generation/normal_category.json \
    --pointnav_policy_path VL-LN-Bench/pointnav_weights.pth\
    --scene_summary_path VL-LN-Bench/raw_data/mp3d/scene_summary\
    --output_dir data/exploration_rename/instance_train_dialog \
