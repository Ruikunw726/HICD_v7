#!/bin/bash
cd /mnt/f/mambacd/home/HICD_v7
export PYTHONPATH=/mnt/f/mambacd/home/HICD_v7
source ~/miniconda/bin/activate
conda activate mamba
python -B script/train_full_v7.py \
  --dataset 0617final \
  --data_dir /mnt/d/CD/0617final \
  --batch_size 32 \
  --img_size 256 \
  --max_epochs 50 \
  --learning_rate 1e-4 \
  --freeze_backbone_epochs 20 \
  --use_amp \
  --num_workers 4 \
  --backbone_type dinov3 \
  --backbone dinov3_base \
  --exp_name v7_0617_phased_weights \
  --config_dir datasets/configs \
  --w_state 2.0 \
  --w_contrastive 0.05 \
  --save_interval 10