#!/bin/bash
cd /mnt/f/mambacd/home/HICD_v7
export PYTHONPATH="/mnt/f/mambacd/home/HICD_v7:$PYTHONPATH"
source ~/miniconda/bin/activate
conda activate mamba

python script/train_full_v7.py \
    --dataset second \
    --data_dir /mnt/c/SECOND_256 \
    --batch_size 16 \
    --img_size 256 \
    --max_epochs 200 \
    --learning_rate 1e-4 \
    --freeze_backbone_epochs 5 \
    --use_amp \
    --num_workers 4 \
    --exp_name v7_second_256_v2 \
    --pixel_class_weights 2.931 0.160 0.061 2.452 0.090 0.307
