#!/bin/bash
SESSION_NAME="hicd_v9"
PROJECT_DIR="/mnt/f/mambacd/home/HICD_v7"
LOG_FILE="$PROJECT_DIR/checkpoints/v9_rare_oversample/train.log"
CKPT_DIR="$PROJECT_DIR/checkpoints/v9_rare_oversample"
RESUME_CKPT="$CKPT_DIR/checkpoint_epoch_latest.pth"

# Kill existing session if any
screen -S $SESSION_NAME -X quit 2>/dev/null
sleep 1

# Create checkpoint dir
mkdir -p "$CKPT_DIR"

# Build resume flag if checkpoint exists
RESUME_FLAG=""
if [ -f "$RESUME_CKPT" ]; then
    RESUME_FLAG="--resume $RESUME_CKPT"
    echo "Resuming from: $RESUME_CKPT"
fi

# Launch training in screen session
screen -dmS $SESSION_NAME bash -c "
cd $PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR
/root/miniconda/envs/mamba/bin/python script/train_full_v7.py \
  --dataset 0617final \
  --data_dir /mnt/d/CD/0617final_256_v2 \
  --batch_size 32 \
  --img_size 256 \
  --max_epochs 50 \
  --learning_rate 1e-4 \
  --use_amp \
  --num_workers 4 \
  --exp_name v9_rare_oversample \
  --backbone_type dinov3 \
  --state_aware \
  --rare_class_oversample \
  --rare_class_boost 50.0 \
  --freeze_backbone_epochs 0 \
  $RESUME_FLAG \
  2>&1 | tee -a $LOG_FILE
"

sleep 5

if screen -ls | grep -q $SESSION_NAME; then
    echo "Training started in screen session: $SESSION_NAME"
    echo "View: screen -r $SESSION_NAME"
    echo "Log: tail -f $LOG_FILE"
else
    echo "ERROR: Failed to start screen session"
    exit 1
fi
