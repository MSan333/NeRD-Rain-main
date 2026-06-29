#!/bin/bash
# NeRD-Rain H20 训练脚本 — 直接 ps=256
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="${TRAIN_DIR:-../data/Rain200L/train/}"
VAL_DIR="${VAL_DIR:-../data/Rain200L/test/}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/}"

echo "=========================================="
echo "  NeRD-Rain H20 训练 (ps=256)"
echo "=========================================="
echo "  Train: $TRAIN_DIR"
echo "  Val:   $VAL_DIR"
echo "  GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "=========================================="

python "$SCRIPT_DIR/train_h20.py" \
  --batch_size 20 \
  --patch_size 256 \
  --num_epochs 400 \
  --start_lr 4.47e-4 \
  --end_lr 4.47e-6 \
  --warmup_epochs 3 \
  --train_dir "$TRAIN_DIR" \
  --val_dir "$VAL_DIR" \
  --model_save_dir "$CKPT_DIR" \
  --session Multiscale_H20_256_bs20 \
  --num_workers 16 \
  | tee "$SCRIPT_DIR/logs_H20_256_bs20.txt"

FINAL="$CKPT_DIR/Deraininig/models/Multiscale_H20_256/model_best.pth"
echo ""
echo "训练完成！最优权重: $FINAL"
