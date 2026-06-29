#!/bin/bash
# NeRD-Rain H20 课程学习训练脚本
# 基于 doc/H20训练加速方案.md 四阶段课程学习策略
# 32 → 64 → 128 → 256 patch_size 渐进训练
#
# 学习率缩放策略：sqrt 缩放（适用于 Adam 优化器）
# new_lr = base_lr × √(batch_size)
# 原始配置：batch_size=1, start_lr=1e-4, end_lr=1e-6

set -e

# ========== 配置 ==========
TRAIN_DIR="${TRAIN_DIR:-../data/Rain200L/train/}"
VAL_DIR="${VAL_DIR:-../data/Rain200L/test/}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/}"
NUM_WORKERS="${NUM_WORKERS:-16}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "  NeRD-Rain H20 课程学习训练"
echo "  32 → 64 → 128 → 256"
echo "=========================================="
echo "  Train: $TRAIN_DIR"
echo "  Val:   $VAL_DIR"
echo "  Ckpt:  $CKPT_DIR"
echo "  GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "=========================================="

# ========== 阶段一: patch_size=32, batch_size=64, epochs=200 ==========
# sqrt(64) = 8, start_lr = 1e-4 × 8 = 8e-4, end_lr = 1e-6 × 8 = 8e-6
echo ""
echo "[Stage 1/4] patch_size=32, batch_size=64, epochs=200, lr=8e-4→8e-6 (sqrt缩放)"
echo "  目标: 学习颜色映射、基础亮度校正"
python "$SCRIPT_DIR/train_h20.py" \
  --batch_size 64 \
  --patch_size 32 \
  --num_epochs 200 \
  --start_lr 8e-4 \
  --end_lr 8e-6 \
  --warmup_epochs 5 \
  --train_dir "$TRAIN_DIR" \
  --val_dir "$VAL_DIR" \
  --model_save_dir "$CKPT_DIR" \
  --session Multiscale_H20_stage1 \
  --num_workers "$NUM_WORKERS" \
  | tee "$SCRIPT_DIR/logs_stage1.txt"

STAGE1_WEIGHTS="$CKPT_DIR/Deraininig/models/Multiscale_H20_stage1/model_best.pth"
echo "[Stage 1] Done! Best weights: $STAGE1_WEIGHTS"

# ========== 阶段二: patch_size=64, batch_size=32, epochs=400 ==========
# sqrt(32) ≈ 5.66, start_lr = 1e-4 × 5.66 ≈ 5.66e-4, end_lr = 1e-6 × 5.66 ≈ 5.66e-6
echo ""
echo "[Stage 2/4] patch_size=64, batch_size=32, epochs=400, lr=5.66e-4→5.66e-6 (sqrt缩放)"
echo "  目标: 学习基础去雨能力、粗略纹理恢复"
python "$SCRIPT_DIR/train_h20.py" \
  --batch_size 32 \
  --patch_size 64 \
  --num_epochs 400 \
  --start_lr 5.66e-4 \
  --end_lr 5.66e-6 \
  --warmup_epochs 5 \
  --pretrain_weights "$STAGE1_WEIGHTS" \
  --train_dir "$TRAIN_DIR" \
  --val_dir "$VAL_DIR" \
  --model_save_dir "$CKPT_DIR" \
  --session Multiscale_H20_stage2 \
  --num_workers "$NUM_WORKERS" \
  | tee "$SCRIPT_DIR/logs_stage2.txt"

STAGE2_WEIGHTS="$CKPT_DIR/Deraininig/models/Multiscale_H20_stage2/model_best.pth"
echo "[Stage 2] Done! Best weights: $STAGE2_WEIGHTS"

# ========== 阶段三: patch_size=128, batch_size=16, epochs=1000 ==========
# sqrt(16) = 4, start_lr = 1e-4 × 4 = 4e-4, end_lr = 1e-6 × 4 = 4e-6
echo ""
echo "[Stage 3/4] patch_size=128, batch_size=16, epochs=1000, lr=4e-4→4e-6 (sqrt缩放)"
echo "  目标: 学习中等尺度雨纹、局部纹理细节"
python "$SCRIPT_DIR/train_h20.py" \
  --batch_size 16 \
  --patch_size 128 \
  --num_epochs 1000 \
  --start_lr 4e-4 \
  --end_lr 4e-6 \
  --warmup_epochs 5 \
  --pretrain_weights "$STAGE2_WEIGHTS" \
  --train_dir "$TRAIN_DIR" \
  --val_dir "$VAL_DIR" \
  --model_save_dir "$CKPT_DIR" \
  --session Multiscale_H20_stage3 \
  --num_workers "$NUM_WORKERS" \
  | tee "$SCRIPT_DIR/logs_stage3.txt"

STAGE3_WEIGHTS="$CKPT_DIR/Deraininig/models/Multiscale_H20_stage3/model_best.pth"
echo "[Stage 3] Done! Best weights: $STAGE3_WEIGHTS"

# ========== 阶段四: patch_size=256, batch_size=8, epochs=1400 ==========
# sqrt(8) ≈ 2.83, start_lr = 1e-4 × 2.83 ≈ 2.83e-4, end_lr = 1e-6 × 2.83 ≈ 2.83e-6
echo ""
echo "[Stage 4/4] patch_size=256, batch_size=8, epochs=1400, lr=2.83e-4→2.83e-6 (sqrt缩放)"
echo "  目标: 完整去雨能力、全局上下文、高频细节精细化"
python "$SCRIPT_DIR/train_h20.py" \
  --batch_size 8 \
  --patch_size 256 \
  --num_epochs 1400 \
  --start_lr 2.83e-4 \
  --end_lr 2.83e-6 \
  --warmup_epochs 3 \
  --pretrain_weights "$STAGE3_WEIGHTS" \
  --train_dir "$TRAIN_DIR" \
  --val_dir "$VAL_DIR" \
  --model_save_dir "$CKPT_DIR" \
  --session Multiscale_H20_stage4 \
  --num_workers "$NUM_WORKERS" \
  | tee "$SCRIPT_DIR/logs_stage4.txt"

FINAL_WEIGHTS="$CKPT_DIR/Deraininig/models/Multiscale_H20_stage4/model_best.pth"

echo ""
echo "=========================================="
echo "  课程学习训练完成！"
echo "  最终权重: $FINAL_WEIGHTS"
echo "=========================================="
