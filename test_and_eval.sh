#!/bin/bash
# 测试脚本：推理 + 评估
# 用法: bash test_and_eval.sh <weights_path> [output_dir]

set -e

WEIGHTS=${1:-"./checkpoints/Deraininig/models/Multiscale_H20_256_bs20/model_best.pth"}
OUTPUT_DIR=${2:-"./evaluations/Evalution_Rain200L_Rain200H_SPA-Data/results/Rain200L"}
INPUT_DIR="../data/Rain200L/test/input/"
WIN_SIZE=256
GPUS="0"

echo "=========================================="
echo "  NeRD-Rain 测试 + 评估"
echo "=========================================="
echo "Weights: $WEIGHTS"
echo "Output:  $OUTPUT_DIR"
echo "=========================================="

if [ ! -f "$WEIGHTS" ]; then
    echo "错误: 没有找到权重文件: $WEIGHTS"
    exit 1
fi

# 激活 nerd 环境
source nerd/bin/activate

echo ""
echo "[1/2] 推理生成去雨图像..."
python test.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --weights "$WEIGHTS" \
    --gpus "$GPUS" \
    --win_size "$WIN_SIZE"

echo ""
echo "[2/2] 评估 PSNR/SSIM (Y通道)..."
python evaluate_python.py

echo ""
echo "=========================================="
echo "  测试完成！"
echo "=========================================="
