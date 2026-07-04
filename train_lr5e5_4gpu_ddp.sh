#!/bin/bash

# 4卡 3090 DDP 续训（lr=5e-5）
# 从 exp3_lr4e5 best checkpoint 出发，lr=5e-5, cosine 500 epoch
# 使用方法: screen -S exp3_lr5e5_4gpu && bash train_lr5e5_4gpu_ddp.sh

# 激活conda环境
source activate nerd

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    train_lr5e5_4gpu.py \
    --session "DDP_4GPU_exp3_lr5e5" \
    --batch_size 1 \
    --patch_size 256 \
    2>&1 | tee logs_exp3_lr5e5_4gpu.txt
