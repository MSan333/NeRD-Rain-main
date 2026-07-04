#!/bin/bash

# 2卡 3090 DDP 续训（降LR至2e-5）
# 从 DDP_4GPU_exp3_lr5e5 best checkpoint 出发，lr=2e-5, cosine 200 epoch
# 使用方法: screen -S exp3_lr2e5 && bash train_lr2e5_ddp.sh

# 激活conda环境
source activate nerd

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    train_lr2e5.py \
    --session "DDP_2GPU_exp3_lr2e5" \
    --batch_size 1 \
    --patch_size 256 \
    2>&1 | tee logs_exp3_lr2e5.txt
