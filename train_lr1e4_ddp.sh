#!/bin/bash

# 2卡 3090 DDP 续训（降LR至1e-4）
# 从 exp3 best checkpoint 出发，lr=1e-4, cosine 500 epoch
# 使用方法: screen -S exp3_lr1e4 && bash train_lr1e4_ddp.sh

# screen -S exp3_lr1e4
# bash train_lr1e4_ddp.sh

# 激活conda环境
source activate nerd

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    train_lr1e4.py \
    --session "DDP_2GPU_exp3_lr1e4" \
    --batch_size 1 \
    --patch_size 256 \
    2>&1 | tee logs_exp3_lr1e4.txt
