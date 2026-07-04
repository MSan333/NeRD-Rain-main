#!/bin/bash

# 2卡 3090 DDP 续训 exp3 (从ep162恢复)
# 每卡 bs=1, 总 bs=2, lr=4e-4 (保持原lr,optimizer从ckpt恢复)
# 使用方法: screen -S exp3 && bash train_ddp.sh

# 激活conda环境
source activate nerd

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    train.py \
    --session "DDP_4GPU_exp3" \
    --batch_size 1 \
    --num_epochs 1000 \
    --patch_size 256 \
    2>&1 | tee logs_Rain200L_exp3_ddp.txt
