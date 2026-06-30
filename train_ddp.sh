#!/bin/bash

# 4卡 3090 DDP 训练
# 每卡 bs=1, 总 bs=4, lr=4e-4 (线性缩放)
# 使用方法: screen -S exp2 && bash train_ddp.sh

# 激活conda环境
source activate nerd

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    train.py \
    --session "DDP_4GPU" \
    --batch_size 1 \
    --num_epochs 3000 \
    --patch_size 256 \
    2>&1 | tee logs_Rain200L_ddp.txt
