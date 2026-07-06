#!/bin/bash
# exp5: SFStar 续训（从 exp4 best 出发，lr 重置 2e-4→2e-6，300 epoch）
# GPU 0,1, 2卡 DDP

CUDA_VISIBLE_DEVICES=0,3 /home/guo_shuaile/.conda/envs/nerd/bin/python -m torch.distributed.run \
    --nproc_per_node=2 \
    train_exp5.py \
    --session "SFStar_2GPU_lr2e4_ft300" \
    --batch_size 1 \
    --patch_size 256 \
    2>&1 | tee logs_exp5.txt
