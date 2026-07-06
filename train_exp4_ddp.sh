#!/bin/bash
# exp4: SFStar 模块训练（续训，GPU 2,3）
# 2卡 DDP, 500 epoch, lr=2e-4, resume from epoch 53

CUDA_VISIBLE_DEVICES=2,3 /home/guo_shuaile/.conda/envs/nerd/bin/python -m torch.distributed.run \
    --nproc_per_node=2 \
    train_exp4.py \
    --session "SFStar_2GPU_lr2e4" \
    --batch_size 1 \
    --patch_size 256 \
    2>&1 | tee logs_exp4_resume.txt
