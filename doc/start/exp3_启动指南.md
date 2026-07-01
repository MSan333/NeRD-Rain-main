# exp3 启动指南

## 1. 实验概述

| 项目 | 配置 |
|------|------|
| 分支 | `server/exp3` |
| 创新点 | RFM（创新点1）+ MFGCP（创新点2）+ HAFL（创新点3） |
| GPU | 4×RTX 3090 DDP |
| Batch Size | 每卡 bs=1，总 bs=4 |
| 学习率 | 4e-4（线性缩放），cosine 衰减到 4e-6 |
| 训练轮数 | 1000 epoch（含 3 epoch warmup） |
| SwanLab 实验名 | `DDP_4GPU_exp3` |

## 2. 服务器环境要求

- **conda 环境**：`nerd`
- **Python 依赖**：参见 `requirements.txt`
- **GPU**：4×RTX 3090（`CUDA_VISIBLE_DEVICES=0,1,2,3`）
- **数据集路径**：`../data/Rain200L/`（需提前准备）

## 3. 启动步骤

```bash
# 1. 拉取代码
git fetch origin
git checkout server/exp3
git pull origin server/exp3

# 2. 激活环境
source activate nerd

# 3. 使用 screen 后台运行
screen -S exp3

# 4. 启动训练
bash train_ddp.sh

# 5. 退出 screen（训练在后台继续）
# Ctrl+A, D
```

## 4. 训练脚本内容说明

- `train_ddp.sh` 使用 `torchrun` 启动 4 进程 DDP 训练
- 训练日志输出到终端，SwanLab 同步到云端
- 模型保存在 `./change2/checkpoints/Deraininig/models/DDP_4GPU_exp3/`

## 5. 监控与恢复

- **SwanLab 面板**：https://swanlab.cn/@sansan/NeRD-Rain
- **断点恢复**：如果训练中断，修改 `train.py` 中 `RESUME = True` 后重新启动即可从最新 checkpoint 恢复
- **查看训练日志**：`screen -r exp3`

## 6. 测评

训练完成后：

```bash
python test_and_evaluate.py \
    --weights ./change2/checkpoints/Deraininig/models/DDP_4GPU_exp3/model_best.pth \
    --input_dir ../data/Rain200L/test/input/ \
    --gt_dir ../data/Rain200L/test/target/ \
    --output_dir ./results/exp3_Rain200L/ \
    --report_dir ./reports/
```
