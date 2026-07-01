# Exp2 加速收敛方案

## 1. 当前训练状况

- **实验**：exp2（RFM + HAFL，4×3090 DDP）
- **配置**：bs=4, lr=4e-4, cosine schedule, num_epochs=3000
- **进度**：epoch 310, best PSNR 39.97 (epoch 281)
- **问题**：cosine schedule 3000 epoch，LR 在前 1000 epoch 几乎不降（0.0004→0.00035），导致模型在高 LR 下持续震荡，30+ epoch 未刷新 best

## 2. 推荐方案：方案 C — 从 checkpoint 恢复 + 缩短 cosine 周期

- 从 best checkpoint (epoch 281, PSNR 39.97) 恢复
- 将 cosine schedule 的总周期改为 1000 epoch（从当前恢复点计算，即剩余约 700 epoch 完成衰减）
- LR 从 4e-4 衰减到 4e-6，在 700 epoch 内完成
- 预计 2.5 天出最终结果

## 3. 具体代码修改

```python
# train.py 修改项：
RESUME = True  # 从 checkpoint 恢复
num_epochs = 1000  # 缩短总 epoch（从 3000 改为 1000）
# start_lr = 4e-4 不变
# end_lr = 4e-6 不变
# warmup_epochs = 3

# Resume 块中重建 scheduler，缩短 cosine 周期，从恢复点重新开始衰减：
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs - warmup_epochs, eta_min=end_lr)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()  # last_epoch: -1 -> 0，LR 恢复为 start_lr
```

## 4. 预期效果

| Epoch | LR (新 schedule) | PSNR 预估 |
|-------|------------------|----------|
| 300 (恢复点) | 4e-4 | 39.97 |
| 500 | ~2.8e-4 | ~40.3 |
| 700 | ~1.5e-4 | ~40.7 |
| 900 | ~4e-5 | ~41.2 |
| 1000 | 4e-6 | ~41.5+ |

## 5. 对比 baseline

- Baseline 最终结果：41.71 dB (Y通道, 3000 epoch)
- 如果 exp2 RGB 验证达到 40+，Y 通道测试预计 42+，超过 baseline
- 如果最终不及 baseline，至少能确认 HAFL 的贡献方向

## 6. 风险

- 从 epoch 281 恢复，cosine schedule 从 0 开始会导致 LR 在初始几个 epoch 重新爬升（warmup），需要确认代码逻辑是否正确处理了 resume + 新 schedule 的衔接
- **建议**：恢复后手动设置 scheduler 的 last_epoch 为 0，让新的 1000 epoch cosine 从头开始衰减
