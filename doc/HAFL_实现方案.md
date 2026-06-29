# HAFL 实现方案 — 分层自适应频域损失

## 一、模块概述

- **全称**：Hierarchical Adaptive Frequency Loss with Progressive Curriculum Learning
- **作用层面**：训练损失函数（不改模型结构）
- **核心思想**：将频域分为低/中/高三个子带，分别约束，并用渐进式权重调度实现课程学习

## 二、技术原理

### 2.1 频率分解

- 对预测图和 GT 做 FFT
- 按归一化频率距离分为三个子带：低频（≤0.25）、中频（0.25~0.75）、高频（≥0.75）
- 分别 iFFT 回空域计算 L1 损失

### 2.2 渐进式权重调度（课程学习）

- `λ_low(t) = 0.8`（恒定，低频全程重要）
- `λ_mid(t) = 0.5 × min(1.0, 2t/T)`（线性增长）
- `λ_high(t) = 0.3 × min(1.0, max(0, 3(t - T/3) / T))`（延迟启动，T/3 后才开始增长）

### 2.3 总损失

```
L_total = L_char + 0.01×L_fft + 0.05×L_edge + 0.1×L_l1 + 0.1×L_HAFL
```

## 三、代码实现

### 3.1 修改文件

- `losses.py`：新增 `HierarchicalAdaptiveFreqLoss` 类
- `train.py`：集成 HAFL 损失，传入 `epoch` 和 `total_epochs`

### 3.2 核心代码（losses.py）

```python
class HierarchicalAdaptiveFreqLoss(nn.Module):
    """分层自适应频域损失 (HAFL)

    创新点3: 将图像频域分为低/中/高三个子带，分别计算L1损失，
    并通过渐进式权重调度实现课程学习（先学低频→再学高频）
    """

    def __init__(self, cutoff_low=0.25, cutoff_high=0.75):
        super(HierarchicalAdaptiveFreqLoss, self).__init__()
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high
        self.l1 = nn.L1Loss()

    def forward(self, pred, target, epoch, total_epochs):
        # 频率分解
        pred_low, pred_mid, pred_high = self.frequency_decompose(pred)
        gt_low, gt_mid, gt_high = self.frequency_decompose(target)

        # 子带损失
        loss_low = self.l1(pred_low, gt_low)
        loss_mid = self.l1(pred_mid, gt_mid)
        loss_high = self.l1(pred_high, gt_high)

        # 渐进式权重调度（课程学习）
        t = epoch / total_epochs
        w_low = 0.8  # 恒定，低频全程重要
        w_mid = 0.5 * min(1.0, 2 * t)  # 线性增长
        w_high = 0.3 * min(1.0, max(0, 3 * (t - 1 / 3)))  # 延迟启动

        return w_low * loss_low + w_mid * loss_mid + w_high * loss_high

    def frequency_decompose(self, x):
        """将图像分解为低/中/高频三个子带"""
        X_fft = torch.fft.fft2(x)
        H, W = x.shape[-2:]

        # 创建频率掩码
        freq_h = torch.fft.fftfreq(H, device=x.device)
        freq_w = torch.fft.fftfreq(W, device=x.device)
        freq_dist = torch.sqrt(freq_h[:, None] ** 2 + freq_w[None, :] ** 2)
        freq_dist = freq_dist / (freq_dist.max() + 1e-8)  # 归一化到[0,1]

        mask_low = (freq_dist <= self.cutoff_low).float()
        mask_high = (freq_dist >= self.cutoff_high).float()
        mask_mid = 1.0 - mask_low - mask_high

        # 分离三个子带
        x_low = torch.fft.ifft2(X_fft * mask_low).real
        x_mid = torch.fft.ifft2(X_fft * mask_mid).real
        x_high = torch.fft.ifft2(X_fft * mask_high).real

        return x_low, x_mid, x_high
```

## 四、实验配置

- **分支**：`local/exp2`
- **模型配置**：RFM + HAFL（1+3），不含 MFGCP
- **对比基准**：RFM（41.79 dB / 0.9906）
- **预期效果**：PSNR +0.05~0.2 dB，SSIM 改善

## 五、消融实验计划

| 实验 | 配置 | 预期 |
|------|------|------|
| A | 固定权重 FFT 损失（现有） | 41.79（基线） |
| B | 分层频域损失（无渐进） | +0.05~0.1 dB |
| C | 分层 + 渐进调度（完整 HAFL） | +0.1~0.2 dB |

## 六、参考论文

- AdaIR (ICLR, 2025)
- FADformer (ECCV, 2024)
- Focal Frequency Loss (ICCV, 2021)
- UHD-Processor (CVPR, 2025)
- ERR (CVPR, 2025)

## 七、可调超参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `cutoff_low` | 低频截止频率 | 0.25 |
| `cutoff_high` | 高频截止频率 | 0.75 |
| HAFL 权重 | 总损失中的系数 | 0.1 |
| `w_low` | 低频子带基础权重 | 0.8 |
| `w_mid` | 中频子带基础权重 | 0.5 |
| `w_high` | 高频子带基础权重 | 0.3 |
