"""
创新点3: HAFL — 分层自适应频域损失
Hierarchical Adaptive Frequency Loss with Progressive Curriculum Learning
"""
import torch
import torch.nn as nn


def frequency_decompose(x, cutoff_low=0.25, cutoff_high=0.75):
    """将特征图分解为低/中/高频三个子带

    Args:
        x: 输入特征图 (B, C, H, W)
        cutoff_low: 低频截止频率 (归一化)
        cutoff_high: 高频截止频率 (归一化)

    Returns:
        x_low, x_mid, x_high: 三个频率子带
    """
    X_fft = torch.fft.fft2(x)
    H, W = x.shape[-2:]

    # 创建频率掩码（基于归一化频率距离）
    freq_h = torch.fft.fftfreq(H, device=x.device)
    freq_w = torch.fft.fftfreq(W, device=x.device)
    freq_dist = torch.sqrt(freq_h[:, None]**2 + freq_w[None, :]**2)
    freq_dist = freq_dist / freq_dist.max()  # 归一化到[0,1]

    mask_low = (freq_dist <= cutoff_low).float()
    mask_high = (freq_dist >= cutoff_high).float()
    mask_mid = 1 - mask_low - mask_high

    # 分离三个子带
    x_low = torch.fft.ifft2(X_fft * mask_low).real
    x_mid = torch.fft.ifft2(X_fft * mask_mid).real
    x_high = torch.fft.ifft2(X_fft * mask_high).real

    return x_low, x_mid, x_high


class HierarchicalAdaptiveFreqLoss(nn.Module):
    """分层自适应频域损失 (HAFL)

    渐进式权重调度：
    - λ_low(t) = 0.8（恒定，低频全程重要）
    - λ_mid(t) = 0.5 × min(1.0, 2t / T_total)（线性增长）
    - λ_high(t) = 0.3 × min(1.0, max(0, 3(t - T_total/3) / T_total))（延迟启动）

    Args:
        cutoff_low: 低频截止频率
        cutoff_high: 高频截止频率
    """
    def __init__(self, cutoff_low=0.25, cutoff_high=0.75):
        super().__init__()
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high
        self.l1 = nn.L1Loss()

    def forward(self, pred, target, epoch, total_epochs):
        """
        Args:
            pred: 预测图像 (B, C, H, W)
            target: 目标图像 (B, C, H, W)
            epoch: 当前 epoch
            total_epochs: 总 epoch 数
        """
        # 频率分解
        pred_low, pred_mid, pred_high = frequency_decompose(pred, self.cutoff_low, self.cutoff_high)
        gt_low, gt_mid, gt_high = frequency_decompose(target, self.cutoff_low, self.cutoff_high)

        # 子带损失
        loss_low = self.l1(pred_low, gt_low)
        loss_mid = self.l1(pred_mid, gt_mid)
        loss_high = self.l1(pred_high, gt_high)

        # 渐进式权重调度
        t = epoch / total_epochs
        w_low = 0.8
        w_mid = 0.5 * min(1.0, 2 * t)
        w_high = 0.3 * min(1.0, max(0, 3 * (t - 1/3)))

        return w_low * loss_low + w_mid * loss_mid + w_high * loss_high
