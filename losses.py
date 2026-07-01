import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        if self.kernel.device != img.device:
            self.kernel = self.kernel.to(img.device)
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x) - torch.fft.fft2(y)
        loss = torch.mean(abs(diff))
        return loss


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

    def forward(self, pred, target, epoch, total_epochs, warmup_epochs=50):
        # HAFL warmup: 前 warmup_epochs 个 epoch cosine上升，防止初期频域损失不稳定
        # 与整体 cosine LR schedule 风格一致：初期极保守，中后期加速追赶
        if epoch < warmup_epochs:
            warmup_scale = 0.5 * (1 - math.cos(math.pi * epoch / warmup_epochs))
        else:
            warmup_scale = 1.0

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

        return warmup_scale * (w_low * loss_low + w_mid * loss_mid + w_high * loss_high)

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
