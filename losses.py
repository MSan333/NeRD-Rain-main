import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x.to('cuda:0') - y.to('cuda:0')
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to('cuda:0')
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
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
        loss = self.loss(self.laplacian_kernel(x.to('cuda:0')), self.laplacian_kernel(y.to('cuda:0')))
        return loss

class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss


# ══════════════════════════════════════════════════════════════════════════
# ★ 新增损失函数: Focal Frequency Loss (与创新点 C3 频域增强配合)
# ══════════════════════════════════════════════════════════════════════════
#
# 【设计动机: 与 C3 (FreqFFN) 的协同】
# C3 在模型中引入了频域增强 FFN，使模型能够在频域空间进行处理。
# 为了更好地训练频域分支，我们引入 Focal Frequency Loss 作为辅助损失。
# 这两者形成 "模型端+损失端" 的频域增强闭环:
#   模型端 (C3 FreqFFN): 可学习频域滤波器，增强/抑制特定频率
#   损失端 (FocalFreqLoss): 自适应聚焦难恢复的频率分量，引导训练
#
# 【与现有 fftLoss 的区别】
# 原始 fftLoss: L_fft = mean(|FFT(pred) - FFT(target)|)
#   → 对所有频率分量等权重处理
#   → 低频 (整体亮度/色调) 容易恢复，但占 loss 主导
#
# FocalFrequencyLoss: L_focal = mean(weight * |FFT(pred) - FFT(target)|²)
#   → weight ∝ (误差幅度)^alpha，误差越大权重越高
#   → 自动聚焦于难恢复的频率 (通常是高频纹理/边缘细节)
#   → 在去雨任务中: 雨线遮挡的高频纹理恢复是最难的部分
#
# 【参考文献】
# Jiang et al., "Focal Frequency Loss for Image Reconstruction and Synthesis",
# ICCV 2021. 虽然论文本身是 2021 年的，但该损失函数与 C3 的频域增强
# 形成协同效果，提升频域恢复质量。
#
# 【在 train.py 中的使用】
# loss_focal = criterion_focal_freq(restored[0], target[0]) + ...
# 总损失: loss = L_char + 0.01*L_fft + 0.05*L_edge + 0.1*L_l1 + 0.05*L_focal
# 权重 0.05 为经验值，与 edge loss 相当
# ══════════════════════════════════════════════════════════════════════════

class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss (FFL) — 自适应聚焦难恢复的频率分量。

    与模型中的 C3 (FreqFFN) 形成频域增强闭环:
      - C3 (模型端): 可学习频域滤波器，在推理时增强/抑制特定频率
      - FFL (损失端): 训练时自适应聚焦难恢复的频率分量，引导 C3 学习

    计算流程:
      1. 对 pred 和 target 分别做 rfft2 (实数 FFT)
      2. 计算频域差异的幅度: |F(pred) - F(target)|
      3. 生成 Focal 权重: weight = (归一化幅度)^alpha
         → 误差越大的频率分量权重越高
      4. 加权 L2 损失: loss = mean(weight * |diff|²)

    Args:
        alpha: Focal 聚焦指数 (默认 1.0)
               alpha 越大，越聚焦于难恢复的频率分量
               alpha=0 退化为普通频域 L2 损失
    """

    def __init__(self, alpha=1.0):
        super(FocalFrequencyLoss, self).__init__()
        self.alpha = alpha  # 聚焦指数: 越大越聚焦于难频率

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) — 模型预测的恢复图像
            target: (B, C, H, W) — 干净的 ground truth 图像

        Returns:
            loss: scalar — 加权频域 L2 损失
        """
        pred = pred.to('cuda:0')
        target = target.to('cuda:0')

        # Step 1: 2D 实数 FFT (rfft2 利用实信号对称性，输出尺寸更小)
        pred_freq = torch.fft.rfft2(pred, norm='backward')
        target_freq = torch.fft.rfft2(target, norm='backward')

        # Step 2: 频域差异 (实部和虚部分别计算)
        diff_real = pred_freq.real - target_freq.real
        diff_imag = pred_freq.imag - target_freq.imag
        diff_mag = torch.sqrt(diff_real ** 2 + diff_imag ** 2 + 1e-12)  # 幅度

        # Step 3: Focal 权重 — 误差越大的频率分量权重越高
        # 归一化到 [0, 1] 后取 alpha 次方
        # 例如: alpha=1 时, 最大误差频率权重为 1, 其余按比例缩小
        weight = (diff_mag / (diff_mag.max() + 1e-12)) ** self.alpha

        # Step 4: 加权频域 L2 损失
        # weight.detach(): 权重不参与梯度计算, 仅作为采样策略
        # 效果: 网络更关注当前恢复最差的频率分量
        loss = (diff_real ** 2 + diff_imag ** 2) * weight.detach()
        return loss.mean()
