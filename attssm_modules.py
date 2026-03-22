"""
==============================================================================
AttSSM Modules — 注意力增强局部性保持 Mamba (Attentive Locality-Preserving SSM)
==============================================================================

本文件实现了用于替换 NeRD-Rain bottleneck 层 TransformerBlock 的创新模块。
NeRD-Rain 原始架构的 bottleneck 使用 Transposed Attention + 门控 FFN，
本模块将其替换为具有更强全局建模能力和空间局部性保持的 Mamba SSM 架构。

参考论文 (均为 2025 CVPR 顶会):
  [1] MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration
      (CVPR 2025, Oral)
      → 核心贡献: 嵌套 S 形扫描 (Nested S-shaped Scanning, NSS)
      → 解决问题: 标准光栅扫描将相邻像素映射到远距离 1D 位置，破坏 2D 局部性
      → 我们的使用: 借鉴 S-scan 思想，实现 2 方向独立权重 stripe-based S 形扫描

  [2] MambaIRv2: Attentive State Space Restoration (CVPR 2025)
      → 核心贡献: 发现并解决 Mamba 在图像恢复中的"像素欠激活"问题
      → 解决问题: SSM 的遗忘机制导致部分像素响应不足，影响细节恢复
      → 我们的使用: 引入 SE-style 通道注意力校正 SSM 输出

  [3] EVSSM: Efficient Visual State Space Model for Image Deblurring (CVPR 2025)
      → 核心贡献: Enhanced Dual-domain FFN (EDFFN)
      → 解决问题: 纯空域 FFN 对频域信息的恢复能力有限
      → 我们的使用: 设计 FreqFFN，空域门控 + patch-wise FFT 可学习滤波并行

=====================================
三个可独立消融的创新点 (Innovation Points):
=====================================

  C1: 局部性保持 S 形 Mamba 扫描 (Locality-Preserving S-shaped Mamba Scan)
      ─ 2 方向 stripe S-scan (水平+垂直) + 独立权重 + 交替 shift
      ─ 来源: MaIR (CVPR 2025)
      ─ 核心动机: 将 2D 特征图展平为 1D 序列时保持空间局部性
      ─ v3 平衡设计: 2方向独立Mamba (expand=2, d_state=16)
        恢复 Mamba 表达能力，同时保持 2 方向 + 2C→C merge 的轻量化结构
      ─ 设计: stripe 内奇偶行/列交替翻转形成 S 形路径，
              奇数 block 偏移 scan_len//2 (类似 Swin 的 shifted window)

  C2: 通道注意力增强 + 跨尺度状态传递 (Channel Attention + Cross-Scale State)
      ─ 来源: MambaIRv2 (CVPR 2025) + 我们的原创设计
      ─ 核心动机:
        (a) 解决像素欠激活: SE 通道注意力重新校准各通道响应权重
        (b) 跨尺度状态传递: NeRD-Rain 有 small/mid/max 三个尺度的 bottleneck，
            将 small 尺度的 SSM 聚合状态传递给 mid，mid 传递给 max，
            实现多尺度信息的隐式流动 (此设计为我们的原创贡献，
            利用了 SSM 天然的状态传递特性 + NeRD-Rain 的多尺度架构)

  C3: 频域增强 FFN (Frequency-Enhanced Feed-Forward Network)
      ─ 来源: EVSSM (CVPR 2025)
      ─ 核心动机: 雨线去除需要恢复被雨线遮挡的高频纹理，
                  纯空域 FFN 难以精确恢复频域细节
      ─ 设计: 双路并行结构:
        空域分支: 保持原始 NeRD-Rain 的门控 FFN 结构 (1×1 → DWConv → GeLU gate → 1×1)
        频域分支: patch-wise rfft2 → 可学习频域滤波器 → irfft2
        融合: x_spatial + gamma * x_freq (gamma 初始化为 0.1, 频域分支一开始即有微弱贡献)
"""

import numbers
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import lru_cache
from mamba_ssm import Mamba


# ============================================================================
# 基础工具层 (Utility Layers)
# ============================================================================
# 以下 LayerNorm 实现与原始 NeRD-Rain model.py 中的保持一致，
# 但独立定义以避免循环 import。支持 (B, C, H, W) 格式的输入。

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    """适用于 (B, C, H, W) 输入的 LayerNorm"""
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return rearrange(
            self.body(rearrange(x, 'b c h w -> b (h w) c')),
            'b (h w) c -> b c h w', h=h, w=w
        )


# ============================================================================
# 创新点 C1: 局部性保持 S 形扫描 (Locality-Preserving Stripe S-Scan)
# ============================================================================
#
# 参考: MaIR (CVPR 2025, Oral) 的 Nested S-shaped Scanning (NSS)
#
# 【问题背景】
# 将 2D 特征图展平为 1D 序列是 Mamba 处理图像的必要步骤。
# 但标准的行优先 (raster) 扫描会在行末→下一行首之间产生远距离跳跃，
# 破坏了 2D 空间的局部性。例如 32×32 特征图中，
# (0,31) 和 (1,0) 在 2D 中相距 31 像素，但在 1D 中相邻，
# 而 (0,31) 和 (0,30) 在 2D 中相邻，但在 1D 中也相邻——
# 但 (0,31) 和 (1,31) 在 2D 中相邻却在 1D 中相距 32。
# 这种局部性破坏会导致 SSM 的状态传递效率低下。
#
# 【解决方案: S 形扫描】
# 将特征图按 stripe 宽度分成若干竖条 (stripe)，
# 条内奇数行左右翻转，奇数 stripe 上下翻转，
# 形成连续的 S 形路径 (蛇形走位)。
# 这样在展平后，2D 中相邻的像素在 1D 中也倾向于相邻。
#
# 【2 方向扫描】
# 单方向 S-scan 仍然偏向某个方向。
# 使用 2 个方向 (水平S + 垂直S) 由各自独立的 Mamba 处理后合并，
# 去掉反向扫描 — 反向信息由交替 shift 机制覆盖。
# v3: 各方向独立权重 (expand=2, d_state=16), 保持 2C→C merge。
#
# 【交替 shift (Alternating Shift)】
# 类似 Swin Transformer 的 shifted window 策略，
# 奇数 block 的 stripe 偏移 scan_len//2，
# 使得相邻 block 的扫描 "边界" 不对齐，
# 从而弥补 stripe 边界处的信息断裂。
# ============================================================================

def _generate_s_scan_indices(H, W, scan_len, shift=0):
    """
    生成单方向 S 形扫描的索引映射 (C1 核心实现)。

    【算法原理】 (改编自 MaIR CVPR 2025 的 NSS):
      Step 1: 将 2D 特征图 (H, W) 按 stripe 宽度 scan_len 分成若干竖条
      Step 2: 奇数 stripe 整体上下翻转 (flip rows)
      Step 3: 每个 stripe 内，奇数行左右翻转 (flip columns)
      最终形成连续的 S 形路径:

      偶数stripe:  奇数stripe:     展平后的效果:
      → → → → →   ← ← ← ← ←      → → → → → ↓
      ← ← ← ← ←   → → → → →      ← ← ← ← ← ↓
      → → → → →   ← ← ← ← ←      → → → → → ...
                                     (S形, 保持2D局部性)

    【shift 参数的作用】:
      shift > 0 时，对列维度做循环移位 (torch.roll)，
      使 stripe 边界发生偏移。这类似 Swin Transformer 中
      shifted window 的思想，让相邻 block 的边界不对齐，
      弥补 stripe 边界处的信息断裂。

    Args:
        H, W: 特征图高宽
        scan_len: stripe 宽度 (自适应: 通常为 max(H//4, 2))
        shift: stripe 偏移量 (奇数block = scan_len//2, 偶数block = 0)

    Returns:
        forward_idx: (H*W,) 正向扫描索引 (2D→1D 的映射)
        inverse_idx: (H*W,) 逆扫描索引 (1D→2D 的还原映射)
    """
    # 创建 2D 坐标网格
    coords = torch.arange(H * W).reshape(H, W)

    # 应用 shift (循环移位)
    if shift > 0:
        coords = torch.roll(coords, shifts=shift, dims=1)

    # S 形扫描: 奇数 stripe 翻转
    num_stripes = math.ceil(W / scan_len)
    for i in range(num_stripes):
        start = i * scan_len
        end = min((i + 1) * scan_len, W)
        if i % 2 == 1:
            # 奇数 stripe: 上下翻转
            coords[:, start:end] = coords[:, start:end].flip(0)
        # 条内奇数行: 左右翻转
        for row in range(H):
            if row % 2 == 1:
                coords[row, start:end] = coords[row, start:end].flip(0)

    forward_idx = coords.reshape(-1)
    # 逆索引: 还原映射
    inverse_idx = torch.argsort(forward_idx)
    return forward_idx, inverse_idx


def get_scan_indices(H, W, scan_len, shift=0, device='cuda'):
    """
    生成 2 方向扫描索引 (C1 的轻量化多方向扫描实现)。

    【2 方向设计动机】:
      单方向 S-scan 虽然保持了局部性，但存在方向偏差:
      水平 S-scan 更擅长捕获水平方向的依赖，对垂直依赖较弱。
      使用水平 + 垂直 2 个互补方向，由各自独立的 Mamba SSM 处理:

      方向 0: 水平 S-scan → mamba_h — 从左上到右下的 S 形路径
      方向 1: 垂直 S-scan → mamba_v — 先转置(H,W)→(W,H)，再水平 S-scan

      去掉反向扫描 (水平反向、垂直反向) — 反向信息由 Mamba 的
      双向感知能力和交替 shift 机制覆盖。
      2 方向的输出通过 merge_proj (2C→C) 合并。

    v3 平衡设计: 保持 2 方向 + 2C→C merge，但各方向用独立 Mamba (expand=2, d_state=16),
    在 v1 (59M) 和 v2 (28M) 之间取得平衡 (~42M)。

    Returns:
        forward_indices: (2, H*W) — 2 方向的正向扫描索引
        inverse_indices: (2, H*W) — 2 方向的逆扫描索引 (还原空间顺序)
    """
    # 水平方向 S-scan
    fwd_h, inv_h = _generate_s_scan_indices(H, W, scan_len, shift)

    # 垂直方向 S-scan (先转置)
    fwd_v, inv_v = _generate_s_scan_indices(W, H, scan_len, shift)
    # 需要一个转置映射: (H,W) -> (W,H)
    transpose_map = torch.arange(H * W).reshape(H, W).T.reshape(-1)

    # 垂直正向: 先转置 → S-scan → 逆 S-scan → 逆转置
    fwd_v_full = transpose_map[fwd_v]
    inv_v_full = torch.argsort(fwd_v_full)

    forward_indices = torch.stack([fwd_h, fwd_v_full]).to(device)     # (2, H*W)
    inverse_indices = torch.stack([inv_h, inv_v_full]).to(device)      # (2, H*W)

    return forward_indices, inverse_indices


# 全局缓存: 避免训练时每个 iteration 重复生成扫描索引
# Key: (H, W, scan_len, shift, device), Value: (forward_indices, inverse_indices)
# 因为索引只依赖于特征图尺寸和扫描参数，同一尺寸的输入可复用
_scan_cache = {}

def cached_scan_indices(H, W, scan_len, shift, device):
    """带缓存的扫描索引获取。训练中同一 batch 的特征图尺寸固定，缓存可避免重复计算。"""
    key = (H, W, scan_len, shift, device)
    if key not in _scan_cache:
        _scan_cache[key] = get_scan_indices(H, W, scan_len, shift, device)
    return _scan_cache[key]


# ============================================================================
# 创新点 C2 (Part 1): 通道注意力校正 (Channel Attention Recalibration)
# ============================================================================
#
# 参考: MambaIRv2 (CVPR 2025) — Attentive State Space Restoration
#
# 【问题背景: 像素欠激活 (Pixel Under-Activation)】
# MambaIRv2 论文发现: 当 Mamba SSM 处理长序列时，由于其遗忘机制
# (forget gate)，早期输入的信息会随着序列推进被逐渐遗忘，
# 导致部分空间位置的像素响应不足 (under-activated)。
# 在图像恢复任务中，这意味着某些区域的细节无法被充分恢复。
#
# 【解决方案: SE-style 通道注意力】
# 在 Mamba SSM 输出后接一个轻量级的 Squeeze-and-Excitation (SE) 模块:
#   1. Squeeze: 全局平均池化将 (B,C,H,W) → (B,C,1,1)
#   2. Excitation: FC(C→C/r) → ReLU → FC(C/r→C) → Sigmoid
#   3. Scale: 用生成的通道权重重新校准 SSM 输出
# 这样可以根据全局统计信息重新分配各通道的响应强度，
# 补偿 SSM 遗忘机制造成的信息损失。
#
# 参数量极少 (reduction=16, 仅 ~2×(C×C/16) = ~4.6K for C=192)，
# 但对恢复质量有显著提升 (消融实验中通常提升 0.1-0.3 dB PSNR)。
# ============================================================================

class ChannelAttention(nn.Module):
    """
    SE-style 通道注意力 (Squeeze-and-Excitation)。

    用于校正 Mamba SSM 输出中各通道的响应权重,
    解决 SSM 遗忘机制导致的像素欠激活问题。

    结构: AdaptiveAvgPool2d(1) → Linear(C→C//r) → ReLU → Linear(C//r→C) → Sigmoid
    参数量: ~2 × C × C/reduction ≈ 4608 (C=192, r=16)
    """
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, max(dim // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(dim // reduction, 4), dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """x: (B, C, H, W)"""
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


# ============================================================================
# LocalityMamba: 融合 C1 + C2 的核心模块
# ============================================================================
#
# 这是本工作最核心的模块，融合了创新点 C1 (局部性保持扫描) 和
# C2 (通道注意力校正 + 跨尺度状态传递)。
#
# 【与原始 TransformerBlock.Attention 的对比】
# 原始:  x → QKV投影 → DWConv → Transposed Attention (O(N²)) → 投影
# 本模块: x → 投影 → 2方向S-scan → 独立Mamba SSM (O(N)) → 逆扫描 → 合并 → CA
#
# 优势:
#   1. 复杂度: O(N) vs O(N²)，对大尺寸特征图（如 128×128）优势明显
#   2. 全局感受野: Mamba SSM 可以看到整个序列（无窗口截断）
#   3. 空间局部性: S 形扫描保持了 2D 空间的邻近关系
#   4. 状态传递: SSM 的隐状态可以在尺度间传递（原创贡献）
#
# 【版本演进】
# v1: 4 方向 × 独立 Mamba(expand=2, d_state=16) → 59M, 参数过多导致欠拟合
# v2: 2 方向 × 共享 Mamba(expand=1, d_state=8)  → 28M, 矫枉过正, PSNR 差 2.83dB
# v3: 2 方向 × 独立 Mamba(expand=2, d_state=16) → ~42M, 在 v1/v2 间取最佳平衡
#     保留 v2 的 "2方向 + 2C→C merge" 轻量化设计,
#     恢复 Mamba 本身的表达能力 (expand=2, d_state=16, 独立权重)。
# ============================================================================

class LocalityMamba(nn.Module):
    """
    局部性保持 Mamba 模块 (融合 C1 S-scan + C2 通道注意力)。

    v3 改动 (在 v1 过重和 v2 过轻之间取平衡):
      - 保留 v2 的 2 方向 + 2C→C merge 轻量化结构
      - 恢复独立权重: 2 方向各自独立的 Mamba (避免梯度冲突)
      - 恢复 expand=2: Mamba 内部通道扩展，恢复选择性机制的操作空间
      - 恢复 d_state=16: SSM 隐状态维度，增强长序列建模能力
      - 保留 v2 的跨尺度状态修复: 0.01 缩放 + LayerNorm

    完整流程:
      1. [可选] 跨尺度状态注入: prev_state → LayerNorm → (B,C,1,1) × 0.01 加到输入
      2. 输入投影: Conv2d 1×1 (维持通道数)
      3. 2 方向 S 形扫描: 将 (B,C,H,W) 按水平/垂直 S-scan 展平为 (B,L,C)
      4. 独立 Mamba SSM: mamba_h 处理水平, mamba_v 处理垂直 (d_state=16, expand=2)
      5. 逆扫描还原: 从 1D 还原为 2D 空间顺序
      6. 合并投影: 2C → C (Conv2d 1×1)
      7. 通道注意力: SE-style 校正 (解决像素欠激活)
      8. 输出状态: 全局平均池化 → (B,C) 作为跨尺度状态

    Args:
        dim: 输入/输出通道数 (= 192 for NeRD-Rain bottleneck)
        d_state: Mamba SSM 隐状态维度 (默认 16)
        d_conv: Mamba 内部 1D 深度卷积核宽度 (默认 4)
        expand: Mamba 内部通道扩展倍数 (默认 2)
        scan_len: S 形扫描的 stripe 宽度 (None 则自适应为 max(H//4, 2))
        block_idx: 当前 block 索引 (偶数 block: shift=0, 奇数 block: shift=scan_len//2)
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2,
                 scan_len=None, block_idx=0):
        super().__init__()
        self.dim = dim
        self.scan_len = scan_len
        self.block_idx = block_idx
        self.n_dirs = 2  # 2 方向扫描 (水平S + 垂直S)

        # 输入投影
        self.in_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # v3: 2 个独立 Mamba SSM (各方向独立权重, 避免梯度冲突)
        self.mamba_h = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_v = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # 合并投影: 2C → C (原 4C→C)
        self.merge_proj = nn.Conv2d(dim * self.n_dirs, dim, kernel_size=1, bias=False)

        # 通道注意力 (C2: 解决像素欠激活)
        self.channel_attn = ChannelAttention(dim, reduction=16)

        # 跨尺度状态注入的 LayerNorm (稳定状态信号)
        self.state_norm = nn.LayerNorm(dim)

    def forward(self, x, prev_state=None):
        """
        Args:
            x: (B, C, H, W) — bottleneck 特征图 (C=192 for NeRD-Rain)
            prev_state: (B, C) 来自上一尺度的 SSM 聚合状态 (可选)
                        small→mid 或 mid→max 时传入

        Returns:
            out: (B, C, H, W) — 处理后的特征图
            state: (B, C) — 当前尺度的聚合状态 (传递给下一尺度)
        """
        B, C, H, W = x.shape
        L = H * W

        # ───────────────────────────────────────────────────────────────
        # C2 原创贡献: 跨尺度状态注入 (Cross-Scale State Injection)
        # ───────────────────────────────────────────────────────────────
        # v2 修复: 缩放系数从 0.1 → 0.01, 并加 LayerNorm 稳定信号
        # 避免上一尺度的状态噪声过强主导当前尺度的特征
        if prev_state is not None:
            state_normed = self.state_norm(prev_state)  # (B, C) LayerNorm 稳定
            state_bias = state_normed.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            x = x + state_bias * 0.01  # v2: 更小的缩放系数

        # 输入投影 (1×1 Conv, 不改变通道数)
        x_proj = self.in_proj(x)

        # ───────────────────────────────────────────────────────────────
        # C1: 确定 S-scan 参数
        # ───────────────────────────────────────────────────────────────
        scan_len = self.scan_len if self.scan_len is not None else max(H // 4, 2)
        shift = (scan_len // 2) if (self.block_idx % 2 == 1) else 0

        # 获取 2 方向扫描索引 (带缓存)
        fwd_indices, inv_indices = cached_scan_indices(H, W, scan_len, shift, x.device)

        # 展平为 (B, C, L), L = H*W
        x_flat = x_proj.reshape(B, C, L)

        # ───────────────────────────────────────────────────────────────
        # C1: 2 方向 S-scan + 独立 Mamba SSM + 逆扫描
        # ───────────────────────────────────────────────────────────────
        # v3: 各方向使用独立的 Mamba (避免共享权重的梯度冲突)
        # 方向 0: 水平 S-scan → self.mamba_h
        # 方向 1: 垂直 S-scan → self.mamba_v
        mamba_list = [self.mamba_h, self.mamba_v]
        outputs = []
        for d in range(self.n_dirs):
            # S 形扫描: 按索引重排像素顺序
            x_scanned = torch.index_select(x_flat, 2, fwd_indices[d])  # (B, C, L)
            x_seq = x_scanned.permute(0, 2, 1)  # (B, L, C)

            # 独立 Mamba SSM: 各方向使用独立权重建模全局依赖
            y_seq = mamba_list[d](x_seq)  # (B, L, C) — 各方向独立 Mamba

            # 逆扫描: 从 S 形序列还原为原始空间顺序
            y_flat = y_seq.permute(0, 2, 1)  # (B, C, L)
            y_restored = torch.index_select(y_flat, 2, inv_indices[d])  # (B, C, L)
            outputs.append(y_restored.reshape(B, C, H, W))

        # 合并 2 方向: (B, 2C, H, W) → (B, C, H, W)
        merged = torch.cat(outputs, dim=1)  # (B, 2C, H, W)
        out = self.merge_proj(merged)  # 1×1 Conv: 2C → C

        # ───────────────────────────────────────────────────────────────
        # C2: 通道注意力校正 (Channel Attention Recalibration)
        # ───────────────────────────────────────────────────────────────
        out = self.channel_attn(out)

        # 提取当前尺度的聚合状态 (用于跨尺度传递)
        state = out.mean(dim=[2, 3])  # (B, C)

        return out, state


# ============================================================================
# 创新点 C3: 频域增强 FFN (Frequency-Enhanced Feed-Forward Network)
# ============================================================================
#
# 参考: EVSSM (CVPR 2025) 的 Enhanced Dual-domain FFN (EDFFN)
#
# 【问题背景: 空域 FFN 的频域盲区】
# 原始 NeRD-Rain 的 FFN (FeedForward) 采用纯空域处理:
#   project_in(1×1) → DWConv(3×3) → GeLU gating → project_out(1×1)
# 这种结构对空域特征变换很有效，但对频域信息的恢复能力有限。
# 在去雨任务中，雨线是具有特定频率特征的退化:
#   - 雨线本身: 高频成分 (锐利边缘)
#   - 被遮挡的背景纹理: 需要恢复的高频细节
# 仅靠空域处理难以精确定位和恢复这些频率分量。
#
# 【解决方案: 双路并行 FFN】
# 在原始空域 FFN 基础上，并行添加一个频域分支:
#
#         ┌─── 空域分支 ────────────────────────┐
#   x ──→│ 1×1投影 → DWConv → GeLU gate → 1×1投影 │──→ x_spatial
#         └──────────────────────────────────────┘
#         ┌─── 频域分支 ────────────────────────┐
#   x ──→│ pad → patch切分 → rfft2 → 可学习滤波 │──→ x_freq
#         │ → irfft2 → 还原 → 1×1投影           │
#         └──────────────────────────────────────┘
#
#   output = x_spatial + gamma * x_freq
#
# 【关键设计细节】
# 1. patch-wise FFT: 不对整个特征图做 FFT (太大, 频谱稀疏),
#    而是切成 8×8 小 patch 分别做 FFT (频谱更紧凑, 更易学习)
# 2. 可学习频域滤波器: shape=(dim, 1, 1, 8, 5), 其中 5=8//2+1 (rfft2 输出)
#    初始化为 0.5, 训练中学习对各频率分量的增益/衰减
# 3. gamma 融合: gamma 初始化为 0.1 (v2 从 0 提高), 让频域分支一开始
#    就有微弱贡献, 避免前期完全不参与、后期突然参与导致训练震荡。
# ============================================================================

class FreqFFN(nn.Module):
    """
    频域增强前馈网络 (C3 核心实现)。

    双路并行结构:
      空域分支: 与原始 NeRD-Rain FeedForward 相同的门控结构
      频域分支: patch-wise rfft2 → 可学习频域滤波器 → irfft2 → 1×1投影
      融合方式: x_spatial + gamma * x_freq (gamma 可学习, 初始为 0.1)

    Args:
        dim: 输入/输出通道数 (= 192 for NeRD-Rain bottleneck)
        ffn_expansion_factor: 隐层扩展倍数 (默认 2.66, 与原始一致)
        bias: 是否使用偏置 (默认 False, 与原始一致)
        freq_patch_size: 频域处理的 patch 大小 (默认 8, 即 8×8 的小块)
    """
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, freq_patch_size=8):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.freq_patch_size = freq_patch_size

        # ═══════ 空域分支 ═══════
        # 与原始 NeRD-Rain 的 FeedForward 完全一致: 门控结构
        # project_in: C → 2×hidden (两路)
        # dwconv: 深度可分离卷积 (3×3, groups=2×hidden)
        # GeLU gating: x1 = GeLU(branch1) * branch2 (门控机制)
        # project_out: hidden → C
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # ═══════ 频域分支 ═══════
        # 可学习频域滤波器: 对每个通道独立的频率响应
        # shape: (dim, 1, 1, ps, ps//2+1)
        #   - dim: 每个通道独立的滤波器
        #   - 1, 1: 对所有 patch 位置共享 (参数高效)
        #   - ps: patch 高度方向的频率数 (= freq_patch_size)
        #   - ps//2+1: rfft2 在宽度方向的频率数 (实对称性)
        # 初始化为 0.5: 不完全通过也不完全阻断, 给训练自由度
        self.freq_filter = nn.Parameter(
            torch.ones(dim, 1, 1, freq_patch_size, freq_patch_size // 2 + 1) * 0.5
        )
        # 频域输出投影: 1×1 Conv 将频域分支输出映射回特征空间
        self.freq_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # ═══════ 融合 ═══════
        # gamma: 可学习标量, 控制频域分支的参与程度
        # v2 修复: 初始化从 0 → 0.1
        # 原因: gamma=0 导致频域分支前期完全不参与, 后期突然参与可能震荡
        # 初始化为 0.1 让频域分支一开始就有微弱贡献, 训练更平滑
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def _pad_to_patch(self, x):
        """将输入 pad 到 freq_patch_size 的整数倍"""
        _, _, H, W = x.shape
        ps = self.freq_patch_size
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, H, W

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) — LayerNorm 后的特征图

        Returns:
            output: (B, C, H, W) — 空域 + 频域融合后的特征图
        """
        # =============================================
        # 空域分支 (与原始 NeRD-Rain FeedForward 完全一致)
        # =============================================
        # 门控结构: GeLU(branch1) * branch2
        # 这种门控设计来自 Restormer (CVPR 2022)
        x_spatial = self.project_in(x)          # (B, 2×hidden, H, W)
        x1, x2 = self.dwconv(x_spatial).chunk(2, dim=1)  # 各 (B, hidden, H, W)
        x_spatial = F.gelu(x1) * x2              # 门控: GeLU 激活 × 信息流
        x_spatial = self.project_out(x_spatial)  # (B, C, H, W)

        # =============================================
        # 频域分支 (C3 核心: patch-wise FFT + 可学习滤波)
        # =============================================
        # Step 1: 将特征图 pad 到 patch_size 的整数倍 (reflect padding)
        x_padded, orig_H, orig_W = self._pad_to_patch(x)
        B, C, Hp, Wp = x_padded.shape
        ps = self.freq_patch_size  # 默认 8

        # Step 2: 切分为 (nH × nW) 个 (ps × ps) 的小 patch
        # reshape: (B, C, Hp//ps, ps, Wp//ps, ps) → permute → (B, C, nH, nW, ps, ps)
        x_patch = x_padded.reshape(B, C, Hp // ps, ps, Wp // ps, ps)
        x_patch = x_patch.permute(0, 1, 2, 4, 3, 5)  # (B, C, nH, nW, ps, ps)

        # Step 3: 对每个 patch 独立做 2D 实数 FFT
        # rfft2 输出: (B, C, nH, nW, ps, ps//2+1) — 利用实信号的共轭对称性
        # norm='ortho': 正交归一化, 保证正逆变换的一致性
        x_freq = torch.fft.rfft2(x_patch.float(), norm='ortho')

        # Step 4: 可学习频域滤波
        # freq_filter shape: (C, 1, 1, ps, ps//2+1) 与 x_freq 广播相乘
        # 效果: 对每个通道的各频率分量施加不同的增益/衰减
        # 例如: 如果某频率的滤波器值 > 0.5, 则增强该频率; < 0.5 则抑制
        x_freq = x_freq * self.freq_filter

        # Step 5: 逆 FFT 还原回空域
        x_patch_out = torch.fft.irfft2(x_freq, s=(ps, ps), norm='ortho').to(x.dtype)

        # Step 6: 从 patch 还原为完整特征图
        x_freq_out = x_patch_out.permute(0, 1, 2, 4, 3, 5)  # (B, C, nH, ps, nW, ps)
        x_freq_out = x_freq_out.reshape(B, C, Hp, Wp)

        # Step 7: 去除 padding, 恢复原始尺寸
        x_freq_out = x_freq_out[:, :, :orig_H, :orig_W]
        x_freq_out = self.freq_proj(x_freq_out)  # 1×1 Conv 投影

        # =============================================
        # 融合: 空域 + gamma × 频域
        # =============================================
        # gamma 初始 = 0.1 (v2), 频域分支从训练开始即有微弱贡献
        return x_spatial + self.gamma * x_freq_out


# ============================================================================
# AttSSMBlock: 完整的残差块 (替换 TransformerBlock)
# ============================================================================
#
# 【与原始 TransformerBlock 的对应关系】
#
# 原始 TransformerBlock:
#   x → LayerNorm → Attention (Transposed, O(N²)) → +residual
#     → LayerNorm → FeedForward (空域门控)          → +residual
#
# 新 AttSSMBlock (v3 平衡设计):
#   x → LayerNorm → LocalityMamba (2方向独立Mamba + CA, O(N)) → +residual
#     → LayerNorm → FreqFFN (空域门控 + 频域滤波, gamma=0.1)   → +residual
#
# 三个创新点在此完整融合:
#   C1: LocalityMamba 中的 S 形扫描
#   C2: LocalityMamba 中的通道注意力 + 跨尺度状态传递
#   C3: FreqFFN 中的频域增强
#
# 接口差异:
#   TransformerBlock.forward(x) → x           (无状态)
#   AttSSMBlock.forward(x, prev_state) → x, state  (有状态)
# ============================================================================

class AttSSMBlock(nn.Module):
    """
    注意力增强局部性保持 SSM 残差块 (AttSSM Block)。

    替换原始 TransformerBlock, 融合 C1+C2+C3 三个创新点:
      x → LayerNorm → LocalityMamba(C1+C2) → +residual
        → LayerNorm → FreqFFN(C3)           → +residual

    Args:
        dim: 通道数 (= 192 for NeRD-Rain bottleneck)
        ffn_expansion_factor: FFN 扩展倍数 (默认 2.66)
        bias: 偏置 (默认 False)
        LayerNorm_type: LayerNorm 类型 ('WithBias' 或 'BiasFree')
        d_state: Mamba 隐状态维度 (默认 16)
        d_conv: Mamba 卷积核宽度 (默认 4)
        expand: Mamba 扩展倍数 (默认 2)
        scan_len: S 形扫描 stripe 宽度 (None 则自适应)
        block_idx: 块索引 (偶数: 无shift, 奇数: 有shift)
        freq_patch_size: 频域 FFN 的 patch 大小 (默认 8)
    """
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type='WithBias', d_state=16, d_conv=4,
                 expand=2, scan_len=None, block_idx=0,
                 freq_patch_size=8):
        super().__init__()

        self.norm1 = LayerNorm2d(dim, LayerNorm_type)
        self.locality_mamba = LocalityMamba(
            dim=dim, d_state=d_state, d_conv=d_conv,
            expand=expand, scan_len=scan_len, block_idx=block_idx
        )

        self.norm2 = LayerNorm2d(dim, LayerNorm_type)
        self.freq_ffn = FreqFFN(
            dim=dim, ffn_expansion_factor=ffn_expansion_factor,
            bias=bias, freq_patch_size=freq_patch_size
        )

    def forward(self, x, prev_state=None):
        """
        Args:
            x: (B, C, H, W)
            prev_state: (B, C) 来自上一尺度的状态 (可选)

        Returns:
            out: (B, C, H, W)
            state: (B, C) 当前块的状态
        """
        # SSM 分支 + 残差
        ssm_out, state = self.locality_mamba(self.norm1(x), prev_state)
        x = x + ssm_out

        # 频域增强 FFN + 残差
        x = x + self.freq_ffn(self.norm2(x))

        return x, state


# ============================================================================
# AttSSMSequential: 多个 AttSSMBlock 的序列容器 (替换 nn.Sequential)
# ============================================================================
#
# 【设计动机】
# 原始 NeRD-Rain 的 latent 层使用 nn.Sequential:
#   self.latent_small = nn.Sequential(*[TransformerBlock(...) for _ in range(3)])
#   latent_small = self.latent_small(x)  # 无状态
#
# AttSSMBlock 引入了 prev_state 参数和 state 返回值，
# nn.Sequential 无法处理这种多输入多输出的接口。
# 因此设计 AttSSMSequential 作为替代容器:
#   self.latent_small = AttSSMSequential(dim=192, num_blocks=3)
#   latent_small, state_small = self.latent_small(x, prev_state=None)
#
# 【跨尺度状态传递的具体流程】 (在 model.py forward() 中使用)
#   small_out, state_s = self.latent_small(x_small, prev_state=None)
#   mid_out, state_m   = self.latent_mid1(x_mid, prev_state=Linear(state_s))
#   max_out, _         = self.latent_max1(x_max, prev_state=Linear(state_m))
#
# 状态只传给容器内第一个 block, 后续 block 不再接收跨尺度状态,
# 避免过度依赖上一尺度的信息。
# ============================================================================

class AttSSMSequential(nn.Module):
    """
    AttSSMBlock 的序列容器 (替换 nn.Sequential(*[TransformerBlock...]))。

    功能:
      1. 串联多个 AttSSMBlock, 自动设置交替 shift (block_idx)
      2. 接收跨尺度状态 prev_state (仅传给第一个 block)
      3. 输出最后一个 block 的 state (用于传递给下一尺度)

    使用方式 (在 model.py 中):
      # 替换前:
      self.latent_small = nn.Sequential(*[TransformerBlock(...) for _ in range(3)])
      latent_small = self.latent_small(x)

      # 替换后:
      self.latent_small = AttSSMSequential(dim=192, num_blocks=3)
      latent_small, state_small = self.latent_small(x, prev_state=None)
    """
    def __init__(self, dim, num_blocks=3, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias',
                 d_state=16, d_conv=4, expand=2, scan_len=None,
                 freq_patch_size=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            AttSSMBlock(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                scan_len=scan_len,
                block_idx=i,
                freq_patch_size=freq_patch_size,
            )
            for i in range(num_blocks)
        ])

    def forward(self, x, prev_state=None):
        """
        Args:
            x: (B, C, H, W)
            prev_state: (B, C) 来自上一尺度 (仅传递给第一个块)

        Returns:
            x: (B, C, H, W)
            state: (B, C) 最后一个块输出的状态
        """
        state = None
        for i, block in enumerate(self.blocks):
            # 只有第一个块接收跨尺度状态
            ps = prev_state if i == 0 else None
            x, state = block(x, prev_state=ps)
        return x, state
