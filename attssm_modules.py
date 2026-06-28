"""
==============================================================================
NeRD-Rain v5 — 频域门控 FFN + 多维通道注意力
==============================================================================

本文件实现两个创新模块, 用于增强 NeRD-Rain bottleneck 层:
  C1: FreqFFN  — 频域门控双域前馈网络
  C2: MDCA     — 多维通道注意力

==============================================================================
C1: 频域门控双域 FFN (Frequency-Gated Dual-domain FFN)
==============================================================================

参考论文:
  [1] EVSSM: Efficient Visual State Space Model for Image Deblurring (CVPR 2025)
      → 核心贡献: Enhanced Dual-domain FFN (EDFFN), 空域+频域并行 + 频率门控
  [2] LoFormer: Local Frequency Transformer for Image Restoration (CVPR 2024)
      → 核心贡献: Dynamic Frequency FFN (DyFN), 动态频率门控融合
  [3] Efficient Image Restoration with Frequency-Aware Transformer (2025)
      → 核心贡献: FAFN, 多尺度 DWConv + FFT + FGM
      → 消融: Baseline 38.26 → +MS-DWConv 38.41 → +FFT 38.61 → +FGM 38.67

动机:
  雨线是具有特定频率特征的退化 (高频锐利边缘), 纯空域 FFN 难以精确
  恢复被雨线遮挡的频域细节。CVPR 2024/2025 多篇论文证明双域 FFN 有效。

v4 → v5 升级:
  旧版: x_spatial + gamma * x_freq (标量加权, 无法空间自适应)
  新版: x_spatial + FGM(x_spatial) * x_freq (频率门控, 空域自适应)
        空域分支增加多尺度 DWConv (3×3, 5×5, 7×7)

架构示意:
        ┌─── 空域分支 ─────────────────────────────────────┐
  x ──→│ 1×1 → [DW3×3 + DW5×5 + DW7×7] → GeLU gate → 1×1 │──→ x_spatial
        └────────────────────────────────────────────────────┘
        ┌─── 频域分支 ─────────────────────────────────┐
  x ──→│ pad → patch → rfft2 → 可学习滤波 → irfft2 → 1×1 │──→ x_freq
        └──────────────────────────────────────────────┘
        ┌─── 频率门控 (FGM) ───────┐
  x ──→│ DW3×3 → 1×1 → Sigmoid    │──→ gate ∈ [0,1]
        └───────────────────────────┘

  output = x_spatial + gate * x_freq

==============================================================================
C2: 多维通道注意力 (Multi-Dimensional Channel Attention, MDCA)
==============================================================================

参考论文:
  [1] Efficient Image Restoration with Frequency-Aware Transformer (2025)
      → 核心贡献: MDCA, 在 Transposed Attention 基础上添加宽度/高度/通道三维注意力
      → 消融: Basic CA 33.18 → +W 33.39 → +H 33.49 → +CH 33.67 (+0.49 dB)
      → 参数增加仅 0.3% (8.62M → 8.64M)

动机:
  原始 Transposed Attention (MDTA) 计算通道间注意力矩阵 (C/h × C/h),
  强于通道建模但缺少对空间维度 (H, W) 的显式建模。
  MDCA 以几乎零开销添加宽度/高度/通道三维注意力, 增强特征表达。

设计:
  MDCA(x) = A_c ⊗ A_w ⊗ A_h ⊗ A_ch
  其中:
    A_c  = 原始 Transposed Attention 输出 (保持不变)
    A_w  = σ(Conv1×1(AvgPool_h(A_c)))   — 宽度维注意力
    A_h  = σ(Conv1×1(AvgPool_w(A_c)))   — 高度维注意力
    A_ch = σ(Conv1×1(AvgPool_hw(A_c)))  — 通道维注意力
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# C1: 频域门控双域 FFN (Frequency-Gated Dual-domain FFN)
# ============================================================================

class FreqFFN(nn.Module):
    """
    频域门控双域前馈网络 (C1 核心实现)。

    三个组件:
      空域分支: 多尺度 DWConv (3×3,5×5,7×7) + GeLU 门控
      频域分支: patch-wise rfft2 → 可学习频域滤波器 → irfft2 → 1×1 投影
      频率门控 (FGM): 空域特征生成 [0,1] 门控图, 自适应控制频域分支贡献

    参考:
      EVSSM (CVPR 2025) — EDFFN: 空域+频域并行 + 频率门控
      LoFormer (CVPR 2024) — DyFN: 动态频率 FFN + 门控融合
      FAFN (2025) — 多尺度 DWConv + FFT + FGM

    Args:
        dim: 输入/输出通道数 (= 192 for NeRD-Rain bottleneck)
        ffn_expansion_factor: 隐层扩展倍数 (默认 2.66, 与原始一致)
        bias: 是否使用偏置 (默认 False, 与原始一致)
        freq_patch_size: 频域处理的 patch 大小 (默认 8)
    """
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, freq_patch_size=8):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.freq_patch_size = freq_patch_size

        # ═══════ 空域分支: 多尺度 DWConv + GeLU 门控 ═══════
        # project_in: C → 2×hidden (两路, 用于门控)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # 多尺度深度可分离卷积 (参考 FAFN 的 MS-DWConv 设计)
        # 3×3: 捕获局部纹理细节
        # 5×5: 捕获中等尺度结构
        # 7×7: 捕获较大范围上下文
        self.dwconv3 = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias
        )
        self.dwconv5 = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=5, stride=1, padding=2,
            groups=hidden_features * 2, bias=bias
        )
        self.dwconv7 = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=7, stride=1, padding=3,
            groups=hidden_features * 2, bias=bias
        )
        # project_out: hidden → C
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # ═══════ 频域分支: patch-wise FFT + 可学习滤波 ═══════
        # 可学习频域滤波器: (dim, 1, 1, ps, ps//2+1)
        #   每个通道独立滤波, 所有 patch 位置共享 (参数高效)
        #   初始化为 0.5: 给训练自由度, 不完全通过也不完全阻断
        self.freq_filter = nn.Parameter(
            torch.ones(dim, 1, 1, freq_patch_size, freq_patch_size // 2 + 1) * 0.5
        )
        # 频域输出投影
        self.freq_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # ═══════ 频率门控机制 (FGM) ═══════
        # 参考 EVSSM (CVPR 2025) 和 FAFN (2025)
        # 用空域特征生成逐像素门控图, 自适应控制频域分支在每个位置的贡献
        # 相比 v4 的标量 gamma, FGM 能根据空间位置动态调整:
        #   - 雨线区域: 需要更多频域恢复 → gate 较大
        #   - 干净区域: 无需频域修正 → gate 较小
        self.freq_gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                      groups=dim, bias=bias),  # DWConv 提取局部特征
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),  # 1×1 通道混合
            nn.Sigmoid()  # 输出 [0, 1] 门控图
        )
        # gamma: 全局缩放因子, 初始化为 0.1, 控制频域分支的整体强度
        # 训练初期保持微弱贡献, 避免频域分支干扰已有的空域学习
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
            output: (B, C, H, W) — 空域 + 频域门控融合后的特征图
        """
        # =============================================
        # 空域分支: 多尺度 DWConv + GeLU 门控
        # =============================================
        x_proj = self.project_in(x)  # (B, 2×hidden, H, W)
        # 多尺度 DWConv: 三个尺度之和 (参考 FAFN 消融: +0.15 dB)
        x_ms = self.dwconv3(x_proj) + self.dwconv5(x_proj) + self.dwconv7(x_proj)
        # GeLU 门控: GeLU(branch1) × branch2 (来自 Restormer)
        x1, x2 = x_ms.chunk(2, dim=1)  # 各 (B, hidden, H, W)
        x_spatial = F.gelu(x1) * x2
        x_spatial = self.project_out(x_spatial)  # (B, C, H, W)

        # =============================================
        # 频域分支: patch-wise FFT + 可学习滤波
        # =============================================
        # Step 1: pad 到 patch_size 整数倍
        x_padded, orig_H, orig_W = self._pad_to_patch(x)
        B, C, Hp, Wp = x_padded.shape
        ps = self.freq_patch_size

        # Step 2: 切分为 patch
        x_patch = x_padded.reshape(B, C, Hp // ps, ps, Wp // ps, ps)
        x_patch = x_patch.permute(0, 1, 2, 4, 3, 5)  # (B, C, nH, nW, ps, ps)

        # Step 3: patch-wise rfft2
        x_freq = torch.fft.rfft2(x_patch.float(), norm='ortho')

        # Step 4: 可学习频域滤波
        x_freq = x_freq * self.freq_filter

        # Step 5: irfft2 还原
        x_patch_out = torch.fft.irfft2(x_freq, s=(ps, ps), norm='ortho').to(x.dtype)

        # Step 6: 还原为完整特征图
        x_freq_out = x_patch_out.permute(0, 1, 2, 4, 3, 5)
        x_freq_out = x_freq_out.reshape(B, C, Hp, Wp)

        # Step 7: 去除 padding
        x_freq_out = x_freq_out[:, :, :orig_H, :orig_W]
        x_freq_out = self.freq_proj(x_freq_out)

        # =============================================
        # 频率门控融合 (FGM)
        # =============================================
        # gate: (B, C, H, W) ∈ [0,1], 由空域输入 x 生成
        # 效果: 空间自适应地控制频域分支的贡献
        gate = self.freq_gate(x)  # (B, C, H, W), 值域 [0, 1]
        return x_spatial + self.gamma * gate * x_freq_out


# ============================================================================
# C2: 多维通道注意力 (Multi-Dimensional Channel Attention, MDCA)
# ============================================================================
#
# 参考: Efficient Image Restoration with Frequency-Aware Transformer (2025)
#
# 【问题背景】
# 原始 Transposed Attention (MDTA, Restormer CVPR 2022) 计算通道间注意力:
#   attn = Softmax(K^T · Q / α)  ← shape: (C/h × C/h)
# 这对通道建模很强, 但缺少对 H, W 空间维度的显式建模。
#
# 【MDCA 方案】
# 在 Transposed Attention 输出 A_c 上, 叠加三个轻量级注意力分支:
#   A_w  = σ(Conv1×1(AvgPool_h(A_c)))    — 沿高度压缩, 得到宽度注意力
#   A_h  = σ(Conv1×1(AvgPool_w(A_c)))    — 沿宽度压缩, 得到高度注意力
#   A_ch = σ(Conv1×1(AvgPool_hw(A_c)))   — 空间全局压缩, 得到通道注意力
# 最终: output = A_c ⊗ A_w ⊗ A_h ⊗ A_ch (逐元素乘)
#
# 【消融支持】(Rain100L)
#   Basic CA: 33.18 → +Width: 33.39 → +Height: 33.49 → +Channel: 33.67
#   每个维度都有贡献, 总增益 +0.49 dB
# 【计算开销】
#   参数: 8.62M → 8.64M (仅 +0.02M, 增加 0.3%)
#   FLOPs: 50.68G → 54.59G (增加 ~8%)
# ============================================================================

class MDCA(nn.Module):
    """
    多维通道注意力 (Multi-Dimensional Channel Attention)。

    在 Transposed Attention 输出上叠加 Width/Height/Channel 三维注意力,
    以几乎零开销增强特征表达能力。

    参考: Efficient Image Restoration with Frequency-Aware Transformer (2025)

    用法:
        在 Attention 输出之后调用:
        attn_out = self.attn(self.norm1(x))  # 原始 Transposed Attention
        attn_out = self.mdca(attn_out)       # MDCA 增强
        x = x + attn_out

    Args:
        dim: 通道数 (与 Attention 的 dim 一致)
    """
    def __init__(self, dim):
        super().__init__()
        # Width 注意力: 沿高度 mean → 1×1 Conv → Sigmoid → (B, C, 1, W)
        self.width_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        # Height 注意力: 沿宽度 mean → 1×1 Conv → Sigmoid → (B, C, H, 1)
        self.height_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        # Channel 注意力: 全局空间 AvgPool → 1×1 Conv → Sigmoid → (B, C, 1, 1)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, 1, 1)
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) — Transposed Attention 的输出

        Returns:
            x: (B, C, H, W) — 经过多维注意力加权的输出
        """
        # Width 注意力: 沿高度 mean → (B, C, 1, W) → Conv → Sigmoid
        a_w = torch.sigmoid(self.width_conv(x.mean(dim=2, keepdim=True)))
        # Height 注意力: 沿宽度 mean → (B, C, H, 1) → Conv → Sigmoid
        a_h = torch.sigmoid(self.height_conv(x.mean(dim=3, keepdim=True)))
        # Channel 注意力: (B, C, 1, 1) → Conv → Sigmoid
        a_ch = self.channel_attn(x)
        # 多维融合: 逐元素乘 (广播: a_w/a_h/a_ch 自动扩展到 (B,C,H,W))
        return x * a_w * a_h * a_ch
