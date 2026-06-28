# NeRD-Rain 改进工作现状

> 基线：NeRD-Rain (CVPR 2024) — Bidirectional Multi-Scale Implicit Neural Representations for Image Deraining
> 更新日期：2026-06-28

---

## 基线架构

NeRD-Rain 采用三级多尺度金字塔 (small ×0.25 / mid ×0.5 / max ×1.0)，每级内部为 Restormer 风格的 Encoder-Bottleneck-Decoder U-Net，基本单元为 TransformerBlock (MDTA + Gated-FFN)。跨尺度通过 INR (隐式神经表示) 做图像空间上采样，Max 级通过 BFF (Bidirectional Feature Fusion) 做多轮特征融合。

损失函数：Charbonnier + 0.01×FFT + 0.05×Edge + 0.1×L1，多尺度监督。

---

## 已完成创新点：RFM — 残差频域模块

### 动机

雨线是具有特定频率特征的退化（高频锐利边缘 + 周期性条纹），原始 NeRD-Rain 的 Transformer 仅在空域建模，缺少对频域信息的显式利用。

### 设计

参考 DeRainMamba (IEEE SPL 2025) 的 FASSM 模块，在 Bottleneck 层引入 RFM 旁路分支：

```
F_out = Attn(LN(x)) + FFN(LN(x)) + RFM(x)
```

RFM 内部：
- **频域分支**：FFT → 幅度谱两层 1×1 Conv+ReLU / 相位谱一层 1×1 Conv → 融合 → iFFT → 1×1 投影
- **空域残差分支**：DWConv 3×3
- 输出 = 频域分支 + 空域残差

### 使用位置

仅在 Bottleneck 层使用 `RFM_TransformerBlock`，编码器/解码器保持原始 TransformerBlock：
- `latent_small` (3 个 block)
- `latent_mid1`, `latent_mid2` (各 3 个 block)
- `latent_max1`, `latent_max2`, `latent_max3` (各 3 个 block)

### 代码位置

| 组件 | 文件 | 行号 |
|------|------|------|
| RFM 模块 | model.py | 118-158 |
| RFM_TransformerBlock | model.py | 211-235 |
| MultiscaleNet (dim=48) | model.py | 303-672 |

### 对比

- `model_S.py`：原始 NeRD-Rain 小模型 (dim=32)，全部使用 TransformerBlock
- `model.py`：改进模型 (dim=48)，Bottleneck 层使用 RFM_TransformerBlock

---

## 待实现模块（已有代码，未接入训练）

### FreqFFN — 频域门控双域 FFN

位置：`attssm_modules.py:79-219`

空域分支（多尺度 DWConv 3×3/5×5/7×7 + GeLU 门控）与频域分支（patch-wise rfft2 + 可学习滤波器）通过频率门控机制 (FGM) 自适应融合。用于替换 TransformerBlock 中的 FeedForward。

### MDCA — 多维通道注意力

位置：`attssm_modules.py:248-294`

在 MDTA 输出上叠加 Width/Height/Channel 三维注意力 (A_w ⊗ A_h ⊗ A_ch)，参数增加仅 0.3%。

### 其他辅助模块

- `layers.py`：DOConv、FFT ResBlock 等（仅在 test/test_speed 中引用，训练未使用）
- `Ablations/`：多种消融变体 (model_a~g, model_woBFPU 等)
