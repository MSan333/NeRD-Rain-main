# MFGCP 实现方案 — 多尺度频域-梯度协同感知模块

## 1. 模块设计

### 1.1 MDPConv（多方向感知卷积）

> 参考论文：DeRainMamba (IEEE SPL 2025, Zhiliang Zhu et al.)

**核心公式：**

$$
F_{\text{out}} = \sum_{i=1}^{5} (F_{\text{in}} * K_i)
$$

推理时可通过**结构重参数化**将 5 个分支合并为单个等效 3×3 卷积核，实现零推理开销。

**5 个方向微分卷积核（均为 depthwise, `groups=dim`）：**

| 缩写 | 全称 | 核尺寸 | 功能描述 |
|------|------|--------|---------|
| HDC | Horizontal Difference Conv | 1×3 | 捕获水平边缘与水平方向雨纹 |
| VC | Vertical Conv | 3×1 | 捕获竖直方向雨线 |
| CDC | Central Difference Conv | 3×3 | 增强中心像素与邻域的对比度差异 |
| ADC | Anti-diagonal Conv | 3×3 | 捕获对角线方向纹理 |
| VDC | Vertical-diagonal Conv | 3×3 | 捕获反对角线方向纹理 |

**输出：** 5 分支逐元素求和 → 1×1 Conv 通道投影

**核心思想：** 雨纹具有强烈的方向性先验（水平、竖直、对角线），通过多方向微分卷积显式建模不同方向的梯度响应，从而更精准地分离雨纹与背景纹理。

---

### 1.2 MFGCP_TransformerBlock

基于现有 `RFM_TransformerBlock` 扩展，在 FFN 支路中并联引入 MDPConv 分支。

**前向传播：**

```
x = x + attn(norm1(x))                          # 注意力支路（保持不变）
x = x + ffn(norm2(x)) + rfm(x) + α * mdpconv(x) # FFN + RFM + MDPConv 并联
```

- **α**：可学习标量参数（per-channel 或全局），初始化为 `0.1`
  - 采用较小初始值，让模块在训练初期主要依赖已有 FFN+RFM，逐步引入梯度感知信息
  - 训练过程中 α 自适应调整，平衡频域增强与空域梯度恢复的贡献

---

### 1.3 集成位置

将现有 `MultiscaleNet` 中所有 `RFM_TransformerBlock` 替换为 `MFGCP_TransformerBlock`：

| 尺度层级 | 变量名 | Block 数量 | 通道数 dim |
|---------|--------|-----------|-----------|
| 小尺度 | `latent_small` | 3 | 48 |
| 中尺度 | `latent_mid` | 2 | 96 |
| 大尺度 | `latent_max` | 3 | 192 |

**共计：8 个 MFGCP_TransformerBlock 实例**

> 注：实际 Block 数量以 `model.py` 中定义为准，上表为参考值。

---

## 2. 参数量估算

以 `dim=48`（小尺度层）为例：

| 组件 | 计算方式 | 参数量 |
|------|---------|-------|
| HDC (1×3 depthwise) | 48 × (1×3) = 144 | 144 |
| VC (3×1 depthwise) | 48 × (3×1) = 144 | 144 |
| CDC (3×3 depthwise) | 48 × (3×3) = 432 | 432 |
| ADC (3×3 depthwise) | 48 × (3×3) = 432 | 432 |
| VDC (3×3 depthwise) | 48 × (3×3) = 432 | 432 |
| proj (1×1 Conv) | 48 × 48 = 2304 | 2304 |
| alpha (可学习标量) | 1 | 1 |
| **单实例合计** | | **~3,889** |

**多尺度总参数量估算：**

| 层级 | dim | 实例数 | 单实例参数 | 小计 |
|------|-----|--------|-----------|------|
| latent_small | 48 | 3 | ~3,889 | ~11,667 |
| latent_mid | 96 | 2 | ~15,553 | ~31,106 |
| latent_max | 192 | 3 | ~62,209 | ~186,627 |
| **总计** | | **8** | | **~229,400（约 0.23M）** |

> 结论：MDPConv 引入的额外参数约 **0.23M**，相对于整体模型参数量占比极小（通常 < 5%），不会显著增加计算负担。

---

## 3. 与现有架构的关系

```
输入特征
    │
    ├──→ LayerNorm → Attention ──────────────────→ (+) → 中间特征
    │                                                   │
    │         ┌─────────────────────────────────────────┘
    │         ▼
    │    LayerNorm
    │         │
    │    ┌────┼────────────────┐
    │    ▼    ▼                ▼
    │  FFN  RFM          α·MDPConv
    │  (通道变换)  (频域增强)   (梯度感知)
    │    │    │                │
    │    └────┴────────┬───────┘
    │                  ▼
    └────────────→ (+) → 输出特征
```

**各组件功能与互补关系：**

| 组件 | 功能定位 | 与其他组件的互补关系 |
|------|---------|-------------------|
| **Attention** | 全局特征交互，建模长距离依赖 | 为 RFM 和 MDPConv 提供富含上下文的全局特征 |
| **FFN** | 通道维度非线性变换 + 空间混合 | 基础特征映射，为其他支路提供非线性表达能力 |
| **RFM** | 频域特征增强，显式抑制雨纹频率成分 | 在频域层面分离雨纹；与 MDPConv 在空域层面形成频域-空域互补 |
| **MDPConv（新增）** | 多方向梯度感知，恢复高频细节 | 补充 RFM 在空域方向性信息建模上的不足；增强边缘和纹理细节 |

**协同机制：** RFM 从频域角度抑制雨纹的周期性频率成分，MDPConv 从空域角度利用雨纹的方向性梯度先验进行分离，两者形成**频域-空域双域协同**，实现更全面的雨纹去除。

---

## 4. 消融实验计划

### 4.1 主要消融实验（Rain200L 数据集，evaluate_python.py Y通道评估）

| 实验编号 | 配置 | Attention | FFN | RFM | MDPConv | PSNR (dB) | SSIM |
|---------|------|-----------|-----|-----|---------|----------|------|
| A | Baseline（原始 TransformerBlock） | ✓ | ✓ | ✗ | ✗ | 41.71 | 0.9903 |
| B | + RFM（现有模型） | ✓ | ✓ | ✓ | ✗ | 41.79 | 0.9906 |
| C | + MDPConv only（无 RFM） | ✓ | ✓ | ✗ | ✓ | 待实验 | 待实验 |
| **D** | **+ RFM + MDPConv（MFGCP 完整）** | **✓** | **✓** | **✓** | **✓** | **训练中** | **训练中** |

### 4.2 细粒度消融变量

**（1）MDPConv 方向数量消融：**

| 方向数 | 使用的核 | 预期效果 |
|--------|---------|---------|
| 1 | HDC only | 基线方向感知，小幅涨点 |
| 3 | HDC + VC + CDC | 覆盖水平/竖直/中心差分，主要涨点来源 |
| **5** | **HDC + VC + CDC + ADC + VDC** | **完整方向覆盖，预期最优** |

**（2）alpha 参数策略消融：**

| 策略 | 初始值 | 描述 |
|------|--------|------|
| 固定值 | α = 0.1 | 简单稳定，无额外参数 |
| 可学习全局标量 | α = 0.1（init） | 自适应调整，推荐方案 |
| 可学习 per-channel | α ∈ R^dim（init=0.1） | 更精细控制，参数量略增 |

**（3）MDPConv 集成方式消融：**

| 方式 | 公式 | 说明 |
|------|------|------|
| 并联相加（推荐） | `ffn(x) + rfm(x) + α·mdpconv(x)` | 各支路独立提取特征后融合 |
| 串联级联 | `mdpconv(rfm(x)) + ffn(x)` | RFM 输出再经 MDPConv 细化 |
| 门控融合 | `ffn(x) + gate(rfm(x), mdpconv(x))` | 学习两模块的动态权重 |

---

## 5. 训练配置建议

### 5.1 学习率策略

| 参数组 | 学习率倍率 | 说明 |
|--------|-----------|------|
| 主干网络（Attention、FFN） | 1×（与原始一致） | 保持主干稳定 |
| RFM 模块 | 1× | 已预训练或联合训练 |
| MDPConv 模块 | 1× | 新增模块，正常学习率 |
| **alpha 参数** | **10×** | 加速 α 收敛，让模块快速找到最优融合权重 |

> 优化器实现示例：
> ```python
> # 将 alpha 参数单独分组，设置更大学习率
> alpha_params = [p for n, p in model.named_parameters() if 'alpha' in n]
> other_params = [p for n, p in model.named_parameters() if 'alpha' not in n]
> optimizer = torch.optim.AdamW([
>     {'params': other_params, 'lr': base_lr},
>     {'params': alpha_params, 'lr': base_lr * 10}
> ], weight_decay=1e-4)
> ```

### 5.2 训练流程建议

```
第一阶段：快速验证（Rain200L，epochs 减半）
    │
    ├─ 目的：确认 MDPConv 带来涨点
    ├─ 数据集：Rain200L（规模小，训练快）
    ├─ 指标：PSNR/SSIM 对比 Baseline-B
    │
    ▼ 确认涨点 ≥ 0.2 dB
第二阶段：完整训练（全数据集，完整 epochs）
    │
    ├─ 数据集：Rain200L + Rain200H + DID-Data + DDN-Data + SPA-Data
    ├─ 保存最优 checkpoint
    └─ 在 5 个测试集上全面评估
```

### 5.3 重参数化推理优化

训练完成后，将 MDPConv 的 5 个 depthwise 卷积分支重参数化为单个等效 3×3 卷积：

```
K_eq = pad(HDC, 3×3) + pad(VC, 3×3) + CDC + ADC + VDC
```

> 推理时零额外开销，训练时保留多分支结构以便梯度流通。

---

## 6. 参考文献

1. **DeRainMamba**: Zhu, Z. et al. "DeRainMamba: A Frequency-Aware State Space Model with Detail Enhancement for Image Deraining." *IEEE Signal Processing Letters (SPL)*, 2025.
2. **Texture-Aware SSM**: "An Efficient Texture-Aware State Space Model for Image Restoration." *IJCAI*, 2025.
3. **NeRD-Rain**: "NeRD-Rain: Bidirectional Multi-Scale Implicit Neural Representations for Image Deraining." *CVPR*, 2024.
