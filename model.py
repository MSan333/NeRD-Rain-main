import torch
from torch import nn
from torch.nn import functional as F
import numbers
from einops import rearrange
from mlp import INR
# ★ 导入 AttSSM 模块 (本文创新贡献)
# AttSSMSequential 替换原始的 nn.Sequential(*[TransformerBlock...])
# 用于 6 个 bottleneck latent 层 (small×1 + mid×2 + max×3)
# 融合了 C1(S-scan) + C2(通道注意力+跨尺度状态) + C3(频域FFN) 三个创新点
from attssm_modules import AttSSMSequential


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                                   groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                          groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, BasicConv=BasicConv):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = BasicConv(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, bias=bias,
                                relu=False, groups=hidden_features * 2)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, BasicConv=BasicConv):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = BasicConv(dim * 3, dim * 3, kernel_size=3, stride=1, bias=bias, relu=False, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, BasicConv=BasicConv):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, BasicConv=BasicConv)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, BasicConv=BasicConv)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Fusion(nn.Module):
    def __init__(self, in_dim=32):
        super(Fusion, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        x_q = self.query_conv(x)
        y_k = self.key_conv(y)
        energy = x_q * y_k
        attention = self.sig(energy)
        attention_x = x * attention
        attention_y = y * attention

        x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

        y_gamma = self.gamma2(torch.cat((y, attention_y), dim=1))
        y_out = y * y_gamma[:, [0], :, :] + attention_y * y_gamma[:, [1], :, :]

        x_s = x_out + y_out

        return x_s


class MultiscaleNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[2, 3, 3],
                 heads=[1, 2, 4],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(MultiscaleNet, self).__init__()

        # ══════════════════════════════════════════════════════════════
        # ★ 创新改动: AttSSM Bottleneck 层构建辅助函数
        # ══════════════════════════════════════════════════════════════
        # 原始代码:
        #   self.latent_xxx = nn.Sequential(*[
        #       TransformerBlock(dim=latent_dim, num_heads=heads[2], ...)
        #       for _ in range(num_blocks[2])
        #   ])
        #
        # 替换为:
        #   self.latent_xxx = AttSSMSequential(dim=latent_dim, num_blocks=n)
        #
        # 替换理由:
        #   1. Mamba SSM 的 O(N) 复杂度 vs Transformer 的 O(N²)
        #   2. S 形扫描保持 2D 空间局部性 (C1)
        #   3. 通道注意力解决像素欠激活 (C2)
        #   4. 频域增强 FFN 更好地恢复高频细节 (C3)
        #   5. 跨尺度状态传递利用多尺度架构的信息流 (C2, 原创)
        #
        # 仅替换 bottleneck 层, encoder/decoder 层保持原始 TransformerBlock
        # ══════════════════════════════════════════════════════════════
        latent_dim = int(dim * 2 ** 2)  # = 48 × 4 = 192 (bottleneck 通道数)
        def make_attssm_latent(n):
            """构建 n 个 AttSSMBlock 的序列容器 (替换 TransformerBlock 序列)"""
            return AttSSMSequential(
                dim=latent_dim, num_blocks=n,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, LayerNorm_type=LayerNorm_type,
            )
        self.patch_embed_small = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1_small = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2_small = Downsample(dim)
        self.encoder_level2_small = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3_small = Downsample(int(dim * 2 ** 1))
        # ★ AttSSM bottleneck (替换原始 TransformerBlock)
        self.latent_small = make_attssm_latent(num_blocks[2])

        self.up3_2_small = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2_small = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2_small = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1_small = Upsample(int(dim * 2 ** 1))
        self.reduce_chan_level1_small = nn.Conv2d(int(dim * 2 ** 1), int(dim * 1 ** 1), kernel_size=1, bias=bias)
        self.decoder_level1_small = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.output_small = nn.Conv2d(int(dim * 1 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.INR = INR(dim).cuda()

        self.patch_embed_mid = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1_mid1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.encoder_level1_mid2 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2_mid = Downsample(dim)
        self.down1_2_mid2 = Downsample(dim)
        self.encoder_level2_mid1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2_mid2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3_mid = Downsample(int(dim * 2 ** 1))
        self.down2_3_mid2 = Downsample(int(dim * 2 ** 1))
        # ★ AttSSM bottleneck (替换原始 TransformerBlock)
        self.latent_mid1 = make_attssm_latent(num_blocks[2])
        self.latent_mid2 = make_attssm_latent(num_blocks[2])

        self.up3_2_mid = Upsample(int(dim * 2 ** 2))
        self.up3_2_mid2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2_mid1 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level2_mid2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2_mid1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level2_mid2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1_mid = Upsample(int(dim * 2 ** 1))
        self.up2_1_mid2 = Upsample(int(dim * 2 ** 1))
        self.reduce_chan_level1_mid1 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 1 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level1_mid2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 1 ** 1), kernel_size=1, bias=bias)
        self.decoder_level1_mid1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1_mid2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.output_mid = nn.Conv2d(int(dim * 1 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_mid_context = nn.Conv2d(int(dim * 1 ** 1), dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.INR2 = INR(dim).cuda()

        self.patch_embed_max = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1_max1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_max2 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_max3 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2_max = Downsample(dim)
        self.down1_2_max2 = Downsample(dim)
        self.down1_2_max3 = Downsample(dim)
        self.encoder_level2_max1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2_max2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2_max3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3_max = Downsample(int(dim * 2 ** 1))
        self.down2_3_max2 = Downsample(int(dim * 2 ** 1))
        self.down2_3_max3 = Downsample(int(dim * 2 ** 1))
        # ★ AttSSM bottleneck (替换原始 TransformerBlock)
        self.latent_max1 = make_attssm_latent(num_blocks[2])
        self.latent_max2 = make_attssm_latent(num_blocks[2])
        self.latent_max3 = make_attssm_latent(num_blocks[2])

        self.up3_2_max = Upsample(int(dim * 2 ** 2))
        self.up3_2_max2 = Upsample(int(dim * 2 ** 2))
        self.up3_2_max3 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2_max1 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level2_max2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level2_max3 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2_max1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level2_max2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level2_max3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1_max = Upsample(int(dim * 2 ** 1))
        self.up2_1_max2 = Upsample(int(dim * 2 ** 1))
        self.up2_1_max3 = Upsample(int(dim * 2 ** 1))
        self.reduce_chan_level1_max1 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 1 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level1_max2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 1 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level1_max3 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 1 ** 1), kernel_size=1, bias=bias)
        self.decoder_level1_max1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1_max2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1_max3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.output_max = nn.Conv2d(int(dim * 1 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_max_context1 = nn.Conv2d(int(dim * 1 ** 1), dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_max_context2 = nn.Conv2d(int(dim * 1 ** 1), dim, kernel_size=3, stride=1, padding=1, bias=bias)

        # ══════════════════════════════════════════════════════════════
        # ★ C2 原创贡献: 跨尺度状态传递投影层
        # ══════════════════════════════════════════════════════════════
        # SSM 的隐状态天然携带了序列的全局摘要信息。
        # NeRD-Rain 的 small→mid→max 三尺度架构提供了从低分辨率到
        # 高分辨率的渐进处理流程。我们利用 SSM 状态来在尺度间
        # 传递全局结构信息:
        #
        #   state_small (192-d) → Linear → 注入到 mid 尺度的 Mamba 输入
        #   state_mid   (192-d) → Linear → 注入到 max 尺度的 Mamba 输入
        #
        # Linear 投影的作用:
        #   不同尺度的特征空间可能有差异 (虽然通道数相同),
        #   Linear 投影将上一尺度的状态映射到当前尺度的特征空间。
        #   使用无偏置的 Linear 保持轻量级。
        #
        # 这是本文的原创设计, 利用了:
        #   (a) SSM 的状态传递特性 (天然适合序列间信息传递)
        #   (b) NeRD-Rain 的多尺度级联架构 (提供了传递通道)
        # 在现有文献 (MaIR, MambaIRv2, EVSSM) 中均未见此设计。
        # ══════════════════════════════════════════════════════════════
        self.state_small2mid = nn.Linear(latent_dim, latent_dim, bias=False)
        self.state_mid2max = nn.Linear(latent_dim, latent_dim, bias=False)

        self.BF1 = Fusion(dim * 4)
        self.BF2 = Fusion(dim * 4)
        self.BF3 = Fusion(dim * 4)

        self.upsmall2mid1 = Upsample(int(dim * 4 ** 1))
        self.upsmall2mid2 = Upsample(int(dim * 2 ** 1))

        self.upmid2max1 = Upsample(int(dim * 4 ** 1))
        self.upmid2max2 = Upsample(int(dim * 2 ** 1))

    def forward(self, inp_img):
        outputs = list()

        inp_img_max = inp_img
        inp_img_mid = F.interpolate(inp_img, scale_factor=0.5)
        inp_img_small = F.interpolate(inp_img, scale_factor=0.25)

        inp_enc_level1_small = self.patch_embed_small(inp_img_small)
        out_enc_level1_small = self.encoder_level1_small(inp_enc_level1_small)

        inp_enc_level2_small = self.down1_2_small(out_enc_level1_small)
        out_enc_level2_small = self.encoder_level2_small(inp_enc_level2_small)

        inp_enc_level4_small = self.down2_3_small(out_enc_level2_small)
        # ★ AttSSM bottleneck: 返回 (feature, state)
        # small 尺度是第一个处理的，没有上一尺度的状态，prev_state=None
        # ssm_state_small 是全局平均池化的 (B, 192) 向量，将传递给 mid 尺度
        latent_small, ssm_state_small = self.latent_small(inp_enc_level4_small, prev_state=None)
        latent_small_mid = self.upsmall2mid1(latent_small)
        latent_small_mid = self.upsmall2mid2(latent_small_mid)

        outputs.append(inp_img_small)
        INR = self.INR(latent_small_mid)
        inp_img_small_ = INR + inp_img_small
        outputs.append(inp_img_small_)

        inp_img_small_ = F.interpolate(inp_img_small_, scale_factor=2)

        mid_img = inp_img_mid + inp_img_small_

        inp_enc_level1_mid = self.patch_embed_mid(mid_img)
        out_enc_level1_mid = self.encoder_level1_mid1(inp_enc_level1_mid)

        inp_enc_level2_mid = self.down1_2_mid(out_enc_level1_mid)
        out_enc_level2_mid = self.encoder_level2_mid1(inp_enc_level2_mid)

        inp_enc_level4_mid = self.down2_3_mid(out_enc_level2_mid)
        # ★ C2 跨尺度状态传递: small → mid
        # ssm_state_small (B,192) 经过 Linear 投影后注入到 mid 尺度
        # 效果: mid 尺度的 Mamba SSM 可以利用 small 尺度提取的全局结构信息
        latent_mid, ssm_state_mid1 = self.latent_mid1(
            inp_enc_level4_mid, prev_state=self.state_small2mid(ssm_state_small))
        latent_mid_INR_max = self.upmid2max1(latent_mid)
        latent_mid_INR_max = self.upmid2max2(latent_mid_INR_max)

        outputs.append(mid_img / 2)
        INR2 = self.INR2(latent_mid_INR_max)
        mid_img_ = INR2 + mid_img
        outputs.append(mid_img_)

        mid_img_ = F.interpolate(mid_img_, scale_factor=2)

        max_img = inp_img_max + mid_img_

        inp_enc_level1_max = self.patch_embed_max(max_img)
        out_enc_level1_max = self.encoder_level1_max1(inp_enc_level1_max)

        inp_enc_level2_max = self.down1_2_max(out_enc_level1_max)
        out_enc_level2_max = self.encoder_level2_max1(inp_enc_level2_max)

        inp_enc_level4_max = self.down2_3_max(out_enc_level2_max)
        # ★ C2 跨尺度状态传递: mid → max
        # ssm_state_mid1 (B,192) 经过 Linear 投影后注入到 max 尺度
        # 传递链完整: small → mid → max (全局结构信息逐级传递)
        # 注意: max 的后续迭代 (max2, max3) 不再接收跨尺度状态,
        #        因为它们处理的是同尺度的 recurrence refinement
        latent_max, _ = self.latent_max1(
            inp_enc_level4_max, prev_state=self.state_mid2max(ssm_state_mid1))
        BFF_max_1 = latent_max

        inp_dec_level2_max = self.up3_2_max(latent_max)
        inp_dec_level2_max = torch.cat([inp_dec_level2_max, out_enc_level2_max], 1)
        inp_dec_level2_max = self.reduce_chan_level2_max1(inp_dec_level2_max)
        out_dec_level2_max = self.decoder_level2_max1(inp_dec_level2_max)

        inp_dec_level1_max = self.up2_1_max(out_dec_level2_max)
        inp_dec_level1_max = torch.cat([inp_dec_level1_max, out_enc_level1_max], 1)
        inp_dec_level1_max = self.reduce_chan_level1_max1(inp_dec_level1_max)
        out_dec_level1_max = self.decoder_level1_max1(inp_dec_level1_max)

        out_dec_level1_max = self.output_max_context1(out_dec_level1_max)
        out_enc_level1_max = self.encoder_level1_max2(out_dec_level1_max)

        inp_enc_level2_max = self.down1_2_max2(out_enc_level1_max)
        out_enc_level2_max = self.encoder_level2_max2(inp_enc_level2_max)

        inp_enc_level4_max = self.down2_3_max2(out_enc_level2_max)
        latent_max, _ = self.latent_max2(inp_enc_level4_max)
        BFF_max_2 = latent_max

        inp_dec_level2_max = self.up3_2_max2(latent_max)
        inp_dec_level2_max = torch.cat([inp_dec_level2_max, out_enc_level2_max], 1)
        inp_dec_level2_max = self.reduce_chan_level2_max2(inp_dec_level2_max)
        out_dec_level2_max = self.decoder_level2_max2(inp_dec_level2_max)

        inp_dec_level1_max = self.up2_1_max2(out_dec_level2_max)
        inp_dec_level1_max = torch.cat([inp_dec_level1_max, out_enc_level1_max], 1)
        inp_dec_level1_max = self.reduce_chan_level1_max2(inp_dec_level1_max)
        out_dec_level1_max = self.decoder_level1_max2(inp_dec_level1_max)

        out_dec_level1_max = self.output_max_context2(out_dec_level1_max)
        out_enc_level1_max = self.encoder_level1_max3(out_dec_level1_max)

        inp_enc_level2_max = self.down1_2_max3(out_enc_level1_max)
        out_enc_level2_max = self.encoder_level2_max3(inp_enc_level2_max)

        inp_enc_level4_max = self.down2_3_max3(out_enc_level2_max)
        latent_max, _ = self.latent_max3(inp_enc_level4_max)
        BFF_max_3 = latent_max

        BFF1 = self.BF1(BFF_max_1, BFF_max_2)
        BFF2 = self.BF2(BFF_max_2, BFF_max_3)

        BFF1 = F.interpolate(BFF1, scale_factor=0.5)
        BFF2 = F.interpolate(BFF2, scale_factor=0.5)

        inp_dec_level2_max = self.up3_2_max3(latent_max)

        BFF3_1 = latent_mid
        latent_mid = latent_mid + BFF1

        inp_dec_level2_mid = self.up3_2_mid(latent_mid)
        inp_dec_level2_mid = torch.cat([inp_dec_level2_mid, out_enc_level2_mid], 1)
        inp_dec_level2_mid = self.reduce_chan_level2_mid1(inp_dec_level2_mid)
        out_dec_level2_mid = self.decoder_level2_mid1(inp_dec_level2_mid)

        inp_dec_level1_mid = self.up2_1_mid(out_dec_level2_mid)
        inp_dec_level1_mid = torch.cat([inp_dec_level1_mid, out_enc_level1_mid], 1)
        inp_dec_level1_mid = self.reduce_chan_level1_mid1(inp_dec_level1_mid)
        out_dec_level1_mid = self.decoder_level1_mid1(inp_dec_level1_mid)

        out_dec_level1_mid = self.output_mid_context(out_dec_level1_mid)
        out_enc_level1_mid = self.encoder_level1_mid2(out_dec_level1_mid)

        inp_enc_level2_mid = self.down1_2_mid2(out_enc_level1_mid)
        out_enc_level2_mid = self.encoder_level2_mid2(inp_enc_level2_mid)

        inp_enc_level4_mid = self.down2_3_mid2(out_enc_level2_mid)
        latent_mid, _ = self.latent_mid2(inp_enc_level4_mid)
        BFF3_2 = latent_mid
        BFF3 = self.BF3(BFF3_1, BFF3_2)
        BFF3 = F.interpolate(BFF3, scale_factor=0.5)

        latent_mid = latent_mid + BFF2

        inp_dec_level2_mid = self.up3_2_mid2(latent_mid)

        latent_small = latent_small + BFF3

        inp_dec_level2_small = self.up3_2_small(latent_small)
        inp_dec_level2_small = torch.cat([inp_dec_level2_small, out_enc_level2_small], 1)
        inp_dec_level2_small = self.reduce_chan_level2_small(inp_dec_level2_small)
        out_dec_level2_small = self.decoder_level2_small(inp_dec_level2_small)

        inp_dec_level1_small = self.up2_1_small(out_dec_level2_small)
        inp_dec_level1_small = torch.cat([inp_dec_level1_small, out_enc_level1_small], 1)
        inp_dec_level1_small = self.reduce_chan_level1_small(inp_dec_level1_small)
        out_dec_level1_small = self.decoder_level1_small(inp_dec_level1_small)

        small_2_mid = out_dec_level1_small

        out_dec_level1_small = self.output_small(out_dec_level1_small) + inp_img_small

        outputs.append(out_dec_level1_small)
        small = F.interpolate(out_dec_level1_small, scale_factor=2)

        inp_dec_level2_mid = torch.cat([inp_dec_level2_mid, out_enc_level2_mid], 1)
        inp_dec_level2_mid = self.reduce_chan_level2_mid2(inp_dec_level2_mid)
        out_dec_level2_mid = self.decoder_level2_mid2(inp_dec_level2_mid)

        inp_dec_level1_mid = self.up2_1_mid2(out_dec_level2_mid)
        inp_dec_level1_mid = torch.cat([inp_dec_level1_mid, out_enc_level1_mid], 1)
        inp_dec_level1_mid = self.reduce_chan_level1_mid2(inp_dec_level1_mid)
        out_dec_level1_mid = self.decoder_level1_mid2(inp_dec_level1_mid)

        small_2_mid = F.interpolate(small_2_mid, scale_factor=2)
        out_dec_level1_mid = out_dec_level1_mid + small_2_mid

        mid_2_max = out_dec_level1_mid

        out_dec_level1_mid = self.output_mid(out_dec_level1_mid) + inp_img_mid

        outputs.append(out_dec_level1_mid)
        mid = F.interpolate(out_dec_level1_mid, scale_factor=2)

        inp_dec_level2_max = torch.cat([inp_dec_level2_max, out_enc_level2_max], 1)
        inp_dec_level2_max = self.reduce_chan_level2_max3(inp_dec_level2_max)
        out_dec_level2_max = self.decoder_level2_max3(inp_dec_level2_max)

        inp_dec_level1_max = self.up2_1_max3(out_dec_level2_max)
        inp_dec_level1_max = torch.cat([inp_dec_level1_max, out_enc_level1_max], 1)
        inp_dec_level1_max = self.reduce_chan_level1_max2(inp_dec_level1_max)
        mid_2_max = F.interpolate(mid_2_max, scale_factor=2)
        out_dec_level1_max = self.decoder_level1_max3(inp_dec_level1_max) + mid_2_max

        out_dec_level1_max = self.output_max(out_dec_level1_max) + inp_img_max

        outputs.append(out_dec_level1_max)

        return outputs[::-1]
