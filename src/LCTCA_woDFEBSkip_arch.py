
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import ops
import numpy as np

from basicsr.archs.arch_util import trunc_normal_

from typing import Sequence, Literal, Optional
from functools import partial

from basicsr.utils.registry import ARCH_REGISTRY


#GlobalContextualLocalBlock (GCLBlock) is a hybrid module that integrates local #convolutional encoding and global context modeling via multi-head self-attention. It is #designed to enhance both fine-grained texture recovery and long-range structural #consistency in super-resolution tasks.
class GlobalContextualLocalBlock(nn.Module):
    def __init__(self, dim, patch_size=2, heads=2):
        super().__init__()
        # 局部表示（Local representation branch）
        self.local_rep = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        # 全局注意力机制（Transformer attention）
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, x):


        b, c, h, w = x.size()
        x_local = self.local_rep(x)  # 局部特征增强
        x_ = rearrange(x_local, 'b c h w -> b (h w) c')  # Flatten for attention
        x_ = self.norm(x_)
        attn_out, _ = self.attn(x_, x_, x_)
        x_ = x_ + attn_out  # 残差连接
        x_out = rearrange(x_, 'b (h w) c -> b c h w', h=h, w=w)  # 恢复原尺寸
        return x_out

#B - batchsize，例如你在使用dataloder的时候设置的batchsize是64那么此项则为64
#C - channel，也就是输入的矩阵的通道数，若你输入的是RGB图片，那么此项为3
#H - high，也就是你输入矩阵的高。
#W - width，也就是你输入矩阵的宽
#解包输入b,c,h,w;以便恢复空间维度
#

class EdgeStructureFusionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Edge-aware Attention Branch 边缘感知增强
        self.e_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.Sigmoid()
        )
        # Structural Enhancement Branch 结构上下文建模
        self.t_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        # Fusion layer to combine edge and structure features
        self.fusion = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        e = self.e_branch(x) * x  # Edge attention modulation
        t = self.t_branch(x)      # Structure transformation
        fused = torch.cat([e, t], dim=1)  # Feature concatenation
        return self.fusion(fused)         # Final fusion


#LGFB
class DeepFeatureExtractionBlock(nn.Module):
    def __init__(self, dim, patch_size=2, heads=2):
        super().__init__()
        # 这里将 transformer_dim 设置为 dim，这里可以根据需要调整
        self.gclb = GlobalContextualLocalBlock(dim, patch_size, heads)
        self.esfb = EdgeStructureFusionBlock(dim)

    def forward(self, x):
        x = self.gclb(x)
        x = self.esfb(x)
        return x   



class SubPixelReconstruction(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        # 1) 先利用卷积把通道提升到 dim * scale^2
        # 2) 再通过 PixelShuffle 还原出更高分辨率特征图
        # 3) 后面再用几层卷积映射到 3 通道输出
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim * scale**2, 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(dim, dim // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 3, 3, 1, 1)
        )

    def forward(self, x):
        return self.upsample(x)






class LCTCANet_woDFEBSkip(nn.Module):
    def __init__(
        self,
        dim=64,
        n_blocks=4,
        heads=2,
        scale=4,
        upscaling_factor=None,  # 映射配置里的字段
        **kwargs
    ):
        super().__init__()
        if upscaling_factor is not None:
            scale = upscaling_factor  # 如果配置里写了 upscaling_factor，就用它
        #self.head = nn.Conv2d(3, dim, 3, 1, 1)
        self.head = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        self.body = nn.Sequential(
            *[DeepFeatureExtractionBlock(dim, patch_size=2, heads=heads) for _ in range(n_blocks)]
        )
        self.tail = SubPixelReconstruction(dim, scale)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x



if __name__ == '__main__':
    import time
    from thop import profile

    model = LCTCANet_woDFEBSkip().cuda()
    # 修改输入张量为 1280x720 分辨率的图像
    #inp = torch.randn(1, 3, 720, 1280).cuda()
    inp = torch.randn(1, 3, 64, 64).cuda()
    start = time.time()
    macs, params = profile(model, inputs=(inp,))
    end = time.time()
    # 假定每个 MAC 操作包含乘法和加法各一次
    multi_adds = macs * 2 # 或者叫 FLOPs
    flops = macs * 2  # 常规估算：1 MAC = 2 FLOPs
    print("MACs: {:.2f}G".format(macs / 1e9))
    print("Multi-Adds: {:.2f}G".format(multi_adds / 1e9))# 或者叫 FLOPs
    print("FLOPs: {:.2f}G".format(flops / 1e9))       # 与 Multi-Adds 一致
    print("Params: {:.2f}M".format(params / 1e6))
    print("Time: {:.2f}s".format(end - start))
