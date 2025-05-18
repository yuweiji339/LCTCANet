import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import ops
import numpy as np

from basicsr.archs.arch_util import trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY

from typing import Sequence, Literal, Optional
from functools import partial

# GlobalContextualLocalBlock (GCLBlock) is a hybrid module that integrates local
# convolutional encoding and global context modeling via multi-head self-attention.
# It is designed to enhance both fine-grained texture recovery and long-range
# structural consistency in super-resolution tasks.

class EdgeStructureFusionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Edge-aware attention branch for enhancing edge information
        self.e_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.Sigmoid()
        )
        # Structural enhancement branch for modeling contextual structure
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


class DeepFeatureExtractionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.esfb = EdgeStructureFusionBlock(dim)

    def forward(self, x):
        identity = x
        x = self.esfb(x)
        return x + identity  # Residual connection to improve stability


class SubPixelReconstruction(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        # 1) Use convolution to expand channels to dim * scale^2
        # 2) Use PixelShuffle to reconstruct higher-resolution feature maps
        # 3) Follow with convolutions to map to 3-channel output
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim * scale**2, 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(dim, dim // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 3, 3, 1, 1)
        )

    def forward(self, x):
        return self.upsample(x)


class LCTCANet_woGCLB(nn.Module):
    def __init__(
        self,
        dim=64,
        n_blocks=4,
        heads=2,
        scale=4,
        upscaling_factor=None,  # Field from configuration mapping
        **kwargs
    ):
        super().__init__()
        if upscaling_factor is not None:
            scale = upscaling_factor  # Use upscaling_factor from config if provided

        self.head = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        self.body = nn.Sequential(
            *[DeepFeatureExtractionBlock(dim) for _ in range(n_blocks)]
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

    model = LCTCANet_woGCLB().cuda()
    # Change input tensor to a 64×64 resolution image
    inp = torch.randn(1, 3, 64, 64).cuda()

    start = time.time()
    macs, params = profile(model, inputs=(inp,))
    end = time.time()

    # Assume each MAC operation includes one multiplication and one addition
    multi_adds = macs * 2  # or called FLOPs
    flops = macs * 2       # Conventional estimate: 1 MAC = 2 FLOPs

    print(f"MACs: {macs/1e9:.2f}G")
    print(f"Multi-Adds: {multi_adds/1e9:.2f}G")
    print(f"FLOPs: {flops/1e9:.2f}G")  # Same as Multi-Adds
    print(f"Params: {params/1e6:.2f}M")
    print(f"Time: {end - start:.2f}s")
