import math
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
# from mamba_ssm.modules.mamba_simple import Mamba

from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

from models.mamba_simple import Mamba


class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # self.norm1 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='none', 
                           if_devide_out=True, use_norm=True)

    def forward(self, input):
        # input: (B, N, C)
        skip = input
        input = self.norm(input)
        output = self.block(input)
        # output = self.norm1(output)
        return output + skip


class CrossMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v0', 
                           if_devide_out=True, use_norm=True)

    def forward(self, input0, input1):
        # input0: (B, N, C) | input1: (B, N, C)
        skip = input0
        input0 = self.norm0(input0)
        input1 = self.norm1(input1)
        output = self.block(input0, extra_emb=input1)
        # output = self.norm2(output)
        return output + skip


class FusionMamba(nn.Module):
    def __init__(self, dim,depth=2):
        super().__init__()
        self.pcd_mamba_layers = nn.ModuleList([])
        self.mask_mamba_layers = nn.ModuleList([])
        for _ in range(depth):
            self.pcd_mamba_layers.append(SingleMambaBlock(dim))
            self.mask_mamba_layers.append(SingleMambaBlock(dim))
        self.pcd_cross_mamba = CrossMambaBlock(dim)
        self.mask_cross_mamba = CrossMambaBlock(dim)
        self.out_proj = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, pcd, mask):
         # pcd: (N, C) | mask: (N, C)
        pcd = pcd.unsqueeze(0)
        mask = mask.unsqueeze(0)
         # pcd: (1, N, C) | mask: (1, N, C)
        for pcd_layer, mask_layer in zip(self.pcd_mamba_layers, self.mask_mamba_layers):
            pcd = pcd_layer(pcd)
            mask = mask_layer(mask)
        pcd_fusion = self.pcd_cross_mamba(pcd, mask)
        mask_fusion = self.mask_cross_mamba(mask, pcd)
        pcd_fusion = pcd_fusion.squeeze()
        mask_fusion = mask_fusion.squeeze()
        fusion = self.out_proj(torch.cat([pcd_fusion, mask_fusion], dim=-1))
        return fusion

import torch
import torch.nn as nn
def test_fusion_block():
    # 实例化模块
    model = FusionMamba(128).to('cuda')

    # 创建随机输入张量
    input0 = torch.randn(30000, 128).to('cuda')
    input1 = torch.randn(30000, 128).to('cuda')
    output = model(input0, input1)
    print(output.shape)

if __name__ == "__main__":
    test_fusion_block()