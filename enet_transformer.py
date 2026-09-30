#!/usr/bin/env python3.10

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Jose Dolz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""ENet with one or more Transformer blocks spliced into the encoder/decoder chain

A TransformerBlock keeps the shape and channel count unchanged, so it can be
dropped in after any stage listed in `_STAGE_DIMS` without touching the surrounding
conv layers.

Stages:
    stage1      K*4   after the first bottleneck stack   (H/4 x W/4)
    stage2      K*8   after the second bottleneck stack   (H/8 x W/8)
    bottleneck  K*4   after the dilated middle stack      (H/8 x W/8)   <- default
    decoder1    K     after the first upsampling stack    (H/4 x W/4)
    decoder2    K     after the second upsampling stack    (H/2 x W/2)

Decoder1/decoder2 use dim=K directly so K must be divisible by num_heads then
"""
import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from ENet import (BottleNeck, BottleNeckDownSampling, BottleNeckUpSampling,
                          conv_block, random_weights_init)


def sinusoidal_2d_pos_embed(h: int, w: int, dim: int, device) -> Tensor:
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sinusoidal pos embed"
    d_quarter = dim // 4
    freq = torch.exp(torch.arange(d_quarter, device=device, dtype=torch.float32)
                      * (-math.log(10000.0) / d_quarter))

    pos_h = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1) * freq.unsqueeze(0)
    pos_w = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(1) * freq.unsqueeze(0)

    pe_h = torch.cat([pos_h.sin(), pos_h.cos()], dim=1)          # (h, dim/2)
    pe_w = torch.cat([pos_w.sin(), pos_w.cos()], dim=1)          # (w, dim/2)

    pe = torch.cat([pe_h.unsqueeze(1).expand(h, w, -1),
                    pe_w.unsqueeze(0).expand(h, w, -1)], dim=-1)  # (h, w, dim)
    return pe.reshape(h * w, dim)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_dim=None, dropout=0.1, use_pos_embed=False):
        super().__init__()
        mlp_dim = mlp_dim or dim * 4
        self.use_pos_embed = use_pos_embed

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(mlp_dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        if self.use_pos_embed:
            tokens = tokens + sinusoidal_2d_pos_embed(H, W, C, x.device).unsqueeze(0)

        t = self.norm1(tokens)
        attn_out, _ = self.attn(t, t, t)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))

        return tokens.transpose(1, 2).reshape(B, C, H, W)


class TransformerStack(nn.Module):
    def __init__(self, dim, num_layers=1, **block_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, **block_kwargs) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ENetTransformer(nn.Module):
    VALID_STAGES = ("stage1", "stage2", "bottleneck", "decoder1", "decoder2")

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        proj_factor: int = kwargs.get("factor", 4)
        K: int = kwargs.get("kernels", 16)

        transformer_at = kwargs.get("transformer_at", ("bottleneck",))
        if isinstance(transformer_at, str):
            transformer_at = (transformer_at,)
        unknown = set(transformer_at) - set(self.VALID_STAGES)
        if unknown:
            raise ValueError(f"unknown transformer_at stage(s) {unknown}, expected subset of {self.VALID_STAGES}")
        self.transformer_at = set(transformer_at)

        num_layers = kwargs.get("num_layers", 1)
        num_heads = kwargs.get("num_heads", 4)
        mlp_ratio = kwargs.get("mlp_ratio", 4)
        dropout = kwargs.get("dropout", 0.1)
        use_pos_embed = kwargs.get("use_pos_embed", False)

        stage_dims = {"stage1": K * 4, "stage2": K * 8, "bottleneck": K * 4,
                      "decoder1": K, "decoder2": K}

        # Initial operations
        # The initial block concatenates the conv branch with the max-pooled input, and
        # everything downstream (bottleneck1_0, and the bottleneck5 skip) assumes that
        # concat is exactly K wide. So the conv branch emits K - in_dim, as in the ENet
        # paper (13 + 3 RGB channels = 16). in_dim=1 gives K-1, i.e. the 2D behavior;
        # in_dim>1 is the 2.5D stack (input.context_slices).
        assert in_dim < K, f"in_dim={in_dim} must be < kernels={K} for the initial block"
        self.conv0 = nn.Conv2d(in_dim, K - in_dim, kernel_size=3, stride=2, padding=1)
        self.maxpool0 = nn.MaxPool2d(2, return_indices=False, ceil_mode=False)

        # Downsampling half
        self.bottleneck1_0 = BottleNeckDownSampling(K, K * 4, proj_factor)
        self.bottleneck1_1 = nn.Sequential(BottleNeck(K * 4, K * 4, proj_factor),
            BottleNeck(K * 4, K * 4, proj_factor),
            BottleNeck(K * 4, K * 4, proj_factor),
            BottleNeck(K * 4, K * 4, proj_factor))
        self.bottleneck2_0 = BottleNeckDownSampling(K * 4, K * 8, proj_factor)
        self.bottleneck2_1 = nn.Sequential(
            BottleNeck(K * 8, K * 8, proj_factor, dropoutRate=0.1),
            BottleNeck(K * 8, K * 8, proj_factor, dilation=2),
            BottleNeck(K * 8, K * 8, proj_factor, dropoutRate=0.1, asym=True),
            BottleNeck(K * 8, K * 8, proj_factor, dilation=4),
            BottleNeck(K * 8, K * 8, proj_factor, dropoutRate=0.1),
            BottleNeck(K * 8, K * 8, proj_factor, dilation=8),
            BottleNeck(K * 8, K * 8, proj_factor, dropoutRate=0.1, asym=True),
            BottleNeck(K * 8, K * 8, proj_factor, dilation=16),
        )

        # Middle operations
        self.bottleneck3 = nn.Sequential(
            BottleNeck(K * 8, K * 8, proj_factor, dropoutRate=0.1),
            BottleNeck(K * 8, K * 8, proj_factor, dilation=2),
            BottleNeck(K * 8, K * 8, proj_factor, dropoutRate=0.1, asym=True),
            BottleNeck(K * 8, K * 8, proj_factor, dilation=4),
            BottleNeck(K * 8, K * 8, proj_factor, dropoutRate=0.1),
            BottleNeck(K * 8, K * 8, proj_factor, dilation=8),
            BottleNeck(K * 8, K * 8, proj_factor, dropoutRate=0.1, asym=True),
            BottleNeck(K * 8, K * 4, proj_factor, dilation=16, dilate_last=True),
        )

        # Upsampling half
        self.bottleneck4 = nn.Sequential(
            BottleNeckUpSampling(K * 8, K * 4, proj_factor),
            BottleNeck(K * 4, K * 4, proj_factor, dropoutRate=0.1),
            BottleNeck(K * 4, K, proj_factor, dropoutRate=0.1),
        )
        self.bottleneck5 = nn.Sequential(
            BottleNeckUpSampling(K * 2, K, proj_factor),
            BottleNeck(K, K, proj_factor, dropoutRate=0.1),
        )

        self.final = nn.Sequential(
            conv_block(K, K, kernel_size=3, padding=1, bias=False, stride=1),
            conv_block(K, K, kernel_size=3, padding=1, bias=False, stride=1),
            nn.Conv2d(K, out_dim, kernel_size=1),
        )

        self.transformers = nn.ModuleDict({
            stage: TransformerStack(stage_dims[stage], num_layers=num_layers,
                                     num_heads=num_heads, mlp_dim=stage_dims[stage] * mlp_ratio,
                                     dropout=dropout, use_pos_embed=use_pos_embed)
            for stage in self.transformer_at
        })

        print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) "
              f"transformer_at={sorted(self.transformer_at)} num_layers={num_layers} "
              f"num_heads={num_heads} use_pos_embed={use_pos_embed} kwargs={kwargs}")

    def _apply_transformer(self, stage: str, x: Tensor) -> Tensor:
        if stage in self.transformers:
            x = self.transformers[stage](x)
        return x

    def forward(self, input):
        # Initial operations
        conv_0 = self.conv0(input)
        maxpool_0 = self.maxpool0(input)
        outputInitial = torch.cat((conv_0, maxpool_0), dim=1)

        # Downsampling half
        bn1_0, indices_1 = self.bottleneck1_0(outputInitial)
        bn1_out = self.bottleneck1_1(bn1_0)
        bn1_out = self._apply_transformer("stage1", bn1_out)

        bn2_0, indices_2 = self.bottleneck2_0(bn1_out)
        bn2_out = self.bottleneck2_1(bn2_0)
        bn2_out = self._apply_transformer("stage2", bn2_out)

        # Middle operations
        bn3_out = self.bottleneck3(bn2_out)
        bn3_out = self._apply_transformer("bottleneck", bn3_out)

        # Upsampling half
        bn4_out = self.bottleneck4((bn3_out, indices_2, bn1_out))
        bn4_out = self._apply_transformer("decoder1", bn4_out)

        bn5_out = self.bottleneck5((bn4_out, indices_1, outputInitial))
        bn5_out = self._apply_transformer("decoder2", bn5_out)

        # Final upsampling and convolutions
        interpolated = F.interpolate(bn5_out, mode='nearest', scale_factor=2)
        return self.final(interpolated)

    def init_weights(self, *args, **kwargs):
        self.apply(random_weights_init)