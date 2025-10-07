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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def random_weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


def conv_block(in_dim, out_dim, **kwconv):
        return nn.Sequential(nn.Conv2d(in_dim, out_dim, **kwconv),
                             nn.BatchNorm2d(out_dim),
                             nn.PReLU())


def conv_block_asym(in_dim, out_dim, *, kernel_size: int):
        return nn.Sequential(nn.Conv2d(in_dim, out_dim,
                                       kernel_size=(kernel_size, 1),
                                       padding=(2, 0)),
                             nn.Conv2d(out_dim, out_dim,
                                       kernel_size=(1, kernel_size),
                                       padding=(0, 2)),
                             nn.BatchNorm2d(out_dim),
                             nn.PReLU())


# ======= New: lightweight attention blocks (kept small & local) =======

class InSliceSelfAttention2D(nn.Module):
    """
    In-slice (self) attention on the center slice.
    Uses tokenization by strided conv -> MHA -> upsample -> 1x1 proj + residual.
    """
    def __init__(self, in_ch: int, embed_ch: int, heads: int = 4, downsample: int = 8):
        super().__init__()
        self.down = downsample
        self.qkv_embed = nn.Conv2d(in_ch, embed_ch, kernel_size=3, padding=1, stride=downsample)
        self.proj_out = nn.Conv2d(embed_ch, in_ch, kernel_size=1)
        self.mha = nn.MultiheadAttention(embed_ch, num_heads=heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        t = self.qkv_embed(x)  # (B, E, h, w)
        B_, E, h, w = t.shape
        tokens = t.flatten(2).transpose(1, 2)  # (B, N, E)
        # Self-attention within slice
        attn_out, _ = self.mha(tokens, tokens, tokens)  # (B, N, E)
        attn_out = attn_out.transpose(1, 2).reshape(B, E, h, w)
        attn_out = F.interpolate(attn_out, size=(H, W), mode="bilinear", align_corners=False)
        out = self.proj_out(attn_out)
        return x + out  # residual


class CrossSliceAttention2D(nn.Module):
    """
    Cross-slice attention: center queries neighbors (all other slices).
    center: (B, C, H, W), neighbors: (B, Nn, C, H, W) with Nn=S-1
    """
    def __init__(self, in_ch: int, embed_ch: int, heads: int = 4, downsample: int = 8):
        super().__init__()
        self.down = downsample
        self.q_embed = nn.Conv2d(in_ch, embed_ch, kernel_size=3, padding=1, stride=downsample)
        self.kv_embed = nn.Conv2d(in_ch, embed_ch, kernel_size=3, padding=1, stride=downsample)
        self.mha = nn.MultiheadAttention(embed_ch, num_heads=heads, batch_first=True)
        self.proj_out = nn.Conv2d(embed_ch, in_ch, kernel_size=1)

    def forward(self, center: Tensor, neighbors: Tensor) -> Tensor:
        # center: (B,C,H,W) ; neighbors: (B, Nn, C, H, W)
        B, C, H, W = center.shape
        Nn = neighbors.shape[1] if neighbors.ndim == 5 else 0
        q = self.q_embed(center)                    # (B, E, h, w)
        h, w = q.shape[2:]
        q_tok = q.flatten(2).transpose(1, 2)        # (B, Nhw, E)

        if Nn == 0:
            # no neighbors -> identity
            return center

        kv = self.kv_embed(neighbors.reshape(-1, C, H, W))  # (B*Nn, E, h, w)
        kv_tok = kv.flatten(2).transpose(1, 2)              # (B*Nn, Nhw, E)
        # merge neighbor tokens along sequence
        kv_tok = kv_tok.reshape(B, Nn * (h * w), -1)        # (B, Nn*Nhw, E)

        attn_out, _ = self.mha(q_tok, kv_tok, kv_tok)       # (B, Nhw, E)
        attn_out = attn_out.transpose(1, 2).reshape(B, -1, h, w)
        attn_out = F.interpolate(attn_out, size=(H, W), mode="bilinear", align_corners=False)
        out = self.proj_out(attn_out)
        return center + out  # residual


class CSAFusion(nn.Module):
    """
    Combines in-slice self-attn and cross-slice attention, then projects back to in_dim.
    """
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, downsample: int = 8, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(in_dim, 32)
        self.intra = InSliceSelfAttention2D(in_dim, hidden, heads=heads, downsample=downsample)
        self.cross = CrossSliceAttention2D(in_dim, hidden, heads=heads, downsample=downsample)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.PReLU(),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=True)
        )

    def forward(self, center: Tensor, neighbors: Tensor) -> Tensor:
        x = self.intra(center)
        x = self.cross(x, neighbors)
        return self.fuse(x)


# ======================= Original building blocks =======================


class BottleNeck(nn.Module):
        def __init__(self, in_dim, out_dim, projectionFactor,
                     *, dropoutRate=0.01, dilation=1,
                     asym: bool = False, dilate_last: bool = False):
                super().__init__()
                self.in_dim = in_dim
                self.out_dim = out_dim
                mid_dim: int = in_dim // projectionFactor

                # Main branch

                # Secondary branch
                self.block0 = conv_block(in_dim, mid_dim, kernel_size=1)

                if not asym:
                        self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=dilation, dilation=dilation)
                else:
                        self.block1 = conv_block_asym(mid_dim, mid_dim, kernel_size=5)

                self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

                self.do = nn.Dropout(p=dropoutRate)
                self.PReLU_out = nn.PReLU()

                if in_dim > out_dim:
                        self.conv_out = conv_block(in_dim, out_dim, kernel_size=1)
                elif dilate_last:
                        self.conv_out = conv_block(in_dim, out_dim, kernel_size=3, padding=1)
                else:
                        self.conv_out = nn.Identity()

        def forward(self, in_) -> Tensor:
                # Main branch
                # Secondary branch
                b0 = self.block0(in_)
                b1 = self.block1(b0)
                b2 = self.block2(b1)
                do = self.do(b2)

                output = self.PReLU_out(self.conv_out(in_) + do)

                return output


class BottleNeckDownSampling(nn.Module):
        def __init__(self, in_dim, out_dim, projectionFactor):
                super().__init__()
                mid_dim: int = in_dim // projectionFactor

                # Main branch
                self.maxpool0 = nn.MaxPool2d(2, return_indices=True)

                # Secondary branch
                self.block0 = conv_block(in_dim, mid_dim, kernel_size=2, padding=0, stride=2)
                self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
                self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

                # Regularizer
                self.do = nn.Dropout(p=0.01)
                self.PReLU = nn.PReLU()

                # Out

        def forward(self, in_) -> tuple[Tensor, Tensor]:
                # Main branch
                maxpool_output, indices = self.maxpool0(in_)

                # Secondary branch
                b0 = self.block0(in_)
                b1 = self.block1(b0)
                b2 = self.block2(b1)
                do = self.do(b2)

                _, c, _, _ = maxpool_output.shape
                output = do
                output[:, :c, :, :] += maxpool_output

                final_output = self.PReLU(output)

                return final_output, indices


class BottleNeckUpSampling(nn.Module):
        def __init__(self, in_dim, out_dim, projectionFactor):
                super().__init__()
                mid_dim: int = in_dim // projectionFactor

                # Main branch
                self.unpool = nn.MaxUnpool2d(2)

                # Secondary branch
                self.block0 = conv_block(in_dim, mid_dim, kernel_size=3, padding=1)
                self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
                self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

                # Regularizer
                self.do = nn.Dropout(p=0.01)
                self.PReLU = nn.PReLU()

                # Out

        def forward(self, args) -> Tensor:
                # nn.Sequential cannot handle multiple parameters:
                in_, indices, skip = args

                # Main branch
                up = self.unpool(in_, indices)

                # Secondary branch
                b0 = self.block0(torch.cat((up, skip), dim=1))
                b1 = self.block1(b0)
                b2 = self.block2(b1)
                do = self.do(b2)

                output = self.PReLU(up + do)

                return output


class ENet_25d(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, **kwargs):
                super().__init__()
                F: int = kwargs["factor"] if "factor" in kwargs else 4  # Projecting factor
                K: int = kwargs["kernels"] if "kernels" in kwargs else 16  # n_kernels

                # from models.enet import (BottleNeck,
                #                          BottleNeckDownSampling,
                #                          BottleNeckUpSampling,
                #                          conv_block)

                # New: 2.5D switches (maps to CLI --2_5d)
                self.two_point_five_d: bool = bool(kwargs.get("two_point_five_d", kwargs.get("use_2_5d", False)))
                self.num_slices: int = int(kwargs.get("num_slices", 3))
                assert self.num_slices % 2 == 1, "num_slices must be odd (center slice + symmetric neighbors)."
                self.attn_heads: int = int(kwargs.get("attn_heads", 4))
                self.attn_down: int = int(kwargs.get("attn_downsample", 8))

                # Optional CSA fusion (only constructed if used)
                if self.two_point_five_d:
                    # We fuse back to in_dim so the rest of the network is untouched
                    self.csa_fusion = CSAFusion(
                        in_dim=in_dim,
                        out_dim=in_dim,
                        heads=self.attn_heads,
                        downsample=self.attn_down,
                        hidden=max(in_dim, K)
                    )

                # Initial operations
                self.conv0 = nn.Conv2d(in_dim, K - 1, kernel_size=3, stride=2, padding=1)
                self.maxpool0 = nn.MaxPool2d(2, return_indices=False, ceil_mode=False)

                # Downsampling half
                self.bottleneck1_0 = BottleNeckDownSampling(K, K * 4, F)
                self.bottleneck1_1 = nn.Sequential(BottleNeck(K * 4, K * 4, F),
                                                   BottleNeck(K * 4, K * 4, F),
                                                   BottleNeck(K * 4, K * 4, F),
                                                   BottleNeck(K * 4, K * 4, F))
                self.bottleneck2_0 = BottleNeckDownSampling(K * 4, K * 8, F)
                self.bottleneck2_1 = nn.Sequential(BottleNeck(K * 8, K * 8, F, dropoutRate=0.1),
                                                   BottleNeck(K * 8, K * 8, F, dilation=2),
                                                   BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True),
                                                   BottleNeck(K * 8, K * 8, F, dilation=4),
                                                   BottleNeck(K * 8, K * 8, F, dropoutRate=0.1),
                                                   BottleNeck(K * 8, K * 8, F, dilation=8),
                                                   BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True),
                                                   BottleNeck(K * 8, K * 8, F, dilation=16))

                # Middle operations
                self.bottleneck3 = nn.Sequential(BottleNeck(K * 8, K * 8, F, dropoutRate=0.1),
                                                 BottleNeck(K * 8, K * 8, F, dilation=2),
                                                 BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True),
                                                 BottleNeck(K * 8, K * 8, F, dilation=4),
                                                 BottleNeck(K * 8, K * 8, F, dropoutRate=0.1),
                                                 BottleNeck(K * 8, K * 8, F, dilation=8),
                                                 BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True),
                                                 BottleNeck(K * 8, K * 4, F, dilation=16, dilate_last=True))

                # Upsampling half
                self.bottleneck4 = nn.Sequential(BottleNeckUpSampling(K * 8, K * 4, F),
                                                 BottleNeck(K * 4, K * 4, F, dropoutRate=0.1),
                                                 BottleNeck(K * 4, K, F, dropoutRate=0.1))
                self.bottleneck5 = nn.Sequential(BottleNeckUpSampling(K * 2, K, F),
                                                 BottleNeck(K, K, F, dropoutRate=0.1))

                # Final upsampling and covolutions
                self.final = nn.Sequential(conv_block(K, K, kernel_size=3, padding=1, bias=False, stride=1),
                                           conv_block(K, K, kernel_size=3, padding=1, bias=False, stride=1),
                                           nn.Conv2d(K, out_dim, kernel_size=1))

                print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with {kwargs}")

        def _apply_csa(self, x: Tensor) -> Tensor:
            """
            x: (B, C_total, H, W) with C_total = in_dim * S
            returns: (B, in_dim, H, W) fused center slice features
            """
            B, C_total, H, W = x.shape
            in_dim = C_total // self.num_slices
            assert C_total == in_dim * self.num_slices, \
                f"Channel count {C_total} not divisible by num_slices={self.num_slices} (expected C_total=C*num_slices)."
            S = self.num_slices
            C = in_dim

            xs = x.view(B, S, C, H, W)
            c_idx = S // 2
            center = xs[:, c_idx, :, :, :]                       # (B,C,H,W)
            neighbors = torch.cat([xs[:, :c_idx], xs[:, c_idx + 1:]], dim=1)  # (B, S-1, C, H, W)
            fused = self.csa_fusion(center, neighbors)
            return fused


        def forward(self, input):
                # If 2.5D is ON, fuse center slice features via CSA, then proceed as usual
                # INTUITION: The goal of 2.5D is to inject through-plane context that a plain 2D encoder doesn’t see. 
                # Doing that before the first stride means all later stages operate on features that already carry inter-slice cues. 
                # Empirically (and intuitively), most of the benefit comes from this front-loaded fusion.
                if self.two_point_five_d:
                    input = self._apply_csa(input)

                # Initial operations
                conv_0 = self.conv0(input)
                maxpool_0 = self.maxpool0(input)
                outputInitial = torch.cat((conv_0, maxpool_0), dim=1)

                # Downsampling half
                bn1_0, indices_1 = self.bottleneck1_0(outputInitial)
                bn1_out = self.bottleneck1_1(bn1_0)
                bn2_0, indices_2 = self.bottleneck2_0(bn1_out)
                bn2_out = self.bottleneck2_1(bn2_0)

                # Middle operations
                bn3_out = self.bottleneck3(bn2_out)

                # Upsampling half
                bn4_out = self.bottleneck4((bn3_out, indices_2, bn1_out))
                bn5_out = self.bottleneck5((bn4_out, indices_1, outputInitial))

                # Final upsampling and covolutions
                interpolated = F.interpolate(bn5_out, mode='nearest', scale_factor=2)
                return self.final(interpolated)

        def init_weights(self, *args, **kwargs):
                self.apply(random_weights_init)