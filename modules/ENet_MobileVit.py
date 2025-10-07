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


# ========== Lightweight Self-Attention Block (MobileViT-style) ==========

class _MobileViTBlock(nn.Module):
    def __init__(self, in_channels, token_dim, num_heads=4, patch_size=(2,2), ff_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.patch_size = patch_size  # (ph, pw)
        if ff_dim is None:
            ff_dim = token_dim * 2

        # projection from input to token-space
        self.conv1 = nn.Conv2d(in_channels, token_dim, kernel_size=1, stride=1, padding=0, bias=False)
        # projection back from token space
        self.conv2 = nn.Conv2d(token_dim, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

        # attention in token space
        self.qkv = nn.Linear(token_dim, token_dim * 3)
        self.attn_proj = nn.Linear(token_dim, token_dim)

        # feedforward in token space
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, token_dim)
        )

        # normalization in token space
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        x_loc = self.conv1(x)  # (B, token_dim, H, W)
        ph, pw = self.patch_size
        # ensure divisibility
        assert H % ph == 0 and W % pw == 0, f"Feature map size {H}×{W} not divisible by patch {ph}×{pw}"
        Nh = H // ph
        Nw = W // pw

        # reshape into patches
        xp = x_loc.view(B, self.token_dim, Nh, ph, Nw, pw)
        xp = xp.permute(0, 2, 4, 3, 5, 1).contiguous()
        xp = xp.view(B, Nh * Nw, ph * pw, self.token_dim)

        # compute token representation
        xp_mean = xp.mean(dim=2)  # (B, Nh*Nw, token_dim)

        # attention: q, k, v
        qkv = self.qkv(xp_mean)  # (B, Ntokens, 3 * token_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        d = self.token_dim // self.num_heads
        q = q.view(B, -1, self.num_heads, d).permute(0, 2, 1, 3)
        k = k.view(B, -1, self.num_heads, d).permute(0, 2, 1, 3)
        v = v.view(B, -1, self.num_heads, d).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)  # (B, heads, Ntokens, d)

        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, -1, self.token_dim)
        attn_out = self.attn_proj(attn_out)
        attn_out = attn_out + xp_mean
        attn_out = self.norm1(attn_out)

        ff = self.ffn(attn_out)
        ff_out = attn_out + ff
        ff_out = self.norm2(ff_out)

        # expand tokens back to patches
        out_tokens = ff_out.unsqueeze(2).repeat(1, 1, ph * pw, 1)
        out = out_tokens.view(B, Nh, Nw, ph, pw, self.token_dim)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, self.token_dim, H, W)

        out = self.conv2(out)
        return x + out  # residual



# ========== ENet Architecture (with attention hook) ==========


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


class ENet(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, **kwargs):
                super().__init__()
                F: int = kwargs["factor"] if "factor" in kwargs else 4  # Projecting factor
                K: int = kwargs["kernels"] if "kernels" in kwargs else 16  # n_kernels

                # from models.enet import (BottleNeck,
                #                          BottleNeckDownSampling,
                #                          BottleNeckUpSampling,
                #                          conv_block)

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
                # attention block: place at deepest encoder stage
                # Here, the deepest output channel is 128
                self._use_mvit = True
                ch = K * 8
                self._mvit = _MobileViTBlock(in_channels=ch,
                                        token_dim=ch // 2,
                                        num_heads=4,
                                        patch_size=(2,2))

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

        def forward(self, input):
                # Initial operations
                conv_0 = self.conv0(input)
                maxpool_0 = self.maxpool0(input)
                outputInitial = torch.cat((conv_0, maxpool_0), dim=1)

                # Downsampling half
                bn1_0, indices_1 = self.bottleneck1_0(outputInitial)
                bn1_out = self.bottleneck1_1(bn1_0)
                bn2_0, indices_2 = self.bottleneck2_0(bn1_out)
                bn2_out = self.bottleneck2_1(bn2_0)

                # apply attention (if enabled)
                if self._use_mvit:
                    bn2_out = self._mvit(bn2_out)

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