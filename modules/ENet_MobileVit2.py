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
import math


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


# ========== Improved MobileViT Block ==========

class _MobileViTBlock(nn.Module):
    """
    MobileViT-style block with learnable 2D relative position bias defined over
    the *patch grid* (Nh x Nw). Uses residual scaling and stochastic depth.
    """
    def __init__(
        self,
        in_channels: int,
        token_dim: int | None = None,
        num_heads: int = 4,
        patch_size: tuple[int, int] = (4, 4),
        ff_dim: int | None = None,
        drop_path_rate: float = 0.1,
        max_grid: tuple[int, int] = (8, 8)  # <-- set to the largest (Nh, Nw) you will see
    ):
        super().__init__()
        self.in_channels = in_channels
        self.token_dim = token_dim if token_dim is not None else int(in_channels * 3 / 4)
        self.num_heads = num_heads
        self.patch_size = patch_size  # (ph, pw)
        self.drop_path_rate = drop_path_rate
        self.max_grid = max_grid

        if ff_dim is None:
            ff_dim = self.token_dim * 2

        # local conv -> token space
        self.conv1 = nn.Conv2d(in_channels, self.token_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(self.token_dim, in_channels, kernel_size=1, bias=False)

        # transformer in token space
        self.qkv = nn.Linear(self.token_dim, self.token_dim * 3, bias=True)
        self.attn_proj = nn.Linear(self.token_dim, self.token_dim, bias=True)

        self.ffn = nn.Sequential(
            nn.Linear(self.token_dim, ff_dim, bias=True),
            nn.GELU(),
            nn.Linear(ff_dim, self.token_dim, bias=True),
        )

        self.norm1 = nn.LayerNorm(self.token_dim)
        self.norm2 = nn.LayerNorm(self.token_dim)

        # residual scaling factor (start near 0 to stabilize)
        self.alpha = nn.Parameter(torch.zeros(1))

        # --- relative position bias over patch grid (Nh, Nw) ---
        Nh_max, Nw_max = self.max_grid
        table_size = (2 * Nh_max - 1) * (2 * Nw_max - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, self.num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def drop_path(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_path_rate == 0.0:
            return x
        keep_prob = 1.0 - self.drop_path_rate
        shape = [x.shape[0]] + [1] * (x.dim() - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return (x / keep_prob) * binary_tensor

    def _relative_position_bias(self, Nh: int, Nw: int, device, dtype) -> Tensor:
        """
        Build a (1, heads, N, N) bias tensor for current grid (Nh,Nw) by indexing
        into a table parameterized for max_grid.
        """
        Nh_max, Nw_max = self.max_grid
        if Nh > Nh_max or Nw > Nw_max:
            raise ValueError(
                f"Current patch grid ({Nh}x{Nw}) exceeds max_grid {self.max_grid}. "
                f"Increase max_grid accordingly."
            )

        # coords in current grid
        coords_h = torch.arange(Nh, device=device)
        coords_w = torch.arange(Nw, device=device)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, Nh, Nw)
        coords_flat = coords.reshape(2, -1)                                      # (2, N)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]                  # (2, N, N)
        rel = rel.permute(1, 2, 0).contiguous()                                  # (N, N, 2)

        # shift into max-grid index space and flatten to single index
        rel[:, :, 0] += Nh_max - 1
        rel[:, :, 1] += Nw_max - 1
        rel[:, :, 0] *= (2 * Nw_max - 1)
        rpb_index = rel.sum(-1).view(-1)                                         # (N*N,)

        bias = self.relative_position_bias_table[rpb_index]                       # (N*N, heads)
        bias = bias.view(Nh * Nw, Nh * Nw, self.num_heads).permute(2, 0, 1)      # (heads, N, N)
        return bias.unsqueeze(0).to(dtype=dtype)                                  # (1, heads, N, N)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        assert H % ph == 0 and W % pw == 0, \
            f"Feature map {H}x{W} not divisible by patch size {ph}x{pw}"

        Nh, Nw = H // ph, W // pw
        Npatch = Nh * Nw
        head_dim = self.token_dim // self.num_heads
        scale = 1.0 / math.sqrt(head_dim)

        # local -> token space
        x_loc = self.conv1(x)  # (B, token_dim, H, W)

        # reshape to patches and average within each patch
        xp = x_loc.view(B, self.token_dim, Nh, ph, Nw, pw)
        xp = xp.permute(0, 2, 4, 3, 5, 1).contiguous()         # (B, Nh, Nw, ph, pw, token_dim)
        xp = xp.view(B, Npatch, ph * pw, self.token_dim)       # (B, N, A, D)
        tokens = xp.mean(dim=2)                                # (B, N, D)

        # attention
        t = self.norm1(tokens)
        qkv = self.qkv(t).reshape(B, Npatch, 3, self.num_heads, head_dim)
        q, k, v = qkv.unbind(dim=2)                            # each: (B, N, heads, head_dim)
        q = q.permute(0, 2, 1, 3)                              # (B, heads, N, d)
        k = k.permute(0, 2, 1, 3)                              # (B, heads, N, d)
        v = v.permute(0, 2, 1, 3)                              # (B, heads, N, d)

        attn = (q @ k.transpose(-2, -1)) * scale               # (B, heads, N, N)
        bias = self._relative_position_bias(Nh, Nw, device=x.device, dtype=attn.dtype)
        attn = attn + bias
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=self.drop_path_rate, training=self.training)

        out = attn @ v                                         # (B, heads, N, d)
        out = out.permute(0, 2, 1, 3).reshape(B, Npatch, self.token_dim)
        out = self.attn_proj(out)

        # residual + stochastic depth + residual scaling
        out = tokens + self.drop_path(self.alpha * out)

        # MLP
        m = self.ffn(self.norm2(out))
        out = out + self.drop_path(m)

        # expand tokens back to spatial
        out_tokens = out.unsqueeze(2).expand(B, Npatch, ph * pw, self.token_dim)  # (B, N, A, D)
        out = out_tokens.view(B, Nh, Nw, ph, pw, self.token_dim)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, self.token_dim, H, W)

        out = self.conv2(out)                                 # (B, C, H, W)
        return x + out

# ========== Modified ENet forward (snippet) ==========


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
                # Choose patch_size=(4,4), token_dim = 3/4 * ch
                self._mvit = _MobileViTBlock(in_channels=ch,
                                        token_dim=int(ch * 3 / 4),
                                        num_heads=4,
                                        patch_size=(4, 4),
                                        drop_path_rate=0.1)

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
                    mv = self._mvit(bn2_out)
                    bn2_out = bn2_out + mv  # residual fusion

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