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

from modules.attention import ECALayer, CBAM, AttentionGate

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


class BottleNeck(nn.Module):
        def __init__(self, in_dim, out_dim, projectionFactor,
                     *, dropoutRate=0.01, dilation=1,
                     asym: bool = False, dilate_last: bool = False,
                     alter_enet: bool = False, attn: str|None=None):
                super().__init__()
                self.in_dim = in_dim
                self.out_dim = out_dim
                mid_dim: int = in_dim // projectionFactor

                # NOTE: Enet changes ###
                # Attention modules
                self.attn = None
                if attn == "eca":
                        self.attn = ECALayer(out_dim)
                elif attn == "cbam":
                        self.attn = CBAM(out_dim)

                # Paper: SpatialDropout (Dropout2d) -> https://arxiv.org/abs/1411.4280
                self.do = nn.Dropout2d(p=dropoutRate) if alter_enet else nn.Dropout(p=dropoutRate)
                ########################

                # Main branch

                # Secondary branch
                self.block0 = conv_block(in_dim, mid_dim, kernel_size=1)

                if not asym:
                        self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=dilation, dilation=dilation)
                else:
                        self.block1 = conv_block_asym(mid_dim, mid_dim, kernel_size=5)

                self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

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
                if self.attn is not None:
                        do = self.attn(do)

                output = self.PReLU_out(self.conv_out(in_) + do)

                return output


class BottleNeckDownSampling(nn.Module):
        def __init__(self, in_dim, out_dim, projectionFactor, *, alter_enet: bool = False, attn: str|None=None):
                super().__init__()
                self.attn = ECALayer(out_dim) if attn == "eca" else (CBAM(out_dim) if attn == "cbam" else None)
                mid_dim: int = in_dim // projectionFactor

                # Main branch
                self.maxpool0 = nn.MaxPool2d(2, return_indices=True)

                # Secondary branch
                self.block0 = conv_block(in_dim, mid_dim, kernel_size=2, padding=0, stride=2)
                self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
                self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

                # Regularizer (paper-style SpatialDropout toggle)
                self.do = nn.Dropout2d(p=0.01) if alter_enet else nn.Dropout(p=0.01)
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
                if self.attn is not None:
                        do = self.attn(do)

                _, c, _, _ = maxpool_output.shape
                output = do
                output[:, :c, :, :] += maxpool_output

                final_output = self.PReLU(output)

                return final_output, indices


class BottleNeckUpSampling(nn.Module):
        def __init__(self, in_dim, out_dim, projectionFactor, *, alter_enet: bool = False, attn: str|None=None,
                     skip_attention: bool = False, skip_channels: int = 0, gate_channels: int = 0):
                super().__init__()
                self.attn = ECALayer(out_dim) if attn == "eca" else (CBAM(out_dim) if attn == "cbam" else None)
                self.skip_gate = AttentionGate(skip_channels, gate_channels, inter=max(8, skip_channels//2)) if skip_attention else None
                mid_dim: int = in_dim // projectionFactor

                # Main branch
                self.unpool = nn.MaxUnpool2d(2)

                # Secondary branch
                self.block0 = conv_block(in_dim, mid_dim, kernel_size=3, padding=1)
                self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
                self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

                # Regularizer (paper-style SpatialDropout toggle)
                self.do = nn.Dropout2d(p=0.01) if alter_enet else nn.Dropout(p=0.01)
                self.PReLU = nn.PReLU()

                # Out

        def forward(self, args) -> Tensor:
                # nn.Sequential cannot handle multiple parameters:
                in_, indices, skip = args

                # Main branch
                up = self.unpool(in_, indices)
                if self.skip_gate is not None:
                        skip = self.skip_gate(skip, up)

                # Secondary branch
                b0 = self.block0(torch.cat((up, skip), dim=1))
                b1 = self.block1(b0)
                b2 = self.block2(b1)
                do = self.do(b2)
                if self.attn is not None:
                        do = self.attn(do)
                output = self.PReLU(up + do)

                return output


class ENet(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, **kwargs):
                super().__init__()
                F: int = kwargs["factor"] if "factor" in kwargs else 4  # Projecting factor
                K: int = kwargs["kernels"] if "kernels" in kwargs else 16  # n_kernels
                alter_enet = bool(kwargs.get("alter_enet", False))
                attn = kwargs.get("attn", None) # "eca", "cbam", or None
                skip_attention = bool(kwargs.get("skip_attention", False))

                # Initial operations
                self.conv0 = nn.Conv2d(in_dim, K - 1, kernel_size=3, stride=2, padding=1)
                self.maxpool0 = nn.MaxPool2d(2, return_indices=False, ceil_mode=False)

                # Downsampling half
                self.bottleneck1_0 = BottleNeckDownSampling(K, K * 4, F, alter_enet=alter_enet, attn=attn)
                self.bottleneck1_1 = nn.Sequential(BottleNeck(K * 4, K * 4, F, alter_enet=alter_enet, attn=attn),
                                                   BottleNeck(K * 4, K * 4, F, alter_enet=alter_enet, attn=attn),
                                                   BottleNeck(K * 4, K * 4, F, alter_enet=alter_enet, attn=attn),
                                                   BottleNeck(K * 4, K * 4, F, alter_enet=alter_enet, attn=attn))
                self.bottleneck2_0 = BottleNeckDownSampling(K * 4, K * 8, F, alter_enet=alter_enet, attn=attn)
                self.bottleneck2_1 = nn.Sequential(BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, alter_enet=alter_enet, attn=attn),
                                                   BottleNeck(K * 8, K * 8, F, dilation=2, alter_enet=alter_enet, attn=attn),
                                                   BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, alter_enet=alter_enet, attn=attn),
                                                   BottleNeck(K * 8, K * 8, F, dilation=4, alter_enet=alter_enet, attn=attn),
                                                   BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, alter_enet=alter_enet, attn=attn),
                                                   BottleNeck(K * 8, K * 8, F, dilation=8, alter_enet=alter_enet, attn=attn),
                                                   BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, alter_enet=alter_enet, attn=attn),
                                                   BottleNeck(K * 8, K * 8, F, dilation=16, alter_enet=alter_enet, attn=attn))

                # Middle operations
                self.bottleneck3 = nn.Sequential(BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, alter_enet=alter_enet, attn=attn),
                                                 BottleNeck(K * 8, K * 8, F, dilation=2, alter_enet=alter_enet, attn=attn),
                                                 BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, alter_enet=alter_enet, attn=attn),
                                                 BottleNeck(K * 8, K * 8, F, dilation=4, alter_enet=alter_enet, attn=attn),
                                                 BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, alter_enet=alter_enet, attn=attn),
                                                 BottleNeck(K * 8, K * 8, F, dilation=8, alter_enet=alter_enet, attn=attn),
                                                 BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, alter_enet=alter_enet, attn=attn),
                                                 BottleNeck(K * 8, K * 4, F, dilation=16, dilate_last=True, alter_enet=alter_enet, attn=attn))

                # Upsampling half
                self.bottleneck4 = nn.Sequential(BottleNeckUpSampling(K*8, K*4, F, alter_enet=alter_enet, attn=attn,
                                 skip_attention=skip_attention, skip_channels=K*4, gate_channels=K*8),
                                                 BottleNeck(K * 4, K * 4, F, dropoutRate=0.1, alter_enet=alter_enet, attn=attn),
                                                 BottleNeck(K * 4, K, F, dropoutRate=0.1, alter_enet=alter_enet, attn=attn))
                self.bottleneck5 = nn.Sequential(BottleNeckUpSampling(K*2, K, F, alter_enet=alter_enet, attn=attn,
                                 skip_attention=skip_attention, skip_channels=K, gate_channels=K*4),
                                                 BottleNeck(K, K, F, dropoutRate=0.1, alter_enet=alter_enet, attn=attn))

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
