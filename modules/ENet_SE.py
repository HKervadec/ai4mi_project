#!/usr/bin/env python3.10

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
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, **kwconv),
        nn.BatchNorm2d(out_dim),
        nn.PReLU()
    )


def conv_block_asym(in_dim, out_dim, *, kernel_size: int):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim,
                  kernel_size=(kernel_size, 1),
                  padding=(kernel_size // 2, 0)),
        nn.Conv2d(out_dim, out_dim,
                  kernel_size=(1, kernel_size),
                  padding=(0, kernel_size // 2)),
        nn.BatchNorm2d(out_dim),
        nn.PReLU()
    )


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block (channel attention)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BottleNeck(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor,
                 *, dropoutRate=0.01, dilation=1,
                 asym: bool = False, dilate_last: bool = False,
                 use_se: bool = True, se_reduction: int = 16, debug: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_se = use_se
        self.debug = debug

        mid_dim = in_dim // projectionFactor

        self.block0 = conv_block(in_dim, mid_dim, kernel_size=1)
        if not asym:
            self.block1 = conv_block(mid_dim, mid_dim,
                                     kernel_size=3, padding=dilation, dilation=dilation)
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

        if use_se:
            self.se = SEBlock(out_dim, reduction=se_reduction)
        else:
            self.se = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        if self.debug:
            print(f"  BottleNeck: x.shape = {tuple(x.shape)}, in_dim = {self.in_dim}, out_dim = {self.out_dim}")
        b0 = self.block0(x)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        shortcut = self.conv_out(x)
        summed = shortcut + do
        act = self.PReLU_out(summed)

        out = self.se(act)
        return out


class BottleNeckDownSampling(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor,
                 use_se: bool = True, se_reduction: int = 16, debug: bool = False):
        super().__init__()
        self.use_se = use_se
        self.debug = debug

        mid_dim = in_dim // projectionFactor
        self.maxpool0 = nn.MaxPool2d(2, return_indices=True)
        self.block0 = conv_block(in_dim, mid_dim, kernel_size=2, padding=0, stride=2)
        self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
        self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)
        self.do = nn.Dropout(p=0.01)
        self.PReLU = nn.PReLU()

        if use_se:
            self.se = SEBlock(out_dim, reduction=se_reduction)
        else:
            self.se = nn.Identity()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.debug:
            print(f"Down: x.shape = {tuple(x.shape)}")
        mp, idx = self.maxpool0(x)
        b0 = self.block0(x)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        _, c, _, _ = mp.shape
        combined = do
        combined[:, :c, :, :] += mp
        acted = self.PReLU(combined)
        out = self.se(acted)
        return out, idx


class BottleNeckUpSampling(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor,
                 use_se: bool = True, se_reduction: int = 16, debug: bool = False):
        super().__init__()
        self.use_se = use_se
        self.debug = debug

        mid_dim = in_dim // projectionFactor
        self.unpool = nn.MaxUnpool2d(2)
        self.block0 = conv_block(in_dim, mid_dim, kernel_size=3, padding=1)
        self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
        self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)
        self.do = nn.Dropout(p=0.01)
        self.PReLU = nn.PReLU()

        if use_se:
            self.se = SEBlock(out_dim, reduction=se_reduction)
        else:
            self.se = nn.Identity()

    def forward(self, args) -> Tensor:
        x, idx, skip = args
        up = self.unpool(x, idx)
        cat = torch.cat((up, skip), dim=1)
        if self.debug:
            print(f"Up: up.shape = {tuple(up.shape)}, skip.shape = {tuple(skip.shape)}, cat.shape = {tuple(cat.shape)}")
        b0 = self.block0(cat)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        summed = up + do
        act = self.PReLU(summed)
        out = self.se(act)
        return out


class ENet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *,
                 factor: int = 4, kernels: int = 16,
                 use_se: bool = True, se_reduction: int = 16, debug: bool = False):
        super().__init__()
        F = factor
        K = kernels
        self.use_se = use_se
        self.se_reduction = se_reduction
        self.debug = debug

        self.conv0 = nn.Conv2d(in_dim, K - 1, kernel_size=3, stride=2, padding=1)
        self.maxpool0 = nn.MaxPool2d(2, return_indices=False, ceil_mode=False)

        # Encoder / downsampling
        self.bottleneck1_0 = BottleNeckDownSampling(K, K * 4, F, use_se=False, debug=debug)
        self.bottleneck1_1 = nn.Sequential(
            BottleNeck(K * 4, K * 4, F, use_se=False, debug=debug),
            BottleNeck(K * 4, K * 4, F, use_se=False, debug=debug),
            BottleNeck(K * 4, K * 4, F, use_se=False, debug=debug),
            BottleNeck(K * 4, K * 4, F, use_se=False, debug=debug),
        )
        self.bottleneck2_0 = BottleNeckDownSampling(K * 4, K * 8, F, use_se=False, debug=debug)
        self.bottleneck2_1 = nn.Sequential(
            BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, use_se=False, debug=debug),
            BottleNeck(K * 8, K * 8, F, dilation=2, use_se=True, se_reduction=se_reduction, debug=debug),
            BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, use_se=True, se_reduction=se_reduction, debug=debug),
            BottleNeck(K * 8, K * 8, F, dilation=4, use_se=False, debug=debug),
            BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, use_se=True, se_reduction=se_reduction, debug=debug),
            BottleNeck(K * 8, K * 8, F, dilation=8, use_se=False, debug=debug),
            BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, use_se=True, se_reduction=se_reduction, debug=debug),
            BottleNeck(K * 8, K * 8, F, use_se=False, debug=debug),
        )

        # Middle: keep full 8K until reduction
        self.bottleneck3 = nn.Sequential(
            BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, use_se=use_se, se_reduction=se_reduction, debug=debug),
            BottleNeck(K * 8, K * 8, F, dilation=2, use_se=use_se, se_reduction=se_reduction, debug=debug),
            BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, use_se=use_se, se_reduction=se_reduction, debug=debug),
            BottleNeck(K * 8, K * 8, F, dilation=4, use_se=use_se, se_reduction=se_reduction, debug=debug),
            BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, use_se=use_se, se_reduction=se_reduction, debug=debug),
            BottleNeck(K * 8, K * 8, F, dilation=8, use_se=use_se, se_reduction=se_reduction, debug=debug),
            BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, use_se=use_se, se_reduction=se_reduction, debug=debug),
            # Single reduction block
            BottleNeck(K * 8, K * 4, F, dilation=16, dilate_last=True, use_se=use_se, se_reduction=se_reduction, debug=debug),
        )

        # Decoder / upsampling: expect 4K input now
        # The first upsample block must match skip + up channels
        self.bottleneck4 = nn.Sequential(
            BottleNeckUpSampling(K * 4 + K * 4, K * 4, F, use_se=False, debug=debug),
            BottleNeck(K * 4, K * 4, F, dropoutRate=0.1, use_se=False, debug=debug),
            BottleNeck(K * 4, K, F, dropoutRate=0.1, use_se=False, debug=debug),
        )
        self.bottleneck5 = nn.Sequential(
            BottleNeckUpSampling(K * 2 + K * 4, K, F, use_se=False, debug=debug),
            BottleNeck(K, K, F, dropoutRate=0.1, use_se=False, debug=debug),
        )

        self.final = nn.Sequential(
            conv_block(K, K, kernel_size=3, padding=1, bias=False, stride=1),
            conv_block(K, K, kernel_size=3, padding=1, bias=False, stride=1),
            nn.Conv2d(K, out_dim, kernel_size=1),
        )

        print(f"> Initialized SE-ENet (final) ({in_dim}->{out_dim}), debug={debug}")

    def forward(self, x: Tensor) -> Tensor:
        conv0 = self.conv0(x)
        mp0 = self.maxpool0(x)
        outputInitial = torch.cat((conv0, mp0), dim=1)

        bn1_0, idx1 = self.bottleneck1_0(outputInitial)
        bn1 = self.bottleneck1_1(bn1_0)
        bn2_0, idx2 = self.bottleneck2_0(bn1)
        bn2 = self.bottleneck2_1(bn2_0)

        bn3 = self.bottleneck3(bn2)

        up4 = self.bottleneck4((bn3, idx2, bn1))
        up5 = self.bottleneck5((up4, idx1, outputInitial))

        interp = F.interpolate(up5, mode='nearest', scale_factor=2)
        return self.final(interp)

    def init_weights(self, *args, **kwargs):
        self.apply(random_weights_init)
