#!/usr/bin/env python3

"""A UNet class covering the 2D and 2.5D variants. 2.5D is the 2D net fed a stack of adjacent
slices in the channel dimension.

    2D    : in_dim=1                  input (B, 1, W, H)
    2.5D  : in_dim=2 * adjacent_slices + 1 input (B, 2n+1, W, H)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def random_weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                        m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


def conv_block(in_dim, out_dim):
        return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
                             nn.BatchNorm2d(out_dim),
                             nn.PReLU(),
                             nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                             nn.BatchNorm2d(out_dim),
                             nn.PReLU())


class UNet(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, *,
                     kernels: int = 32, factor: int = 2, depth: int = 4):
                super().__init__()
                K: int = kernels  # base width
                F_: int = factor  # width growth per level

                widths: list[int] = [K * F_ ** i for i in range(depth + 1)]

                self.downs = nn.ModuleList([conv_block(i, o) for i, o
                                            in zip([in_dim] + widths[:-2], widths[:-1])])
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.bottleneck = conv_block(widths[-2], widths[-1])
                self.ups = nn.ModuleList([nn.ConvTranspose2d(o, i, kernel_size=2, stride=2)
                                          for i, o in zip(widths[:-1], widths[1:])])
                self.up_convs = nn.ModuleList([conv_block(2 * i, i) for i in widths[:-1]])
                self.final = nn.Conv2d(widths[0], out_dim, kernel_size=1)

        def forward(self, input) -> Tensor:
                skips: list[Tensor] = []
                x = input
                for down in self.downs:
                        x = down(x)
                        skips.append(x)
                        x = self.pool(x)

                x = self.bottleneck(x)

                for up, up_conv, skip in zip(reversed(self.ups), reversed(self.up_convs), reversed(skips)):
                        x = up(x)
                        if x.shape[2:] != skip.shape[2:]:  # odd input sizes
                                x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
                        x = up_conv(torch.cat([skip, x], dim=1))

                return self.final(x)

        def init_weights(self, *args, **kwargs):
                self.apply(random_weights_init)


if __name__ == '__main__':
        # 2D
        net = UNet(1, 5, kernels=8)
        assert net(torch.rand(2, 1, 256, 256)).shape == (2, 5, 256, 256)
        # 2.5D: same class, wider input
        net = UNet(3, 5, kernels=8)
        assert net(torch.rand(2, 3, 256, 256)).shape == (2, 5, 256, 256)
        print("UNet ok")
