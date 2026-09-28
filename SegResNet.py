
"""
3D SegResNet.py

A U-Net-based network with repetitive residual blocks and deep supervision.

Following architecture description from Ghaedi et al. (2025). 

"""

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, num_groups=8, stride=1):
        super().__init__()
        self.resblock = nn.Sequential(
                nn.GroupNorm(num_groups, dim),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(dim, dim, kernel_size, stride, padding=1, bias=False),
                nn.GroupNorm(num_groups, dim),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(dim, dim, kernel_size, stride, padding=1, bias=False)
        )

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()        
        # 3D convolution with 1x1x1 stride + upsampling
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, input):
        conv_output = self.conv(input)
        return self.up(conv_output)



class SegResNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        K: int = kwargs.get("kernels", 16)          # base feature maps
        c1, c2, c3, c4 = K, K * 2, K * 4, K * 8     # Reduce and doubling image size with factor 2

        # Initial 3D convolution with stride 1x1x1
        self.init_conv = nn.Conv3d(in_dim, c1, kernel_size=3, padding=1, bias=False)
        self.enc0 = ResidualBlock(c1)

        # Encoder path: three convolutions with stride 2x2x2 for size reduction
        self.down1 = nn.Conv3d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc1 = nn.Sequential(*[ResidualBlock(c2) for _ in range(2)])   # x2

        self.down2 = nn.Conv3d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc2 = nn.Sequential(*[ResidualBlock(c3) for _ in range(2)])

        self.down3 = nn.Conv3d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc3 = nn.Sequential(*[ResidualBlock(c4) for _ in range(4)])

        # Decoder path
        self.up3 = UpsamplingBlock(c4, c3)
        self.dec3 = ResidualBlock(c3)

        self.up2 = UpsamplingBlock(c3, c2)
        self.dec2 = ResidualBlock(c2)

        self.up1 = UpsamplingBlock(c2, c1)
        self.dec1 = ResidualBlock(c1)

        # Final block: GN + Leaky ReLU + Conv3D + dropout
        self.final_block = nn.Sequential(
                nn.GroupNorm(8, c1),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(c1, c1, kernel_size=3, padding=1, bias=False),
                nn.Dropout3d(p=0.2)
        )

        self.final = nn.Conv3d(c1, out_dim, kernel_size=1)

        n_params: int = sum(q.numel() for q in self.parameters())
        print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) "
              f"with base={K}, params={n_params}")

    def forward(self, input: Tensor) -> Tensor:
        # Encoder
        e0 = self.enc0(self.init_conv(input))
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        e3 = self.enc3(self.down3(e2))

        # Decoder (adding the output of the encoders for skip connections)
        d3 = self.dec3(self.up3(e3) + e2)
        d2 = self.dec2(self.up2(d3) + e1)
        d1 = self.dec1(self.up1(d2) + e0)

        # Final output
        out = self.final_block(d1)
        return self.final(out)

    def init_weights(self, *args, **kwargs):
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)