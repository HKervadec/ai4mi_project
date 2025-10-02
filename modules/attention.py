import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ECALayer(nn.Module):
    """Efficient Channel Attention (no FC reduction; 1D conv over pooled channels)."""
    # https://arxiv.org/abs/1910.03151
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        y = self.gap(x)                          # (B,C,1,1)
        y = y.squeeze(-1).transpose(-1, -2)      # (B,1,C)
        y = self.conv(y)                         # (B,1,C)
        y = y.transpose(-1, -2).unsqueeze(-1)    # (B,C,1,1)
        w = self.sig(y)
        return x * w

class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel then spatial)."""
    # https://arxiv.org/abs/1807.06521
    def __init__(self, channels: int, r: int = 16, spatial: bool = True):
        super().__init__()
        # Channel attention
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        mid = max(8, channels // r)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
        self.sig = nn.Sigmoid()
        # Spatial attention
        self.spatial = spatial
        if spatial:
            self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=7//2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        ca = self.sig(self.mlp(self.gap(x)) + self.mlp(self.gmp(x)))
        x = x * ca
        if self.spatial:
            avg = torch.mean(x, dim=1, keepdim=True)
            mx, _ = torch.max(x, dim=1, keepdim=True)
            sa = torch.sigmoid(self.spatial_conv(torch.cat([avg, mx], dim=1)))
            x = x * sa
        return x

# TODO: debug, see: outfiles/skip_attn_15071723.out
class AttentionGate(nn.Module):
    """Attention gate for skip connections (Attention U-Net style)."""
    # https://arxiv.org/abs/1804.03999
    def __init__(self, in_skip: int, in_gating: int, inter: int):
        super().__init__()
        self.theta = nn.Conv2d(in_skip, inter, kernel_size=1, bias=False)
        self.phi   = nn.Conv2d(in_gating, inter, kernel_size=1, bias=False)
        self.psi   = nn.Conv2d(inter, 1, kernel_size=1, bias=False)
        self.bn    = nn.BatchNorm2d(inter)
        self.sig   = nn.Sigmoid()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, skip: Tensor, gate: Tensor) -> Tensor:
        # resize gate to skip spatial size if needed
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        q = self.theta(skip)
        k = self.phi(gate)
        x = self.relu(self.bn(q + k))
        alpha = self.sig(self.psi(x))            # (B,1,H,W)
        return skip * alpha
