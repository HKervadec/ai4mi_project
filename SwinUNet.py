#!/usr/bin/env python3

"""A compact, convolution-free Swin-Unet for 2D segmentation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def window_partition(x: Tensor, window_size: int) -> Tensor:
    """Split [B, H, W, C] tokens into [B*n_windows, ws, ws, C] windows."""
    b, h, w, c = x.shape
    assert h % window_size == 0 and w % window_size == 0
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, c)


def window_reverse(windows: Tensor, window_size: int, height: int, width: int) -> Tensor:
    """Restore [B, H, W, C] tokens from partitioned windows."""
    b = windows.shape[0] // (height // window_size * width // window_size)
    x = windows.view(b, height // window_size, width // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, height, width, -1)


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.projection = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection_dropout = nn.Dropout(dropout)

        relative_size = 2 * window_size - 1
        self.relative_position_bias = nn.Parameter(torch.zeros(relative_size * relative_size, num_heads))

        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing='ij'))
        coords = coords.flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[0] += window_size - 1
        relative_coords[1] += window_size - 1
        relative_coords[0] *= relative_size
        self.register_buffer('relative_position_index', relative_coords.sum(0), persistent=False)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attention = (query * self.scale) @ key.transpose(-2, -1)
        relative_bias = self.relative_position_bias[self.relative_position_index.reshape(-1)]
        relative_bias = relative_bias.view(tokens, tokens, self.num_heads).permute(2, 0, 1)
        attention = attention + relative_bias.unsqueeze(0)

        if mask is not None:
            windows_per_image = mask.shape[0]
            attention = attention.view(batch_windows // windows_per_image, windows_per_image,
                                       self.num_heads, tokens, tokens)
            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.reshape(-1, self.num_heads, tokens, tokens)

        attention = self.attention_dropout(attention.softmax(dim=-1))
        x = (attention @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.projection_dropout(self.projection(x))


class SwinBlock(nn.Module):
    def __init__(self, dim: int, resolution: int, num_heads: int, window_size: int,
                 shift_size: int = 0, mlp_ratio: float = 2.0):
        super().__init__()
        assert resolution % window_size == 0
        self.dim = dim
        self.resolution = resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

        if shift_size == 0:
            self.register_buffer('attention_mask', None, persistent=False)
        else:
            image_mask = torch.zeros((1, resolution, resolution, 1))
            height_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            width_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            counter = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    image_mask[:, height_slice, width_slice, :] = counter
                    counter += 1
            mask_windows = window_partition(image_mask, window_size).reshape(-1, window_size * window_size)
            attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attention_mask = attention_mask.masked_fill(attention_mask != 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 0, 0.0)
            self.register_buffer('attention_mask', attention_mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, tokens, channels = x.shape
        assert tokens == self.resolution * self.resolution and channels == self.dim

        shortcut = x
        x = self.norm1(x).view(batch_size, self.resolution, self.resolution, channels)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(x, self.window_size).reshape(-1, self.window_size * self.window_size, channels)
        windows = self.attention(windows, self.attention_mask)
        x = window_reverse(windows.view(-1, self.window_size, self.window_size, channels),
                           self.window_size, self.resolution, self.resolution)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.reshape(batch_size, tokens, channels)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, resolution: int, dim: int):
        super().__init__()
        assert resolution % 2 == 0
        self.resolution = resolution
        self.dim = dim
        self.norm = nn.LayerNorm(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, tokens, channels = x.shape
        assert tokens == self.resolution * self.resolution and channels == self.dim
        x = x.view(batch_size, self.resolution, self.resolution, channels)
        x = torch.cat((x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                       x[:, 0::2, 1::2], x[:, 1::2, 1::2]), dim=-1)
        x = x.view(batch_size, -1, channels * 4)
        return self.reduction(self.norm(x))


class PatchExpand(nn.Module):
    def __init__(self, resolution: int, dim: int):
        super().__init__()
        self.resolution = resolution
        self.dim = dim
        self.expand = nn.Linear(dim, dim * 2, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, tokens, channels = x.shape
        assert tokens == self.resolution * self.resolution and channels == self.dim
        x = self.expand(x).view(batch_size, self.resolution, self.resolution, 2, 2, channels // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, -1, channels // 2)
        return self.norm(x)


class SwinStage(nn.Module):
    def __init__(self, dim: int, resolution: int, depth: int, num_heads: int, window_size: int):
        super().__init__()
        self.blocks = nn.Sequential(*[
            SwinBlock(dim, resolution, num_heads, window_size,
                      shift_size=0 if index % 2 == 0 else window_size // 2)
            for index in range(depth)
        ])

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class SwinUNet(nn.Module):
    """Tiny, pure-Transformer Swin-Unet for fixed 256x256 2D inputs."""
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        self.image_size = kwargs.get('image_size', 256)
        self.patch_size = kwargs.get('patch_size', 4)
        self.embed_dim = kwargs.get('embed_dim', 12)
        self.window_size = kwargs.get('window_size', 8)
        assert self.image_size % self.patch_size == 0

        base_resolution = self.image_size // self.patch_size
        assert base_resolution % (self.window_size * 8) == 0
        dims = (self.embed_dim, self.embed_dim * 2, self.embed_dim * 4, self.embed_dim * 8)
        resolutions = (base_resolution, base_resolution // 2, base_resolution // 4, base_resolution // 8)
        heads = (2, 4, 8, 8)

        self.patch_embedding = nn.Linear(in_dim * self.patch_size * self.patch_size, dims[0])
        self.encoder0 = SwinStage(dims[0], resolutions[0], depth=2, num_heads=heads[0], window_size=self.window_size)
        self.merge0 = PatchMerging(resolutions[0], dims[0])
        self.encoder1 = SwinStage(dims[1], resolutions[1], depth=2, num_heads=heads[1], window_size=self.window_size)
        self.merge1 = PatchMerging(resolutions[1], dims[1])
        self.encoder2 = SwinStage(dims[2], resolutions[2], depth=2, num_heads=heads[2], window_size=self.window_size)
        self.merge2 = PatchMerging(resolutions[2], dims[2])
        self.encoder3 = SwinStage(dims[3], resolutions[3], depth=2, num_heads=heads[3], window_size=self.window_size)

        self.expand2 = PatchExpand(resolutions[3], dims[3])
        self.fuse2 = nn.Linear(dims[2] * 2, dims[2])
        self.decoder2 = SwinStage(dims[2], resolutions[2], depth=2, num_heads=heads[2], window_size=self.window_size)
        self.expand1 = PatchExpand(resolutions[2], dims[2])
        self.fuse1 = nn.Linear(dims[1] * 2, dims[1])
        self.decoder1 = SwinStage(dims[1], resolutions[1], depth=2, num_heads=heads[1], window_size=self.window_size)
        self.expand0 = PatchExpand(resolutions[1], dims[1])
        self.fuse0 = nn.Linear(dims[0] * 2, dims[0])
        self.decoder0 = SwinStage(dims[0], resolutions[0], depth=2, num_heads=heads[0], window_size=self.window_size)

        self.final_expand = nn.Linear(dims[0], self.patch_size * self.patch_size * dims[0])
        self.head = nn.Linear(dims[0], out_dim)
        self.out_dim = out_dim

        print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with "
              f"image_size={self.image_size}, patch_size={self.patch_size}, embed_dim={self.embed_dim}")

    def _patchify(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape
        assert (channels, height, width) == (1, self.image_size, self.image_size)
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        return self.patch_embedding(patches)

    def _unpatchify(self, x: Tensor) -> Tensor:
        batch_size, tokens, channels = x.shape
        resolution = self.image_size // self.patch_size
        assert tokens == resolution * resolution
        x = self.final_expand(x).view(batch_size, resolution, resolution,
                                      self.patch_size, self.patch_size, channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(batch_size, channels,
                                                 self.image_size, self.image_size)
        return self.head(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.encoder0(self._patchify(x))
        x1 = self.encoder1(self.merge0(x0))
        x2 = self.encoder2(self.merge1(x1))
        x3 = self.encoder3(self.merge2(x2))

        x = self.decoder2(self.fuse2(torch.cat((self.expand2(x3), x2), dim=-1)))
        x = self.decoder1(self.fuse1(torch.cat((self.expand1(x), x1), dim=-1)))
        x = self.decoder0(self.fuse0(torch.cat((self.expand0(x), x0), dim=-1)))
        return self._unpatchify(x)

    def init_weights(self, *args, **kwargs):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, WindowAttention):
                nn.init.trunc_normal_(module.relative_position_bias, std=0.02)
