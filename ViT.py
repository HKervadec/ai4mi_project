import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, in_channels=1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))

    def forward(self, x):
        return x + self.pos_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x)[0]


class MultiLayerPerceptron(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.layers(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        # self.mlp = nn.Sequential(
        #     nn.Linear(embed_dim, mlp_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_dim, embed_dim)
        # )
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConvDecoder(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, out_dim, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, 256, embed_dim)
        batch_size, tokens, channels = x.shape
        side = int(tokens ** 0.5)

        x = x.transpose(1, 2).reshape(
            batch_size, channels, side, side
        )
        return self.decoder(x)


# class OutputProjection(nn.Module):
#     def __init__(self, img_size, patch_size, embed_dim, out_dim):
#         super().__init__()
#         self.patch_size = patch_size
#         self.out_dim = out_dim
#         self.projection = nn.Linear(embed_dim, patch_size * patch_size * out_dim)
#         self.fold = nn.Fold(output_size=(img_size, img_size), kernel_size=patch_size, stride=patch_size)

    # def forward(self, x):
    #     B, T, C = x.shape
    #     x = self.projection(x)

    #     x = x.permute(0, 2, 1)
    #     x = self.fold(x)
    #     return x 


class ViT(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim= 768, num_heads=8, depth=6,
                 mlp_dim=1024, out_dim=5):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=1, embed_dim=embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim=embed_dim, seq_len=(img_size // patch_size) ** 2)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mlp_head = MultiLayerPerceptron(embed_dim=embed_dim, mlp_dim=mlp_dim)
        # self.output_proj = OutputProjection(img_size, patch_size, embed_dim, out_dim)
        self.decoder = ConvDecoder(embed_dim, out_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)

        x = self.mlp_head(x[:, 1:, :]) # Remove CLS token
        # return self.mlp_head(x[:, 0])
        # x = self.output_proj(x[:, 1:, :]) # Remove CLS token
        return self.decoder(x)