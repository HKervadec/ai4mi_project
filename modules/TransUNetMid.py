# modules/TransUNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Blocks ---------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DownBlock(nn.Module):
    """Conv block with stride-2 downsample (keeps it simple + efficient)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, stride=2)
    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    """Transpose-conv upsample + skip fusion + conv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class PatchEmbed(nn.Module):
    """Tokenize deepest CNN feature map (TransUNet paper style)."""
    def __init__(self, in_ch, embed_dim=96, patch_size=2):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)                      # (B, E, Hp, Wp)
        B, E, Hp, Wp = x.shape
        tokens = x.flatten(2).transpose(1, 2) # (B, N, E), N = Hp*Wp
        return tokens, (Hp, Wp)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=96, depth=3, num_heads=4, mlp_ratio=3.0, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                batch_first=True
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# --------- Model ---------
class TransUNetMid(nn.Module):
    """
    A slightly-heavier, paper-aligned TransUNet:
      - 4-level CNN encoder (U-Net style)
      - Transformer encoder on tokens from deepest feature map
      - 3-stage cascaded upsampler with skip fusion (CUP)
    Returns logits (no softmax) shaped (B, num_classes, H, W).
    """
    def __init__(self,
                 in_ch: int = 1,
                 num_classes: int = 5,
                 base_ch: int = 24,          # Lite used 16
                 embed_dim: int = 96,        # Lite used 64
                 trans_depth: int = 3,       # Lite used 2
                 n_heads: int = 4,
                 patch_size: int = 2,
                 mlp_ratio: float = 3.0,
                 dropout: float = 0.0,
                 **kwargs):
        super().__init__()

        # ----- Encoder (4 levels) -----
        # Level 0 (no downsample)
        self.enc0 = ConvBlock(in_ch, base_ch)
        # Level 1..3 downsample
        self.down1 = DownBlock(base_ch,      base_ch * 2)  # 1/2
        self.down2 = DownBlock(base_ch * 2,  base_ch * 4)  # 1/4
        self.down3 = DownBlock(base_ch * 4,  base_ch * 8)  # 1/8

        # ----- Tokenize deepest map & Transformer -----
        self.patch_embed = PatchEmbed(base_ch * 8, embed_dim=embed_dim, patch_size=patch_size)
        self.trans_enc   = TransformerEncoder(embed_dim=embed_dim,
                                              depth=trans_depth,
                                              num_heads=n_heads,
                                              mlp_ratio=mlp_ratio,
                                              dropout=dropout)
        # Project tokens back to CNN channel space
        self.token_to_feat = nn.Linear(embed_dim, base_ch * 8)

        # ----- Decoder (3 stages) with skip fusion -----
        # Skips from enc2 (base*4), enc1 (base*2), enc0 (base)
        self.up2 = UpBlock(in_ch=base_ch * 8, skip_ch=base_ch * 4, out_ch=base_ch * 4)  # 1/8 → 1/4
        self.up1 = UpBlock(in_ch=base_ch * 4, skip_ch=base_ch * 2, out_ch=base_ch * 2)  # 1/4 → 1/2
        self.up0 = UpBlock(in_ch=base_ch * 2, skip_ch=base_ch,      out_ch=base_ch)     # 1/2 → 1/1

        self.final = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        input_size = (H, W)

        # ----- Encoder -----
        e0 = self.enc0(x)       # (B, base, H, W)
        e1 = self.down1(e0)     # (B, base*2, H/2, W/2)
        e2 = self.down2(e1)     # (B, base*4, H/4, W/4)
        e3 = self.down3(e2)     # (B, base*8, H/8, W/8)

        # ----- Transformer on deepest feature map -----
        tokens, (Hp, Wp) = self.patch_embed(e3)        # (B, N, E)
        tokens = self.trans_enc(tokens)                # (B, N, E)
        feat   = self.token_to_feat(tokens)            # (B, N, base*8)
        feat   = feat.transpose(1, 2).reshape(B, -1, Hp, Wp)  # (B, base*8, Hp, Wp)

        # ----- Decoder (CUP) with skip fusions -----
        d2 = self.up2(feat, e2)   # fuse with enc2 (base*4)
        d1 = self.up1(d2,  e1)    # fuse with enc1 (base*2)
        d0 = self.up0(d1,  e0)    # fuse with enc0 (base)

        logits = self.final(d0)   # (B, num_classes, h, w)

        # Guarantee same spatial size as input
        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return logits

    def init_weights(self, *args, **kwargs):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
