import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic conv block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# Slim encoder
class SlimEncoder(nn.Module):
    def __init__(self, in_ch, base_ch=16):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, stride=2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, stride=2)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        return [e1, e2, e3]

# Patch embedding for transformer
class SlimPatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim=64, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)
        B, E, Hp, Wp = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return tokens, (Hp, Wp)

# Transformer encoder (slim)
class SlimTransformerEncoder(nn.Module):
    def __init__(self, dim, depth=2, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout,
                batch_first=True
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

# Decoder + fusion with skip connections
class SlimDecoderFusion(nn.Module):
    def __init__(self, embed_dim, decoder_chs, skip_chs, num_classes):
        super().__init__()
        # Project token → feature map
        self.token_to_map = nn.Linear(embed_dim, decoder_chs[0])
        self.up_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        for i in range(len(decoder_chs) - 1):
            in_ch = decoder_chs[i] + skip_chs[i]
            out_ch = decoder_chs[i + 1]
            # upsample via transpose convolution
            self.up_blocks.append(
                nn.ConvTranspose2d(decoder_chs[i], decoder_chs[i], kernel_size=2, stride=2)
            )
            self.conv_blocks.append(ConvBlock(in_ch, out_ch))
        self.final = nn.Conv2d(decoder_chs[-1], num_classes, kernel_size=1)

    def forward(self, tokens, hw, skip_feats, input_size):
        Hp, Wp = hw
        B, N, E = tokens.shape
        # project tokens to decoder_chs[0] channels
        x = self.token_to_map(tokens)  # (B, N, dec_ch0)
        x = x.transpose(1, 2).reshape(B, -1, Hp, Wp)
        out = x
        for i, (up, conv) in enumerate(zip(self.up_blocks, self.conv_blocks)):
            out = up(out)
            skip = skip_feats[i]
            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(out, size=skip.shape[2:], mode='bilinear', align_corners=False)
            out = torch.cat([out, skip], dim=1)
            out = conv(out)
        # final logits
        logits = self.final(out)
        # Now logits is continuous, of shape (B, num_classes, h, w)
        # Upsample logits to input size if necessary (still continuous)
        if logits.shape[2:] != input_size:
            # We use bilinear interpolation on logits (allowed)
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return logits

class TransUNetLite(nn.Module):
    def __init__(self, in_ch=1, num_classes=5,
                 base_ch=16, embed_dim=64, trans_depth=2, n_heads=4):
        super().__init__()
        self.encoder = SlimEncoder(in_ch, base_ch=base_ch)
        skip1 = base_ch * 2
        skip2 = base_ch
        deepest = base_ch * 4
        self.patch_embed = SlimPatchEmbed(deepest, embed_dim=embed_dim, patch_size=2)
        self.trans_enc = SlimTransformerEncoder(embed_dim, depth=trans_depth, num_heads=n_heads)
        decoder_chs = [deepest, base_ch * 2, base_ch]
        skip_chs = [skip1, skip2]
        self.decoder = SlimDecoderFusion(embed_dim, decoder_chs, skip_chs, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        input_size = (H, W)
        e1, e2, e3 = self.encoder(x)
        tokens, hw = self.patch_embed(e3)
        tokens = self.trans_enc(tokens)
        skip_feats = [e2, e1]
        logits = self.decoder(tokens, hw, skip_feats, input_size)
        return logits

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if hasattr(m, 'weight'):
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias'):
                    nn.init.zeros_(m.bias)
