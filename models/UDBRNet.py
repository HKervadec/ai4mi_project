import torch
import torch.nn as nn
from collections import OrderedDict
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
import numpy as np


class RefEncoder(nn.Module):
    def __init__(self, in_channel, init_feature):
        super(RefEncoder, self).__init__()
        self.in_channel = in_channel
        self.init_feature = init_feature

        self.conv0 = nn.Conv2d(self.in_channel, self.init_feature, 3, padding=1)

        self.conv1 = nn.Conv2d(self.init_feature, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        self.upscore2 = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        encoded = self.upscore2(hx5)

        skips = [hx1, hx2, hx3, hx4]

        return encoded, skips


class RefDecoder(nn.Module):
    def __init__(self, out_channel):
        super(RefDecoder, self).__init__()
        self.out_channel = out_channel

        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64, self.out_channel, 3, padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x, skips):
        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((x, skips[-1]), 1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, skips[-2]), 1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, skips[-3]), 1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, skips[-4]), 1))))

        residual = self.conv_d0(d1)

        return residual


class RefUnet(nn.Module):
    def __init__(self, in_chennel, out_channel, init_features):
        super(RefUnet, self).__init__()

        self.in_chennel = in_chennel
        self.out_chennel = out_channel
        self.init_features = init_features
        self.ref_encoder = RefEncoder(self.in_chennel, self.init_features)
        self.ref_attention = RefEncoder(self.in_chennel, self.init_features)
        self.ref_decoder = RefDecoder(self.out_chennel)

    def forward(self, x, weights=None):
        self.weights = weights

        enc, _ = self.ref_encoder(x)

        if isinstance(weights, torch.Tensor):
            # print(f"Shape of weights: {weights.shape}")
            # Shape of weights: torch.Size([8, 5, 256, 256])
            _, skips = self.ref_attention(weights)

        dec = self.ref_decoder(enc, skips)

        return torch.add(x, dec)


class Uncertainity:
    def __init__(self, seg_class=5):
        self.seg_class = seg_class

    def get_attention(self, main, aux1, aux2):
        main_seg_sm = torch.squeeze(torch.softmax(main, dim=1))
        aux1_seg_sm = torch.squeeze(torch.softmax(aux1, dim=1))
        aux2_seg_sm = torch.squeeze(torch.softmax(aux2, dim=1))

        main_seg = torch.argmax(main_seg_sm, dim=0)
        aux1_seg = torch.argmax(aux1_seg_sm, dim=0)
        aux2_seg = torch.argmax(aux2_seg_sm, dim=0)

        weight = main_seg_sm + aux1_seg_sm + aux2_seg_sm

        weight = weight / weight.max()

        sub = torch.zeros_like(main_seg_sm)

        for i in range(self.seg_class):
            main_seg_ch = torch.zeros_like(main_seg)
            aux1_seg_ch = torch.zeros_like(aux1_seg)
            aux2_seg_ch = torch.zeros_like(aux2_seg)

            main_seg_ch[main_seg == i] = 1
            aux1_seg_ch[aux1_seg == i] = 1
            aux2_seg_ch[aux2_seg == i] = 1

            mask_union = torch.bitwise_or(main_seg_ch, aux1_seg_ch)
            mask_union = torch.bitwise_or(mask_union, aux2_seg_ch)
            mask_inter = torch.bitwise_and(main_seg_ch, aux1_seg_ch)
            mask_inter = torch.bitwise_and(mask_inter, aux2_seg_ch)
            sub[i] = torch.abs(mask_union - mask_inter)

        sub = torch.add(sub, 1)
        attn = torch.mul(weight, sub)
        return attn


class Encoder(nn.Module):
    def __init__(self, in_channels, init_features):
        super(Encoder, self).__init__()
        self.features = init_features
        self.in_channels = in_channels
        self.encoder1 = Unet_block._block(self.in_channels, self.features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = Unet_block._block(self.features, self.features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = Unet_block._block(
            self.features * 2, self.features * 4, name="enc3"
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = Unet_block._block(
            self.features * 4, self.features * 8, name="enc4"
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        skips = [enc1, enc2, enc3, enc4]

        return enc4, skips


class Bottleneck(nn.Module):
    def __init__(self, init_features):
        super(Bottleneck, self).__init__()
        self.features = init_features
        self.bottleneck = Unet_block._block(
            self.features * 8, self.features * 16, name="bottleneck"
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.bottleneck(self.pool4(x))


class Decoder(nn.Module):
    def __init__(self, init_features):
        super(Decoder, self).__init__()
        self.features = init_features
        self.upconv4 = nn.ConvTranspose2d(
            self.features * 16, self.features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = Unet_block._block(
            (self.features * 8) * 2, self.features * 8, name="dec4"
        )

        self.upconv3 = nn.ConvTranspose2d(
            self.features * 8, self.features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = Unet_block._block(
            (self.features * 4) * 2, self.features * 4, name="dec3"
        )

        self.upconv2 = nn.ConvTranspose2d(
            self.features * 4, self.features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = Unet_block._block(
            (self.features * 2) * 2, self.features * 2, name="dec2"
        )

        self.upconv1 = nn.ConvTranspose2d(
            self.features * 2, self.features, kernel_size=2, stride=2
        )
        self.decoder1 = Unet_block._block(self.features * 2, self.features, name="dec1")

    def forward(self, x, skips):
        dec4 = self.upconv4(x)
        dec4 = torch.cat((dec4, skips[-1]), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, skips[-2]), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, skips[-3]), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, skips[-4]), dim=1)
        dec1 = self.decoder1(dec1)

        return dec1


class UDBRNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=5, init_features=64):
        super(UDBRNet, self).__init__()
        self.features = init_features
        self.in_channel = in_channels
        self.out_channel = out_channels

        self.uncertainity_weight = None
        self.uncertainity = Uncertainity(seg_class=self.out_channel)

        self.encoder = Encoder(self.in_channel, self.features)
        self.bottleneck_layer = Bottleneck(self.features)
        self.decoder = Decoder(self.features)

        self.addNoise = FeatureNoise()
        self.dropOut = FeatureDrop()

        self.conv = nn.Conv2d(
            in_channels=self.features, out_channels=self.out_channel, kernel_size=1
        )

        self.ref_unet = RefUnet(self.out_channel, self.out_channel, 64)

    def forward(self, x):
        encoded, skips = self.encoder(x)

        bn = self.bottleneck_layer(encoded)
        bottleneck_with_noise = self.addNoise(bn)
        bottleneck_with_dropOut = self.dropOut(bn)

        decoded1 = self.decoder(bn, skips)
        decoded2 = self.decoder(bottleneck_with_noise, skips)
        decoded3 = self.decoder(bottleneck_with_dropOut, skips)

        output1 = self.conv(decoded1)
        output2 = self.conv(decoded2)
        output3 = self.conv(decoded3)

        # print(f"Shape of output1: {output1.shape}")
        # Shape of output1: torch.Size([8, 5, 256, 256])

        self.uncertainity_weight = self.uncertainity.get_attention(
            output1, output2, output3
        )

        # Ensure that self.uncertainity_weight has the correct dimensions
        if self.uncertainity_weight.dim() == 5:
            self.uncertainity_weight = self.uncertainity_weight.squeeze(0)

        # refined = self.ref_unet(output1, self.uncertainity_weight[None, :, :, :])

        # self.uncertainity_weight is of shape [8, 5, 256, 256]
        # rm None because is adding an extra dimension
        refined = self.ref_unet(output1, self.uncertainity_weight)

        # return output1, output2, output3, refined

        # Return only first output for debugging
        return output1

    def init_weights(self, *args, **kwargs):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        # self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)
        self.uni_dist = Uniform(-uniform_range, uniform_range)
        self.Nor_dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def feature_based_noise(self, x):
        noise_vector = (
            self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        )  # uniform noise
        # noise_vector = self.Nor_dist.sample(x.shape[1:]).to(x.device).squeeze().unsqueeze(0) #gaussioan noise
        noise_vector = torch.div(noise_vector, noise_vector.max() - noise_vector.min())
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        # x = self.upsample(x)
        return x


class FeatureDrop(nn.Module):
    def __init__(self):
        super(FeatureDrop, self).__init__()
        # self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x):
        x = self.feature_dropout(x)
        # x = self.upsample(x)
        return x


class Unet_block(nn.Module):
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                ]
            )
        )
