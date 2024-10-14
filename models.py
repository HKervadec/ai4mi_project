import segmentation_models_pytorch as smp
from torch import nn
from pathlib import Path
import torch


class SegmentationModelBase(nn.Module):

    def initialize_base(self, kwargs):
        self.encoder_name = kwargs['encoder_name']
        self.encoder_weights = 'imagenet'
        self.unfreeze_enc_last_n_layers = kwargs['unfreeze_enc_last_n_layers'] # How many of the last enc layers are unfrozen

    # Freeze all the layers of the encoder except the last n
    def freeze_encoder_layers(self):
        enc_num_layers = 0
        for _ in self.encoder.children():
            enc_num_layers += 1
        freeze_first_n_layers = enc_num_layers - self.unfreeze_enc_last_n_layers
        for layer_num, layer in enumerate(self.encoder.children()):
            if layer_num < freeze_first_n_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"> Initialized encoder {self.encoder_name} with first {freeze_first_n_layers}/{enc_num_layers} layers frozen")

    def init_weights(self, args):
        # If in evaluation mode, load the model from the file
        if args.evaluation:
            print(f"Loading model weights from  {args.dest} ...")
            trained_weights_path = args.dest / "bestweights.pt"
            device = 'cpu' if not args.gpu else None
            self.load_state_dict(torch.load(trained_weights_path, map_location=device))
        else:
            pass


class UNet(smp.Unet, SegmentationModelBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        self.initialize_base(kwargs)
        super().__init__(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=in_channels,
            classes=out_channels
        )
        self.freeze_encoder_layers()


class UNetPlusPlus(smp.UnetPlusPlus, SegmentationModelBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        self.initialize_base(kwargs)
        super().__init__(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=in_channels,
            classes=out_channels
        )
        self.freeze_encoder_layers()


class DeepLabV3Plus(smp.DeepLabV3Plus, SegmentationModelBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        self.initialize_base(kwargs)
        super().__init__(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=in_channels,
            classes=out_channels
        )
        self.freeze_encoder_layers()