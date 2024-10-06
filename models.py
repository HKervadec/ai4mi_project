import segmentation_models_pytorch as smp


class UNet(smp.Unet):
    def __init__(self, in_channels, out_channels, **kwargs):
        encoder_name = kwargs['encoder_name'] if "encoder_name" in kwargs else 'resnet18'
        encoder_weights = kwargs['encoder_weights'] if "encoder_weights" in kwargs else 'imagenet'
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels
        )
        for param in self.encoder.parameters():
            param.requires_grad = False

    def init_weights(self):
        pass

