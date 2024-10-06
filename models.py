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

        # Freeze all the layers of the encoder except the last n
        unfreeze_enc_last_n_layers = kwargs['unfreeze_enc_last_n_layers']
        enc_num_layers = 0
        for _ in self.encoder.children():
            enc_num_layers += 1
        freeze_first_n_layers = enc_num_layers - unfreeze_enc_last_n_layers
        for layer_num, layer in enumerate(self.encoder.children()):
            if layer_num < freeze_first_n_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"> Initialized encoder {encoder_name} with first {freeze_first_n_layers}/{enc_num_layers} layers frozen")


    def init_weights(self):
        pass

