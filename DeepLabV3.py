import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class DeepLabV3(torch.nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DeepLabV3, self).__init__()
        self.deeplabv3 = deeplabv3_mobilenet_v3_large(
            weights='COCO_WITH_VOC_LABELS_V1' if pretrained else None,
            weights_backbone='IMAGENET1K_V1' if pretrained else None,
        )
        
        self.deeplabv3.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
        if pretrained:    
            # freeze all layers except the classifier
            for param in self.deeplabv3.parameters():
                param.requires_grad = False

            for param in self.deeplabv3.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.deeplabv3(x)['out']