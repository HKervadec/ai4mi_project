"""The two course-provided networks. Factories give every model the same entry signature
(in_channels, num_classes, **model-specific kwargs) regardless of the class's own constructor."""
from torch import nn

from ENet import ENet
from ShallowNet import shallowCNN
from src.registry import register


@register("model", "enet")
def build_enet(in_channels: int, num_classes: int, kernels: int = 8, factor: int = 2) -> nn.Module:
    net = ENet(in_channels, num_classes, kernels=kernels, factor=factor)
    net.init_weights()
    return net


@register("model", "shallow_cnn")
def build_shallow_cnn(in_channels: int, num_classes: int, n_filters: int = 4) -> nn.Module:
    net = shallowCNN(in_channels, num_classes, nG=n_filters)
    net.init_weights()
    return net
