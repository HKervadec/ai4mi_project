"""Network architectures."""

import torch
from torch import nn

from ENet import ENet
from ShallowNet import shallowCNN
from enet_transformer import ENetTransformer


def _no_pretrained(name: str, pretrained: bool) -> None:
    if pretrained:
        raise ValueError(f"model '{name}' has no pretrained weights")


def build_enet(in_channels: int, K: int, pretrained: bool = False, **params) -> nn.Module:
    _no_pretrained("enet", pretrained)
    net = ENet(in_channels, K, **params)
    net.init_weights()
    return net


def build_shallow(in_channels: int, K: int, pretrained: bool = False, **params) -> nn.Module:
    _no_pretrained("shallow", pretrained)
    net = shallowCNN(in_channels, K, **params)
    net.init_weights()
    return net


def build_enet_transformer(in_channels: int, K: int, pretrained: bool = False, **params) -> nn.Module:
    # params: transformer_at (str or list, default after bottleneck (bn3)), num_layers, num_heads,
    # mlp_ratio, dropout, use_pos_embed
    _no_pretrained("enet_transformer", pretrained)
    net = ENetTransformer(in_channels, K, **params)
    net.init_weights()
    return net


# name -> build(in_channels, K, pretrained, **params) -> nn.Module returning B x K x H x W logits
MODELS: dict = {
    "enet": build_enet,
    "shallow": build_shallow,
    "enet_transformer": build_enet_transformer,
}


def build_model(cfg_model, in_channels: int, K: int) -> nn.Module:
    name = cfg_model.name
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'. Known: {sorted(MODELS)}")
    params = dict(cfg_model.get("params") or {})
    model = MODELS[name](in_channels, K, pretrained=cfg_model.get("pretrained", False), **params)

    init_from = cfg_model.get("init_from")
    if init_from:
        state = torch.load(init_from, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"> Loaded {init_from}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    return model
