"""Optimizers and learning-rate schedulers."""

import torch

# name -> torch optimizer class, or fn(params, lr, **kw)
OPTIMIZERS: dict = {
    "adam": torch.optim.Adam,
}


def _none(optimizer, epochs):
    return None


def _cosine(optimizer, epochs, min_lr: float = 0.0):
    # lr follows half a cosine from the initial lr down to min_lr over the whole run
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)


def _poly(optimizer, epochs, exponent: float = 0.9):
    # nnU-Net schedule: lr * (1 - epoch / epochs) ** exponent
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: (1 - e / epochs) ** exponent)


# name -> fn(optimizer, epochs, **kw) -> scheduler or None, stepped once per epoch
SCHEDULERS: dict = {
    "none": _none,
    "cosine": _cosine,
    "poly": _poly,
}


def build_optimizer(model, cfg_optimizer):
    params = dict(cfg_optimizer)
    name = params.pop("name")
    lr = params.pop("lr")
    if name not in OPTIMIZERS:
        raise KeyError(f"unknown optimizer '{name}'. Known: {sorted(OPTIMIZERS)}")
    if "betas" in params:
        params["betas"] = tuple(params["betas"])
    return OPTIMIZERS[name](model.parameters(), lr=lr, **params)


def build_scheduler(optimizer, cfg_scheduler, epochs: int):
    params = dict(cfg_scheduler or {"name": "none"})
    name = params.pop("name")
    if name not in SCHEDULERS:
        raise KeyError(f"unknown scheduler '{name}'. Known: {sorted(SCHEDULERS)}")
    return SCHEDULERS[name](optimizer, epochs, **params)
