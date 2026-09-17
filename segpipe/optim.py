"""Optimizers and learning-rate schedulers."""

import torch

# name -> torch optimizer class, or fn(params, lr, **kw)
OPTIMIZERS: dict = {
    "adam": torch.optim.Adam,
}


def _none(optimizer, epochs):
    return None


# name -> fn(optimizer, epochs, **kw) -> scheduler or None, stepped once per epoch
SCHEDULERS: dict = {
    "none": _none,
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
