"""Optimizers and LR schedulers selectable from the config (optim.name / scheduler.name)."""
import torch

from src.registry import register


@register("optim", "adam")
def build_adam(params, **kwargs):
    return torch.optim.Adam(params, **kwargs)


@register("optim", "adamw")
def build_adamw(params, **kwargs):
    return torch.optim.AdamW(params, **kwargs)


@register("optim", "sgd")
def build_sgd(params, **kwargs):
    return torch.optim.SGD(params, **kwargs)


@register("scheduler", "none")
def build_no_scheduler(optimizer, epochs):
    return None


@register("scheduler", "cosine")
def build_cosine(optimizer, epochs, eta_min: float = 0.0):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)


@register("scheduler", "step")
def build_step(optimizer, epochs, step_size: int = 10, gamma: float = 0.1):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
