"""Loss functions and loss-level regularizers."""

import torch
import torch.nn.functional as F

from losses import CrossEntropy, CrossEntropyDice, CrossEntropyTversky, FocalLoss, FocalDice


class BoundaryRegularizer:
    """L1 between Sobel edge maps of predicted and GT segmentations.

    per_class=False: all organs merged into one foreground (only the outer outline counts).
    per_class=True: one edge map per organ, so boundaries between touching organs count too.
    """

    def __init__(self, per_class: bool = False, **kwargs):
        self.per_class = per_class
        print(f"Initialized {self.__class__.__name__} with per_class={per_class}")

    def __call__(self, pred_probs, onehot_target):
        if self.per_class:
            # One channel per organ (background dropped): B x (K-1) x H x W
            pred_fg = pred_probs[:, 1:, ...]
            target_fg = onehot_target[:, 1:, ...].float()
        else:
            # Convert multi-class probabilities to foreground probability.
            pred_fg = pred_probs[:, 1:, ...].sum(dim=1, keepdim=True)
            target_fg = onehot_target[:, 1:, ...].sum(dim=1, keepdim=True).float()
        C = pred_fg.shape[1]

        # Horizontal and vertical Sobel filters, shape (C, 1, 3, 3): applied to each channel separately.
        sobel_x = pred_probs.new_tensor([[[-1, 0, 1],
                                          [-2, 0, 2],
                                          [-1, 0, 1]]]).unsqueeze(0).repeat(C, 1, 1, 1)
        sobel_y = pred_probs.new_tensor([[[-1, -2, -1],
                                          [ 0,  0,  0],
                                          [ 1,  2,  1]]]).unsqueeze(0).repeat(C, 1, 1, 1)

        pred_x = F.conv2d(pred_fg, sobel_x, padding=1, groups=C)
        pred_y = F.conv2d(pred_fg, sobel_y, padding=1, groups=C)
        target_x = F.conv2d(target_fg, sobel_x, padding=1, groups=C)
        target_y = F.conv2d(target_fg, sobel_y, padding=1, groups=C)

        # eps keeps the sqrt gradient finite where the edge map is 0
        pred_boundary = torch.sqrt(pred_x ** 2 + pred_y ** 2 + 1e-8)
        target_boundary = torch.sqrt(target_x ** 2 + target_y ** 2 + 1e-8)

        return F.l1_loss(pred_boundary, target_boundary)


class CombinedLoss:
    """base_loss + sum(weight * regularizer), all called as (probs, onehot_target)."""

    def __init__(self, base_loss, regularizers):
        self.base_loss = base_loss
        self.regularizers = regularizers

    def __call__(self, pred_probs, target):
        loss = self.base_loss(pred_probs, target)
        for weight, regularizer in self.regularizers:
            loss = loss + weight * regularizer(pred_probs, target)
        return loss


# name -> class with __init__(idk, **params) and __call__(probs, onehot_target) -> scalar
LOSSES: dict = {
    "ce": CrossEntropy,
    "ce_dice": CrossEntropyDice,
    "ce_tversky": CrossEntropyTversky,
    "focal": FocalLoss,
    "focal_dice": FocalDice,
}

# name -> class with __init__(**params) and __call__(probs, onehot_target) -> scalar
REGULARIZERS: dict = {
    "boundary": BoundaryRegularizer,
}


def build_loss(cfg, K: int):
    params = dict(cfg.loss)
    name = params.pop("name")
    classes = params.pop("classes", "all")
    if name not in LOSSES:
        raise KeyError(f"unknown loss '{name}'. Known: {sorted(LOSSES)}")
    base_loss = LOSSES[name](idk=list(range(K)) if classes == "all" else list(classes), **params)

    # regularizers: [{name: boundary, weight: 0.1, ...}] -> added to the base loss
    regularizers = []
    for reg_cfg in cfg.get("regularizers") or []:
        reg_params = dict(reg_cfg)
        reg_name = reg_params.pop("name")
        weight = reg_params.pop("weight", 1.0)
        if reg_name not in REGULARIZERS:
            raise KeyError(f"unknown regularizer '{reg_name}'. Known: {sorted(REGULARIZERS)}")
        regularizers.append((weight, REGULARIZERS[reg_name](**reg_params)))

    if not regularizers:
        return base_loss
    return CombinedLoss(base_loss, regularizers)
