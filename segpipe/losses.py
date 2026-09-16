"""Loss functions and loss-level regularizers."""

from losses import CrossEntropy

# name -> class with __init__(idk, **params) and __call__(probs, onehot_target) -> scalar
LOSSES: dict = {
    "ce": CrossEntropy,
}

# Not used by build_loss yet
REGULARIZERS: dict = {}


def build_loss(cfg, K: int):
    if cfg.get("regularizers"):
        raise NotImplementedError("regularizers are not added to the loss yet")
    params = dict(cfg.loss)
    name = params.pop("name")
    classes = params.pop("classes", "all")
    if name not in LOSSES:
        raise KeyError(f"unknown loss '{name}'. Known: {sorted(LOSSES)}")
    return LOSSES[name](idk=list(range(K)) if classes == "all" else list(classes), **params)
