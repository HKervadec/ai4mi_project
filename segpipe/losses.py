"""Loss functions and loss-level regularizers."""

from losses import CrossEntropy

# name -> class with __init__(idk, **params) and __call__(probs, onehot_target) -> scalar
LOSSES: dict = {
    "ce": CrossEntropy,
}

# name -> class with __init__(**params) and __call__(probs, onehot_target) -> scalar
REGULARIZERS: dict = {}


class WithRegularizers():
    def __init__(self, loss, regularizers: list[tuple[float, object]]):
        self.loss = loss
        self.regularizers = regularizers

    def __call__(self, pred_softmax, weak_target):
        total = self.loss(pred_softmax, weak_target)
        for weight, regularizer in self.regularizers:
            total = total + weight * regularizer(pred_softmax, weak_target)
        return total


def build_loss(cfg, K: int):
    params = dict(cfg.loss)
    name = params.pop("name")
    classes = params.pop("classes", "all")
    if name not in LOSSES:
        raise KeyError(f"unknown loss '{name}'. Known: {sorted(LOSSES)}")
    loss = LOSSES[name](idk=list(range(K)) if classes == "all" else list(classes), **params)

    regularizers = []
    for item in cfg.get("regularizers") or []:
        reg_params = dict(item)
        reg_name = reg_params.pop("name")
        weight = reg_params.pop("weight", 1.0)
        if reg_name not in REGULARIZERS:
            raise KeyError(f"unknown regularizer '{reg_name}'. Known: {sorted(REGULARIZERS)}")
        regularizers.append((weight, REGULARIZERS[reg_name](**reg_params)))
    return WithRegularizers(loss, regularizers) if regularizers else loss
