"""Losses take (pred_probs, gt_onehot), both (B, K, H, W), like the course CrossEntropy."""
from losses import CrossEntropy
from src.registry import register


@register("loss", "cross_entropy")
def build_cross_entropy(num_classes: int, idk: list[int] | None = None) -> CrossEntropy:
    return CrossEntropy(idk=list(range(num_classes)) if idk is None else idk)
