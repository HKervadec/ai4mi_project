import torch
import numpy as np
from medpy import metric
from functools import partial
from torch import Tensor, einsum
from utils.tensor_utils import one_hot, sset


def meta_dice(
    sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8
) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(
        torch.float32
    )
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(
        torch.float32
    )

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bk...->bk")
dice_batch = partial(meta_dice, "bk...->k")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a & b
    assert sset(res, [0, 1])

    return res


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a | b
    assert sset(res, [0, 1])

    return res


def iou_coef(label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape

    inter_size: Tensor = intersection(label, pred).sum().float()
    union_size: Tensor = union(label, pred).sum().float()

    return (inter_size + smooth) / (union_size + smooth)


def hausdorff_distance(pred: Tensor, label: Tensor) -> Tensor:
    # implementation based on
    # https://github.com/nazib/MIDL_code/blob/hpc/evaluate.py
    assert pred.shape == label.shape
    assert sset(pred, [0, 1]) and sset(label, [0, 1])

    pred_np = pred.cpu().numpy()
    label_np = label.cpu().numpy()

    if np.sum(label_np) == 0 or np.sum(pred_np) == 0:
        return torch.tensor(float("nan"))

    return torch.tensor(metric.hd(pred_np, label_np))
