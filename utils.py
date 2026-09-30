#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from functools import partial
from multiprocessing import Pool
from contextlib import AbstractContextManager
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, cast

import torch
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from tqdm import tqdm
from torch import Tensor, einsum

tqdm_ = partial(tqdm, dynamic_ncols=True,
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]')


class Dcm(AbstractContextManager):
    # Dummy Context manager
    def __exit__(self, *args, **kwargs):
        pass


# Functools
A = TypeVar("A")
B = TypeVar("B")


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def starmmap_(fn: Callable[[Tuple[A]], B], iter: Iterable[Tuple[A]]) -> List[B]:
    return Pool().starmap(fn, iter)


def crop_or_pad_arr(img: np.ndarray, size: tuple[int, int], value: int) -> np.ndarray:
    """
    Center crop (if too big) or pad with value (if too small) the x and y axes to size.
    The function is designed to be reversible.
    """
    result = img
    for axis, s in enumerate(size):
        diff = result.shape[axis] - s
        if diff > 0:  # if too big, crop the center
            start = diff // 2
            result = result.take(range(start, start + s), axis=axis)
        elif diff < 0:  # if too small, pad around with value
            before = (-diff) // 2
            after = -diff - before
            pad_width = [(0, 0)] * result.ndim
            pad_width[axis] = (before, after)
            result = np.pad(result, pad_width, constant_values=value)

    return result


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res


def probs2class(probs: Tensor) -> Tensor:
    b, _, *img_shape = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, *img_shape)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


# Save the raw predictions
def save_images(segs: Tensor, names: Iterable[str], root: Path) -> None:
        for seg, name in zip(segs, names):
                save_path = (root / name).with_suffix(".png")
                save_path.parent.mkdir(parents=True, exist_ok=True)

                if len(seg.shape) == 2:
                        Image.fromarray(seg.detach().cpu().numpy().astype(np.uint8)).save(save_path)
                elif len(seg.shape) == 3:
                        np.save(str(save_path), seg.detach().cpu().numpy())
                else:
                        raise ValueError(seg.shape)


# Metrics
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bk...->bk")
dice_batch = partial(meta_dice, "bk...->k")  # used for 3d dice


def volume_dice_from_slices(
    slices: list[tuple[int, Tensor, Tensor]], K: int
) -> Tensor:
    """Compute per-class 3D Dice from one patient's hard-label slices.

    ``dice_batch`` sums over its batch and spatial dimensions. Using the depth
    slices as its batch dimension is therefore equivalent to computing Dice
    over the complete 3D volume.
    """
    assert slices
    slices = sorted(slices, key=lambda item: item[0])
    slice_ids = [slice_id for slice_id, _, _ in slices]
    assert slice_ids == list(range(len(slice_ids))), slice_ids

    pred_volume = torch.stack([pred for _, pred, _ in slices]).to(torch.int64)
    gt_volume = torch.stack([gt for _, _, gt in slices]).to(torch.int64)
    return dice_batch(class2one_hot(gt_volume, K), class2one_hot(pred_volume, K))


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Return a 6-connected, voxel-centre surface for a 3D binary mask."""
    assert mask.ndim == 3, mask.shape
    mask = mask.astype(bool, copy=False)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    structure = generate_binary_structure(rank=3, connectivity=1)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return mask & ~eroded


def physical_volume_diagonal(shape: tuple[int, int, int],
                             spacing: tuple[float, float, float]) -> float:
    """Return the physical field-of-view diagonal in millimetres."""
    assert len(shape) == len(spacing) == 3
    assert all(length > 0 for length in shape), shape
    assert all(step > 0 for step in spacing), spacing
    return float(np.linalg.norm(np.asarray(shape) * np.asarray(spacing)))


def surface_distance_metrics_binary(
    pred_mask: np.ndarray, gt_mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Compute symmetric 3D HD95, HD, and ASSD in mm for one binary class.

    The directed surface-distance sets are calculated once and reused for all
    three metrics. The return order is ``(hd95, hd, assd)``.
    """
    assert pred_mask.shape == gt_mask.shape
    assert pred_mask.ndim == 3, pred_mask.shape
    assert len(spacing) == 3

    pred_mask = pred_mask.astype(bool, copy=False)
    gt_mask = gt_mask.astype(bool, copy=False)
    pred_empty = not pred_mask.any()
    gt_empty = not gt_mask.any()
    if pred_empty and gt_empty:
        return (0.0, 0.0, 0.0)
    if pred_empty or gt_empty:
        penalty = physical_volume_diagonal(pred_mask.shape, spacing)
        return (penalty, penalty, penalty)

    pred_surface = surface_voxels(pred_mask)
    gt_surface = surface_voxels(gt_mask)
    pred_to_gt = distance_transform_edt(~gt_surface, sampling=spacing)[pred_surface]
    gt_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)[gt_surface]
    hd95 = max(float(np.percentile(pred_to_gt, 95, method="linear")),
               float(np.percentile(gt_to_pred, 95, method="linear")))
    hd = max(float(pred_to_gt.max()), float(gt_to_pred.max()))
    assd = float(np.concatenate((pred_to_gt, gt_to_pred)).mean())
    return (hd95, hd, assd)


def _volumes_from_slices(slices: list[tuple[int, Tensor, Tensor]]) -> tuple[np.ndarray, np.ndarray]:
    """Stack ordered hard-label slices into prediction and ground-truth volumes."""
    assert slices
    slices = sorted(slices, key=lambda item: item[0])
    slice_ids = [slice_id for slice_id, _, _ in slices]
    assert slice_ids == list(range(len(slice_ids))), slice_ids

    pred_volume = torch.stack([pred for _, pred, _ in slices]).cpu().numpy()
    gt_volume = torch.stack([gt for _, _, gt in slices]).cpu().numpy()
    return pred_volume, gt_volume


def volume_surface_metrics_from_slices(
    slices: list[tuple[int, Tensor, Tensor]], K: int,
    spacing: tuple[float, float, float],
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute per-class 3D HD95, HD, and ASSD from one patient's slices."""
    pred_volume, gt_volume = _volumes_from_slices(slices)
    hd95s = torch.full((K,), torch.nan, dtype=torch.float32)
    hds = torch.full((K,), torch.nan, dtype=torch.float32)
    assds = torch.full((K,), torch.nan, dtype=torch.float32)
    for class_id in range(1, K):
        hd95s[class_id], hds[class_id], assds[class_id] = surface_distance_metrics_binary(
            pred_volume == class_id, gt_volume == class_id, spacing,
        )
    return hd95s, hds, assds


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
