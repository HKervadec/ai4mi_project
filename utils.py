#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

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
from tqdm import tqdm
from torch import Tensor, einsum
from scipy.spatial.distance import directed_hausdorff 
from skimage.segmentation import find_boundaries

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

def meta_jaccard(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)
    
    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    union_size: Tensor = einsum(sum_str, [union(label,pred)]).type(torch.float32)

    jaccards: Tensor = (inter_size + smooth) / (union_size + smooth)
    
    return jaccards

#Jaccard metrics for batch and for entire dataset
#using partial like Dice metric to pre-define summation pattern
jaccard_coef = partial(meta_jaccard, "bk...->bk")
jaccard_batch = partial(meta_jaccard, "bk...->k")  # for 3d IoU


def boundary_points(seg: Tensor) -> np.ndarray:
    """
    Extracting boundary points from the binary segmentation mask
    """
    seg = seg.cpu().numpy()
    boundary = np.zeros_like(seg) # Initialize boundary array
    

    # Checking if the tensor is 2D or 3D
    if len(seg.shape) == 2:  # 2D case
        # boundary = seg - np.pad(seg, ((1, 1), (1, 1)), mode='constant')[1:-1, 1:-1]
        boundary = find_boundaries(seg, mode='inner')
    elif len(seg.shape) == 3:  # 3D case
        for i in range(seg.shape[0]):  # Iterate through slices (depth)
            # boundary[i] = seg[i] - np.pad(seg[i], ((1, 1), (1, 1)), mode='constant')[1:-1, 1:-1]
            boundary[i] = find_boundaries(seg[i], mode='inner')
    else:
        raise ValueError(f"Unsupported tensor shape for boundary extraction: {seg.shape}")
    
    boundary_points = np.argwhere(boundary)
    # print(f"Boundary points (non-zero): {np.count_nonzero(boundary)}")  # Debugging output
    return boundary_points

def average_hausdorff_distance(label: Tensor, pred: Tensor) -> float:
    """
    Computes the Average Hausdorff Distance between the boundary points of label and pred.
    """
    assert label.shape == pred.shape
    assert sset(label, [0, 1])
    assert sset(pred, [0, 1])

    label_boundary = boundary_points(label)
    pred_boundary = boundary_points(pred)
    if len(label_boundary.shape) == 2:
        # Compute directed Hausdorff distance in both directions and average them
        forward_hausdorff = directed_hausdorff(label_boundary, pred_boundary)[0]
        backward_hausdorff = directed_hausdorff(pred_boundary, label_boundary)[0]
        # print(f"Forward Hausdorff Distance: {forward_hausdorff}")
        # print(f"Backward Hausdorff Distance: {backward_hausdorff}")

        ahd = (forward_hausdorff + backward_hausdorff) / 2
    else:
         ahd = np.zeros(label_boundary.shape[0])
         for i in range(label_boundary.shape[0]):
            # Compute directed Hausdorff distance in both directions and average them
            forward_hausdorff = directed_hausdorff(label_boundary[i], pred_boundary[i])[0]
            backward_hausdorff = directed_hausdorff(pred_boundary[i], label_boundary[i])[0]
            # print(f"Forward Hausdorff Distance: {forward_hausdorff}")
            # print(f"Backward Hausdorff Distance: {backward_hausdorff}")

            ahd[i] = (forward_hausdorff + backward_hausdorff) / 2

    return ahd

def average_symmetric_surface_distance(label: Tensor, pred: Tensor) -> float:
    assert label.shape == pred.shape
    assert sset(label, [0, 1])
    assert sset(pred, [0, 1])

    label_boundary = boundary_points(label)
    pred_boundary = boundary_points(pred)

    #If either boundary is empty, return 0 as there's nothing to compare
    if len(label_boundary) == 0 or len(pred_boundary) == 0:
        return 0.0

    #Computing the directed distances in both directions
    forward_distances = np.min(np.linalg.norm(label_boundary[:, None] - pred_boundary[None, :], axis=2), axis=1)
    backward_distances = np.min(np.linalg.norm(pred_boundary[:, None] - label_boundary[None, :], axis=2), axis=1)

    #Computing the average of both sets of distances
    assd = (forward_distances.mean() + backward_distances.mean()) / 2

    return assd




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
