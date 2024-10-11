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

from scipy.spatial.distance import directed_hausdorff
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
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


# Additional metrics
# IoU (Jaccard Index)
def meta_iou(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    union_size: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred]) - inter_size).type(torch.float32)

    ious: Tensor = (inter_size + smooth) / (union_size + smooth)

    return ious


iou_coef = partial(meta_iou, "bk...->bk")


# Hausdorff Distance Function
def boundary_points(tensor: torch.Tensor) -> torch.Tensor:
    """
    Extracts boundary points using the Sobel operator.
    This works by detecting the edges in the segmentation mask.

    Args:
        tensor (torch.Tensor): A binary mask of shape (B, C, H, W),
                               where B is the batch size, C is the number of classes.

    Returns:
        List[torch.Tensor]: A list of tensors containing boundary points coordinates for each sample in the batch.
    """
    # Use Sobel filter for edge detection (3x3 kernels)
    sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).unsqueeze(0).unsqueeze(0).to(tensor.device)
    sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0).to(tensor.device)

    boundaries = []
    # Apply Sobel filter for each channel separately
    for i in range(tensor.shape[1]):
        # Extract the i-th channel and add a singleton dimension to match conv2d input format
        channel = tensor[:, i:i + 1, :, :]
        grad_x = F.conv2d(channel.float(), sobel_x, padding=1)
        grad_y = F.conv2d(channel.float(), sobel_y, padding=1)

        # Magnitude of gradients (edge strength)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Threshold the gradient magnitude to get binary edges
        boundary = (grad_mag > 0).float()  # 1 for boundary, 0 for non-boundary
        boundaries.append(boundary)

    # Stack the boundaries back into a single tensor
    stacked_boundaries = torch.cat(boundaries, dim=1)

    # Extract boundary point coordinates for each item in the batch
    boundary_points_list = []
    for b in range(stacked_boundaries.shape[0]):
        boundary_points = []
        for c in range(stacked_boundaries.shape[1]):
            # Get non-zero coordinates (boundary points)
            coords = torch.nonzero(stacked_boundaries[b, c], as_tuple=False)
            if coords.size(0) > 0:
                boundary_points.append(coords)
        boundary_points_list.append(boundary_points)

    return boundary_points_list


def meta_hausdorff(sum_str: str, label: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    hausdorff_distances = []
    for class_idx in range(label.shape[1]):
        # Extract boundary points for each class
        label_boundaries = boundary_points(label[:, class_idx:class_idx + 1])
        pred_boundaries = boundary_points(pred[:, class_idx:class_idx + 1])

        batch_hausdorff = []
        for b in range(len(label_boundaries)):
            if len(label_boundaries[b]) > 0 and len(pred_boundaries[b]) > 0:
                # Convert the boundary points to numpy arrays
                label_points = label_boundaries[b][0].cpu().numpy()
                pred_points = pred_boundaries[b][0].cpu().numpy()

                # Compute Hausdorff Distance for each class
                hd_label_to_pred = directed_hausdorff(label_points, pred_points)[0]
                hd_pred_to_label = directed_hausdorff(pred_points, label_points)[0]

                batch_hausdorff.append(max(hd_label_to_pred, hd_pred_to_label))
            else:
                # If no boundary points exist for this class, set distance to zero
                batch_hausdorff.append(0.0)

        hausdorff_distances.append(batch_hausdorff)

    return torch.tensor(hausdorff_distances)


hausdorff_coef = partial(meta_hausdorff, "bk...->bk")


# ASSD
def meta_assd(sum_str: str, label: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    assd_distances = []
    for class_idx in range(label.shape[1]):
        # Extract boundary points for each class
        label_boundaries = boundary_points(label[:, class_idx:class_idx + 1])
        pred_boundaries = boundary_points(pred[:, class_idx:class_idx + 1])

        batch_assd = []
        for b in range(len(label_boundaries)):
            if len(label_boundaries[b]) > 0 and len(pred_boundaries[b]) > 0:
                # Convert the boundary points to numpy arrays
                label_points = label_boundaries[b][0].cpu().numpy()
                pred_points = pred_boundaries[b][0].cpu().numpy()

                # Compute ASSD for each class
                hd_label_to_pred = np.mean(
                    [np.min(np.linalg.norm(label_points - point, axis=1)) for point in pred_points])
                hd_pred_to_label = np.mean(
                    [np.min(np.linalg.norm(pred_points - point, axis=1)) for point in label_points])

                batch_assd.append((hd_label_to_pred + hd_pred_to_label) / 2)
            else:
                # If no boundary points exist for this class, set distance to zero
                batch_assd.append(0.0)

        assd_distances.append(batch_assd)

    return torch.tensor(assd_distances)


assd_coef = partial(meta_assd, "bk...->bk")


# Volumetric Similarity
def meta_vol_sim(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    label_vol = einsum(sum_str, [label]).type(torch.float32)
    pred_vol = einsum(sum_str, [pred]).type(torch.float32)

    vol_sim = 1 - (torch.abs(label_vol - pred_vol) / (label_vol + pred_vol + smooth))

    return vol_sim


vol_sim_coef = partial(meta_vol_sim, "bk...->bk")
