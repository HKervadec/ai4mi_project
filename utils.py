#!/usr/bin/env python3
import argparse
from argparse import Namespace
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

import wandb
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch import Tensor, einsum
from scipy.spatial.distance import directed_hausdorff
from skimage.segmentation import find_boundaries

import os  #TODO: remove on final submission

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

    # TODO check why
    seg = seg.to(torch.int64)  #casting the seg tensor to torch.int64 when encoding the gt for metrics

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


def compute_precision(label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    """
    Precision as TP / (TP + FP).
    """
    # assert label.shape == pred.shape
    # assert one_hot(label)
    # assert one_hot(pred)

    # True positives (TP)
    inter_size: Tensor = einsum("bk...->bk", [intersection(label, pred)]).type(torch.float32)
    # Predicted positives (TP + FP)
    total_pixel_pred = einsum("bk...->bk", [pred]).type(torch.float32)

    precision = (inter_size + smooth) / (total_pixel_pred + smooth)
    return precision


def compute_recall(label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    """
    Recall = TP / (TP + FN)
    """
    # assert label.shape == pred.shape
    # assert one_hot(label)
    # assert one_hot(pred)

     # True positives (TP)
    inter_size: Tensor = einsum("bk...->bk", [intersection(label, pred)]).type(torch.float32)
    # Actual positives (TP + FN)
    total_pixel_truth = einsum("bk...->bk", [label]).type(torch.float32)

    recall = (inter_size + smooth) / (total_pixel_truth + smooth)
    return recall


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
    Note: Averaged Version!
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

        if forward_hausdorff > 1000:
            if backward_hausdorff > 1000:
                ahd = 1000
            else:
                ahd = backward_hausdorff
        else:
            if backward_hausdorff > 1000:
                ahd = forward_hausdorff
            else:
                ahd = (forward_hausdorff + backward_hausdorff) / 2
    else:
         ahd = torch.zeros(label_boundary.shape[0])
         for i in range(label_boundary.shape[0]):
            # Compute directed Hausdorff distance in both directions and average them
            forward_hausdorff = directed_hausdorff(label_boundary[i], pred_boundary[i])[0]
            backward_hausdorff = directed_hausdorff(pred_boundary[i], label_boundary[i])[0]
            # print(f"Forward Hausdorff Distance: {forward_hausdorff}")
            # print(f"Backward Hausdorff Distance: {backward_hausdorff}")
            if forward_hausdorff > 1000:
                if backward_hausdorff > 1000:
                      ahd[i] = 1000
                else:
                    ahd[i] = backward_hausdorff
            else:
                if backward_hausdorff > 1000:
                    ahd[i] = forward_hausdorff
                else:
                    ahd[i] = (forward_hausdorff + backward_hausdorff) / 2

    return ahd

def average_hausdorff_distance_per_class(label: Tensor, pred: Tensor, K: int) -> List[float]:
    """
    Computes the Average Hausdorff Distance (AHD) for each class separately.
    The AHD for each class is calculated based on the one-hot encoded ground truth and predicted masks.
    """
    assert label.shape == pred.shape
    assert label.shape[0] == K

    ahd_per_class = []

    for k in range(K):
        # Extract the boundary points for each class (k)
        label_boundary = boundary_points(label[k])
        pred_boundary = boundary_points(pred[k])

        if len(label_boundary) == 0 or len(pred_boundary) == 0:
            # If either the ground truth or predicted boundary is empty, skip this class
            ahd_per_class.append(float('inf'))
            continue

        # Compute directed Hausdorff distance in both directions and average them
        forward_hausdorff = directed_hausdorff(label_boundary, pred_boundary)[0]
        backward_hausdorff = directed_hausdorff(pred_boundary, label_boundary)[0]

        if forward_hausdorff > 1000:
            ahd = backward_hausdorff if backward_hausdorff <= 1000 else 1000
        elif backward_hausdorff > 1000:
            ahd = forward_hausdorff
        else:
            ahd = (forward_hausdorff + backward_hausdorff) / 2

        ahd_per_class.append(ahd)

    return ahd_per_class

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

def visualize_sample(gt: Tensor, pred: Tensor, epoch: int, batch_idx: int):
    """
    Visualize a sample of the ground truth and predicted masks.
    """

    # Summing along the channel axis to combine all classes into a single mask
    gt_combined = torch.argmax(gt, dim=0).cpu().numpy()  # Combine classes in ground truth
    pred_combined = torch.argmax(pred, dim=0).cpu().numpy()  # Combine classes in prediction

    plt.figure(figsize=(12, 6))

    # Plot Ground Truth
    plt.subplot(1, 2, 1)
    plt.imshow(gt_combined, cmap='gray')
    plt.title(f"Ground Truth (Epoch {epoch}, Batch {batch_idx})")

    # Plot Prediction
    plt.subplot(1, 2, 2)
    plt.imshow(pred_combined, cmap='gray')
    plt.title(f"Predicted Mask (Epoch {epoch}, Batch {batch_idx})")


    # Save the visualization to file
    save_path = Path("results/segthor/ce")  / f"epoch_{epoch}_batch_{batch_idx}.png"
    plt.savefig(save_path)
    plt.close()  # Close the figure to avoid memory issues


def wandb_login(disable_wandb: bool):
    if disable_wandb:
        print("!! WandB disabled !!")
        return
    else:
        try:
            with open("wandb.password", "rt") as f:
                pw = f.readline().strip()
                os.environ["WANDB_API_KEY"] = pw
                wandb.login()
        except FileNotFoundError:
            raise FileNotFoundError("File wandb.password was not found in the project root. Either add it or disable wandb by running --disable_wandb")


def wandb_save_model(disabled: bool, model_path, metadata: dict):
    if disabled:
        print(f"WandB disabled, will not save the model weights to its artifacts!")
        pass
    else:
        artifact = wandb.Artifact("bestmodel.pt", type='model', metadata=metadata)
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)
        print("Saved model weights to WandB artifacts")


# K - num of classes, e - epoch num
# Loss and every metric are an array of [train_result, valid_result]
def save_loss_and_metrics(K: int, e: int, dest: Path,
                          loss: [Tensor, Tensor],
                          dice: [Tensor, Tensor],
                          jaccard: [Tensor, Tensor],
                          precision: [Tensor, Tensor],
                          recall: [Tensor, Tensor],
                          ahd_validation: Tensor,
                          assd_validation: Tensor) -> dict:

    # Save and log metrics and losses

    # Save the validation results, for visualization purposes
    np.save(dest / "loss_val.npy", loss[1])
    np.save(dest / "dice_val.npy", dice[1])
    np.save(dest / "jaccard_val.npy", jaccard[1])
    np.save(dest / "precision_val.npy", precision[1])
    np.save(dest / "recall_val.npy", recall[1])
    np.save(dest / "ahd_val.npy", ahd_validation)
    np.save(dest / "assd_val.npy", assd_validation)

    metrics = {
        "train/loss": loss[0][e, :].mean().item(),
        "train/dice_avg": dice[0][e, :, 1:].mean().item(),
        "train/jaccard": jaccard[0][e, :, 1:].mean().item(),
        "train/precision": precision[0][e, :, 1:].mean().item(),
        "train/recall": recall[0][e, :, 1:].mean().item(),

        "valid/loss": loss[1][e, :].mean().item(),
        "valid/dice_avg": dice[1][e, :, 1:].mean().item(),
        "valid/jaccard": jaccard[1][e, :, 1:].mean().item(),
        "valid/precision": precision[1][e, :, 1:].mean().item(),
        "valid/recall": recall[1][e, :, 1:].mean().item(),
        "valid/ahd": ahd_validation[e, :, 1:].mean().item(),
        "valid/assd": assd_validation[e, :].mean().item(),
    }
    for k in range(1, K):
        metrics[f"train/dice-{k}"] = dice[0][e, :, k].mean().item()
        metrics[f"valid/dice-{k}"] = dice[1][e, :, k].mean().item()
        metrics[f"train/jaccard-{k}"] = jaccard[0][e, :, k].mean().item()
        metrics[f"valid/jaccard-{k}"] = jaccard[1][e, :, k].mean().item()
        metrics[f"train/precision-{k}"] = precision[0][e, :, k].mean().item()
        metrics[f"valid/precision-{k}"] = precision[1][e, :, k].mean().item()
        metrics[f"train/recall-{k}"] = recall[0][e, :, k].mean().item()
        metrics[f"valid/recall-{k}"] = recall[1][e, :, k].mean().item()
        metrics[f"valid/ahd-{k}"] = ahd_validation[e, :, k].mean().item()
    return metrics


def get_run_name(args: Namespace, parser: argparse.ArgumentParser) -> str:
    prefix = args.run_prefix + '_' if args.run_prefix else ''
    lr = f'lr({"{:.0E}".format(args.lr)})_' if args.lr != parser.get_default('lr') else ''
    lr += f'LR-WD({args.lr_weight_decay})_' if args.lr_weight_decay != parser.get_default('lr_weight_decay') else ''
    dropout = f'dropout({args.dropoutRate})' if args.dropoutRate != parser.get_default('dropoutRate') else ''
    encoder_name = ''
    if args.model != 'ENet':
        encoder_name = f'_{args.encoder_name}'
        if args.unfreeze_enc_last_n_layers != parser.get_default('unfreeze_enc_last_n_layers'):
            encoder_name += f'(unfreeze-{args.unfreeze_enc_last_n_layers})'
    run_name = f'{prefix}{dropout}{lr}{args.loss}_{args.model}{encoder_name}'
    run_name = 'DEBUG_' + run_name if args.debug else run_name
    return run_name


# Using the suggestions from https://pytorch.org/docs/stable/notes/randomness.html
def seed_everything(args) -> None:
    import os

    seed = args.seed
    print(f"> Using seed: {seed}")
    # Seed python
    random.seed(seed)
    # Seed numpy
    np.random.seed(seed)
    # Seed torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Cannot set this since there are some operations which do not have deterministic equivalent (like max_unpool2d)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    # For the cuBLAS API of the CUDA implementation
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)