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

import random
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from contextlib import AbstractContextManager
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, cast

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import Tensor
import logging
from lightning import seed_everything

tqdm_ = partial(
    tqdm,
    dynamic_ncols=True,
    leave=True,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
)


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
    # seg.shape = [B, height, width]
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(
        1, seg[:, None, ...], 1
    )

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


# For reproducibility
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Get the device (MPS, CUDA, or CPU)
def get_device(use_gpu):
    if use_gpu:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(">> Picked MPS (Apple Silicon GPU) to run experiments")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(">> Picked CUDA to run experiments")
        else:
            device = torch.device("cpu")
            print(">> CUDA/MPS not available, falling back to CPU")
    else:
        device = torch.device("cpu")
        print(f">> Picked CPU to run experiments")

    return device



# Print Args in columns
def print_args(args, num_columns=2):
    print(">>> Arguments:")
    # Convert args (Namespace) to dictionary
    args_dict = vars(args)
    
    # Find the maximum length of the keys and values to ensure proper alignment
    max_key_len = max(len(str(key)) for key in args_dict.keys())
    max_val_len = max(len(str(value)) for value in map(str, args_dict.values()))  # Convert values to strings for formatting
    
    # Prepare a format string for aligned output (key and value alignment)
    format_string = '{:<'+str(max_key_len)+'} : {:<'+str(max_val_len)+'}'

    # Get all the items in the dictionary
    items = list(args_dict.items())

    # Determine the number of rows based on the number of columns
    num_rows = (len(items) + num_columns - 1) // num_columns  # Ceiling division

    # Print the dictionary in the specified number of columns
    for row in range(num_rows):
        row_str = []
        for col in range(num_columns):
            index = row + col * num_rows
            if index < len(items):
                key, value = items[index]
                row_str.append(format_string.format(key, str(value)))  # Ensure value is converted to string
        print('; '.join(row_str))
    print()

# Avoid printing 'Seed set to 42' already in print_args
def set_seed(seed, workers=False):
    # Suppress PyTorch Lightning INFO logging (which includes the seed message)
    logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.WARNING)
    
    # Set the seed
    seed_everything(seed, workers=workers)
