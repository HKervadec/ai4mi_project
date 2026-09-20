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
from typing import Callable, Union, Optional
import numpy as np
import random
import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as TF
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False,
                 is_25d: bool = False, crop_size: Optional[int] = 192,      # ADDED Dimensionality switch,
                 filter_empty: bool = True):                  # optional 192x192 center crop (None: if you need to run the basic), and filtering out empty slices
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize
        self.is_25d: bool = is_25d     # ADDED dimensionality attribute 
        self.crop_size: Optional[int] = crop_size if crop_size and crop_size > 0 else None     # ADDED the optional crop
        self.crop_transform = (
            transforms.CenterCrop(size=(self.crop_size, self.crop_size)) if self.crop_size else None
        )
        self.test_mode: bool = subset == 'test'

        raw_files = make_dataset(root_dir, subset)  # load raw file pairs
        # Filter empty background slices (applied to train only)
        if filter_empty and subset == 'train':                   # conditional slice filtering
            self.files = []                                      # filtered path list initialization
            for img_p, gt_p in raw_files:                        # iterate over dataset paths
                if np.array(Image.open(gt_p)).sum() > 0:         # check for non-zero organ mask
                    self.files.append((img_p, gt_p))             # keep slice if foreground exists
        else:                                                    # fallback for test set or unfiltered run
            self.files = raw_files                               # keep full raw file list
       # self.files = make_dataset(root_dir, subset)

        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)
    
    def _maybe_crop(self, tensor: Tensor) -> Tensor:      # ADDED crop switch to tensor
        if self.crop_transform is not None:
            return self.crop_transform(tensor)
        return tensor

    
    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]
        gt_pil_aug = None


            img: Tensor = self._maybe_crop(self.img_transform(pil_curr))

            img: Tensor = self._maybe_crop(torch.cat([slice_prev, slice_curr, slice_next], dim=0))
        

        data_dict = {"images": img,
                     "stems": img_path.stem}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(Image.open(gt_path))

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        return data_dict
