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
from typing import Callable, Union

import torch
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
    """
    Minimal 2D/2.5D dataset.

    - 2D (default): returns a single image tensor of shape (C, H, W) and (optionally) its GT.
    - 2.5D: returns a channel-stacked window of size `num_slices` centered on the current slice,
      shape (C * num_slices, H, W). GT remains the center-slice supervision.

    Assumes filenames like: Patient_03_0007.png
      -> series key: 'Patient_03'
      -> slice index: 7
    """
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False,
                 two_point_five_d: bool = False, num_slices: int = 3):

        self.two_point_five_d: bool = two_point_five_d
        self.num_slices: int = num_slices
        if self.two_point_five_d:
            assert self.num_slices >= 3 and self.num_slices % 2 == 1, \
                f"num_slices must be odd and >=3, got {self.num_slices}"

        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        # Build per-series lookup for 2.5D windows (kept lightweight)
        if self.two_point_five_d:
            self._build_series_index()

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        # Build input image tensor: single slice (2D) or channel-stacked window (2.5D)
        if self.two_point_five_d:
            radius = self.num_slices // 2
            window_indices = self._series_window_indices(index, radius)

            imgs = []
            for wi in window_indices:
                wi_img_path, _ = self.files[wi]
                im = self.img_transform(Image.open(wi_img_path))  # expected (C, H, W), typically C=1
                imgs.append(im)
            img = torch.cat(imgs, dim=0)  # (C * num_slices, H, W)
        else:
            img: Tensor = self.img_transform(Image.open(img_path))  # (C, H, W)

        data_dict = {"images": img, "stems": img_path.stem}

        # Center-slice supervision only (unchanged)
        if not self.test_mode:
            gt: Tensor = self.gt_transform(Image.open(gt_path))
            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)
            data_dict["gts"] = gt

        return data_dict

    # ---------- helpers for series grouping/windowing ----------

    @staticmethod
    def _series_key_and_index(stem: str) -> tuple[str, int]:
        """
        For names like 'Patient_03_0007' -> ('Patient_03', 7).
        Assumes the last underscore + digits encodes the slice index.
        """
        key, idx_str = stem.rsplit('_', 1)
        return key, int(idx_str)

    def _build_series_index(self) -> None:
        """
        Build mappings to safely form 2.5D windows within a series (patient):
          - self.series_to_indices: dict[series_key] -> ordered list of dataset indices for that series
          - self.index_to_series_pos: dict[dataset_idx] -> (series_key, position_in_series)
        Sorting within a series uses the numeric slice index parsed from the filename.
        """
        tmp: dict[str, list[tuple[int, int]]] = {}
        for i, (img_path, _gt) in enumerate(self.files):
            key, zi = self._series_key_and_index(img_path.stem)
            tmp.setdefault(key, []).append((i, zi))

        self.series_to_indices: dict[str, list[int]] = {}
        self.index_to_series_pos: dict[int, tuple[str, int]] = {}

        for key, lst in tmp.items():
            lst.sort(key=lambda t: t[1])  # sort by numeric z
            ordered_idxs = [i for i, _ in lst]
            self.series_to_indices[key] = ordered_idxs
            for pos, idx in enumerate(ordered_idxs):
                self.index_to_series_pos[idx] = (key, pos)

    def _series_window_indices(self, center_idx: int, radius: int) -> list[int]:
        """
        Return a centered window of dataset indices within the same series.
        Boundary handling uses edge-replication (clamp).
        """
        key, pos = self.index_to_series_pos[center_idx]
        seq = self.series_to_indices[key]
        L = len(seq)

        window: list[int] = []
        for off in range(-radius, radius + 1):
            p = pos + off
            if p < 0:
                p = 0
            elif p >= L:
                p = L - 1
            window.append(seq[p])
        return window
