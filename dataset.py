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
from typing import Callable, Union, List, Tuple
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


def make_dataset(root, subset) -> list[tuple[Path, Path]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images = sorted(img_path.glob("*.png"))
    full_labels = sorted(full_path.glob("*.png"))

    return list(zip(images, full_labels))

class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False, remove_background=False):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize
        self.remove_background: bool = remove_background

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        # If the flag to remove background-only slices is set, filter the files
        if self.remove_background:
            self.files = self._filter_background_only_slices()

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)
    
    def _filter_background_only_slices(self) -> List[Tuple[Path, Path]]:
        """
        Filter out the image-ground truth pairs where the ground truth contains only background (label 0).
        """
        filtered_files = []
        for img_path, gt_path in self.files:
            gt = np.array(Image.open(gt_path))  # Load ground truth image
            if np.any(gt > 0):  # Keep if any pixel in the ground truth is not background (label > 0)
                filtered_files.append((img_path, gt_path))
        
        return filtered_files

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor = self.img_transform(Image.open(img_path))
        gt: Tensor = self.gt_transform(Image.open(gt_path))

        _, W, H = img.shape
        K, _, _ = gt.shape
        assert gt.shape == (K, W, H)

        return {"images": img,
                "gts": gt,
                "stems": img_path.stem}
    

def make_dataset_from_img_gt_dirs(img_dir, gt_dir) -> list[tuple[Path, Path]]:
    """
    Creates a dataset by pairing images and ground truth files from given directories.
    img_dir: Path to the directory containing the images.
    gt_dir: Path to the directory containing the ground truth labels.
    """
    # Ensure both directories are valid and contain .png files
    img_dir = Path(img_dir)
    gt_dir = Path(gt_dir)

    # Get sorted lists of image and ground truth files
    images = sorted(img_dir.glob("*.png"))
    full_labels = sorted(gt_dir.glob("*.png"))

    # Return a list of tuples (image_path, ground_truth_path)
    return list(zip(images, full_labels))

class SliceDatasetWithTransforms(Dataset):
    def __init__(self, subset, img_dirs, gt_dirs, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False, remove_background=False):
        """
        img_dirs: List of image directories (e.g., ['img', 'img_spatial_aug', 'img_intensity_aug'])
        gt_dirs: List of ground truth directories (e.g., ['gt', 'gt_spatial_aug', 'gt_intensity_aug'])
        """
        self.img_dirs = img_dirs
        self.gt_dirs = gt_dirs
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize
        self.remove_background: bool = remove_background

        # Combine the datasets from all the directories
        self.files = []
        if img_dirs and gt_dirs:
            for img_dir, gt_dir in zip(img_dirs, gt_dirs):
                self.files += make_dataset_from_img_gt_dirs(img_dir, gt_dir)

        # If the flag to remove background-only slices is set, filter the files
        if self.remove_background:
            self.files = self._filter_background_only_slices()

        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def _filter_background_only_slices(self) -> List[Tuple[Path, Path]]:
        """
        Filter out the image-ground truth pairs where the ground truth contains only background (label 0).
        """
        filtered_files = []
        for img_path, gt_path in self.files:
            gt = np.array(Image.open(gt_path))  # Load ground truth image
            if np.any(gt > 0):  # Keep if any pixel in the ground truth is not background (label > 0)
                filtered_files.append((img_path, gt_path))
        
        return filtered_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor = self.img_transform(Image.open(img_path))
        gt: Tensor = self.gt_transform(Image.open(gt_path))

        _, W, H = img.shape
        K, _, _ = gt.shape
        assert gt.shape == (K, W, H)

        return {"images": img,
                "gts": gt,
                "stems": img_path.stem}