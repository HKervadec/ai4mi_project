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
    """Gives 2D slices or 2.5D stacks, depending on the 'neighbours' param.

        neighbours=0  -> 2D    img (1, W, H)
        neighbours=n  -> 2.5D  img (2n+1, W, H), gt of the centre slice
    """
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False,
                 neighbours: int = 0):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize
        self.neighbours: int = neighbours

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        # First and last index of each patient, so a window never crosses volumes
        self.bounds: dict[str, tuple[int, int]] = {}
        for i, (img_path, _) in enumerate(self.files):
            pid: str = img_path.stem.rsplit('_', 1)[0]
            lo, _ = self.bounds.get(pid, (i, i))
            self.bounds[pid] = (lo, i)

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def window(self, index: int) -> list[int]:
        # Clamped at the volume edges: the first slice is simply repeated
        lo, hi = self.bounds[self.files[index][0].stem.rsplit('_', 1)[0]]
        return [min(max(j, lo), hi)
                for j in range(index - self.neighbours, index + self.neighbours + 1)]

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]
        idxs: list[int] = self.window(index) if self.neighbours else [index]

        imgs: list[Tensor] = [self.img_transform(Image.open(self.files[j][0])) for j in idxs]
        img: Tensor = torch.cat(imgs, dim=0)

        data_dict = {"images": img,
                     "stems": img_path.stem}  # Always the stem of the centre slice

        if not self.test_mode:
            gts: list[Tensor] = [self.gt_transform(Image.open(self.files[j][1])) for j in idxs]
            gt: Tensor = gts[len(gts) // 2]

            assert gt.shape[1:] == img.shape[1:], (gt.shape, img.shape)

            data_dict["gts"] = gt

        return data_dict


if __name__ == '__main__':
    # The windows must clamp at the volume edges and never cross into another patient
    ds = SliceDataset.__new__(SliceDataset)
    ds.neighbours = 2
    ds.files = [(Path(f"{pid}_{z:04d}.png"), None)
                for pid in ['Patient_01', 'Patient_02'] for z in range(5)]
    ds.bounds = {'Patient_01': (0, 4), 'Patient_02': (5, 9)}

    assert ds.window(0) == [0, 0, 0, 1, 2]  # start of a volume
    assert ds.window(2) == [0, 1, 2, 3, 4]  # middle
    assert ds.window(4) == [2, 3, 4, 4, 4]  # end, does not leak into Patient_02
    assert ds.window(5) == [5, 5, 5, 6, 7]  # start of the next volume
    assert all(ds.window(i)[ds.neighbours] == i for i in range(10))  # centre is the slice itself
    print("windowing ok")
