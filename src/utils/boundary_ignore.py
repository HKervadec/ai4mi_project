# Usage:
#   1) Generate ignore masks once, after slice_segthor.py:
#        python -m src.utils.boundary_ignore --sliced_dir <path/to/sliced/dataset> [--num_classes 5] [--min_area 20] [--buffer 1]
#      This scans train/gt and val/gt, flags any organ present with area >= min_area at a
#      patient's first/last z-slice (likely field-of-view truncation, not a real anatomical end),
#      and writes bitmask PNGs to an "ignore" subfolder (bit k set => class k excluded at that pixel).
#   2) In main.py, use MaskedSliceDataset instead of SliceDataset:
#        from src.utils.boundary_ignore import MaskedSliceDataset, ignore_transform, MaskedCrossEntropy
#        dataset = MaskedSliceDataset("train", root_dir, img_transform=img_transform,
#            gt_transform=partial(gt_transform, K), ignore_transform=partial(ignore_transform, K))
#        loss_fn = MaskedCrossEntropy(idk=[0, 1, 3, 4])
#        loss = loss_fn(pred_probs, gt, data["ignore"].to(device))

import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Callable, Union

import numpy as np
import torch
from PIL import Image
from torch import einsum, Tensor
from skimage.io import imread, imsave

from utils.dataset import SliceDataset
from utils.utils import simplex, sset


SLICE_RE = re.compile(r"^(?P<id>.+)_(?P<idz>\d{4})\.png$")


def group_by_patient(gt_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in gt_dir.glob("*.png"):
        match = SLICE_RE.match(path.name)
        if match is None:
            continue
        groups[match.group("id")].append((int(match.group("idz")), path))
    for id_ in groups:
        groups[id_].sort(key=lambda pair: pair[0])
    return groups


def class_areas(gt_slice: np.ndarray, num_classes: int, step: int) -> list[int]:
    return [int(np.count_nonzero(gt_slice == k * step)) for k in range(num_classes)]


def flagged_classes(areas: list[int], min_area: int) -> set[int]:
    return {k for k, area in enumerate(areas) if k > 0 and area >= min_area}


def build_ignore_slices(
    slices: list[tuple[int, Path]],
    num_classes: int,
    step: int,
    min_area: int,
    buffer: int,
) -> dict[int, set[int]]:
    ignore: dict[int, set[int]] = defaultdict(set)
    n = len(slices)

    first_flagged = flagged_classes(
        class_areas(imread(slices[0][1]), num_classes, step), min_area
    )
    last_flagged = flagged_classes(
        class_areas(imread(slices[-1][1]), num_classes, step), min_area
    )

    for offset in range(min(buffer + 1, n)):
        ignore[slices[offset][0]] |= first_flagged

    for offset in range(min(buffer + 1, n)):
        ignore[slices[n - 1 - offset][0]] |= last_flagged

    return ignore


def write_ignore_masks(
    slices: list[tuple[int, Path]],
    ignore: dict[int, set[int]],
    id_: str,
    dest_dir: Path,
    shape: tuple[int, int],
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for idz, _ in slices:
        mask = np.zeros(shape, dtype=np.uint8)
        for k in ignore.get(idz, set()):
            mask |= np.uint8(1 << k)
        imsave(str(dest_dir / f"{id_}_{idz:04d}.png"), mask, check_contrast=False)


def process_subset(subset_dir: Path, num_classes: int, min_area: int, buffer: int) -> None:
    gt_dir = subset_dir / "gt"
    ignore_dir = subset_dir / "ignore"
    if not gt_dir.exists():
        return

    step = 255 // (num_classes - 1)
    groups = group_by_patient(gt_dir)

    for id_, slices in groups.items():
        sample = imread(slices[0][1])
        ignore = build_ignore_slices(slices, num_classes, step, min_area, buffer)
        write_ignore_masks(slices, ignore, id_, ignore_dir, sample.shape)


def ignore_transform(K: int, img: Image.Image) -> Tensor:
    arr = np.array(img)
    keep = np.ones((K, *arr.shape), dtype=np.float32)
    for k in range(1, K):
        keep[k][(arr & (1 << k)) != 0] = 0.0
    return torch.tensor(keep, dtype=torch.float32)


class MaskedSliceDataset(SliceDataset):
    def __init__(
        self,
        subset,
        root_dir,
        img_transform,
        gt_transform,
        ignore_transform,
        augment=False,
        equalize=False,
        debug=False,
    ):
        super().__init__(
            subset,
            root_dir,
            img_transform,
            gt_transform,
            augment=augment,
            equalize=equalize,
            debug=debug,
        )
        self.ignore_transform: Callable = ignore_transform

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        data_dict = super().__getitem__(index)

        if not self.test_mode:
            img_path, _ = self.files[index]
            ignore_path = img_path.parent.parent / "ignore" / img_path.name
            data_dict["ignore"] = self.ignore_transform(Image.open(ignore_path))

        return data_dict


class MaskedCrossEntropy:
    def __init__(self, **kwargs):
        self.idk = kwargs["idk"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax: Tensor, weak_target: Tensor, ignore: Tensor) -> Tensor:
        assert pred_softmax.shape == weak_target.shape == ignore.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])
        assert sset(ignore, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float() * ignore[:, self.idk, ...].float()

        loss = -einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class MaskedPartialCrossEntropy(MaskedCrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


def main(args: argparse.Namespace) -> None:
    sliced_dir = Path(args.sliced_dir)
    for subset in ["train", "val"]:
        process_subset(sliced_dir / subset, args.num_classes, args.min_area, args.buffer)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Boundary-truncation ignore mask generator")
    parser.add_argument("--sliced_dir", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--min_area", type=int, default=20)
    parser.add_argument("--buffer", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
