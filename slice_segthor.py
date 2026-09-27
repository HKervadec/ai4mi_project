#!/usr/bin/env python3.7

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

import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from skimage.exposure import equalize_adapthist
from skimage.io import imsave
from skimage.transform import resize

from utils import crop_or_pad_arr, map_, tqdm_


def compute_target_spacing(
    src_path: Path, training_ids: list[str]
) -> tuple[float, float, float]:
    """
    Compute median spacing (training set only)
    """
    spacings = []
    for id_ in training_ids:
        ct_path = src_path / "train" / id_ / f"{id_}.nii.gz"
        nib_obj = nib.load(str(ct_path))
        dx, dy, dz = nib_obj.header.get_zooms()
        spacings.append((dx, dy, dz))

    spacings = np.asarray(spacings)
    median_spacing = tuple(np.median(spacings, axis=0).tolist())

    return median_spacing

def compute_norm_stats(
    src_path: Path, training_ids: list[str]
) -> tuple[float, float]:
    """
    Compute percentile clip range from foreground voxels (training set only)
    """
    fg_values = []
    for id_ in training_ids:
        ct_path = src_path / "train" / id_ / f"{id_}.nii.gz"
        gt_path = src_path / "train" / id_ / "GT.nii.gz"

        ct = np.asarray(nib.load(str(ct_path)).dataobj)
        gt = np.asarray(nib.load(str(gt_path)).dataobj)
        fg_values.append(ct[gt > 0])

    fg_values = np.concatenate(fg_values)
    fg_p_low, fg_p_high = np.percentile(fg_values, [0.5, 99.5])

    return fg_p_low, fg_p_high

def resample_arr(
    img: np.ndarray, 
    original_spacing: tuple[float, float, float], 
    target_spacing: tuple[float, float, float],
    order: int,
) -> np.ndarray:
    """
    Resample the data to the target spacing
    """
    zoom_factors = [c / t for c, t in zip(original_spacing, target_spacing)]
    resampled = zoom(img, zoom_factors, order=order)
    return resampled

def norm_arr(
    img: np.ndarray, norm_stats: tuple[float, float]
) -> np.ndarray:
    """
    Clip voxels values to the foreground percentile range
    and normalize to [0, 1]
    """
    casted = img.astype(np.float32)

    fg_p_low, fg_p_high = norm_stats
    clipped = np.clip(casted, fg_p_low, fg_p_high)

    norm = (
        (clipped - fg_p_low) / (fg_p_high - fg_p_low)
    ) # normalize by training data stats

    # NOTE: changed since the normalization is done by training data stats,
    # so the min and max of the normalized image may not be exactly 0 and 1.
    # assert 0 == norm.min(), norm.min()
    # assert norm.max() == 1, norm.max()
    assert 0 <= norm.min(), norm.min()
    assert norm.max() <= 1, norm.max()

    return norm


def clahe_arr(img: np.ndarray) -> np.ndarray:
    """
    3D CLAHE on the normalized [0, 1] image
    """
    return equalize_adapthist(img) # default, untuned


def sanity_ct(ct, x, y, z, dx, dy, dz) -> bool:
    assert ct.dtype in [np.int16, np.int32], ct.dtype
    assert -1000 <= ct.min(), ct.min()
    assert ct.max() <= 31743, ct.max()

    assert 0.896 <= dx <= 1.37, dx  # Rounding error
    assert dx == dy
    assert 2 <= dz <= 3.7, dz

    assert (x, y) == (512, 512)
    assert x == y
    assert 135 <= z <= 284, z

    return True


def sanity_gt(gt, ct) -> bool:
    assert gt.shape == ct.shape
    assert gt.dtype in [np.uint8], gt.dtype

    # Do the test on 3d: assume all organs are present..
    # assert set(np.unique(gt)) == set(range(5))

    return True


resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)


def slice_patient(
    id_: str,
    dest_path: Path,
    source_path: Path,
    shape: tuple[int, int],
    norm_stats: tuple[float, float],
    target_spacing: tuple[float, float, float],
    crop_size: int,
    test_mode: bool = False,
) -> tuple[float, float, float]:
    id_path: Path = source_path / ("train" if not test_mode else "test") / id_

    ct_path: Path = (id_path / f"{id_}.nii.gz") if not test_mode else (source_path / "test" / f"{id_}.nii.gz")
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    # dx, dy, dz = nib_obj.header.get_zooms()
    # dx, dy, dz = nib_obj.header.get_zooms()

    assert sanity_ct(ct, *ct.shape, *nib_obj.header.get_zooms())

    gt: np.ndarray
    if not test_mode:
        gt_path: Path = id_path / "GT.nii.gz"
        gt_nib = nib.load(str(gt_path))
        # print(nib_obj.affine, gt_nib.affine)
        gt = np.asarray(gt_nib.dataobj)
        assert sanity_gt(gt, ct)
    else:
        gt = np.zeros_like(ct, dtype=np.uint8)

    ct, gt = (
        resample_arr(ct, nib_obj.header.get_zooms(), target_spacing, order=3), # cubic interpolation
        resample_arr(gt, nib_obj.header.get_zooms(), target_spacing, order=0) # nearest neighbor interpolation
    ) # both use nib_obj since ct and gt have same spacing.

    # Make sure the final resize has the same scale for all patients
    n_organ = (gt > 0).sum()
    ct = crop_or_pad_arr(ct, (crop_size, crop_size), value=-1000) # pad with air
    gt = crop_or_pad_arr(gt, (crop_size, crop_size), value=0)
    assert (gt > 0).sum() == n_organ # the crop must not cut any organ

    x, y, z = ct.shape
    norm_ct: np.ndarray = norm_arr(ct, norm_stats)
    clahe_ct: np.ndarray = clahe_arr(norm_ct)

    to_slice_ct = (255 * clahe_ct).astype(np.uint8) # convert to uint8 for saving as png
    to_slice_gt = gt

    for idz in range(z):
        img_slice = resize_(to_slice_ct[:, :, idz], shape).astype(np.uint8)
        gt_slice = resize_(to_slice_gt[:, :, idz], shape, order=0).astype(np.uint8)
        assert img_slice.shape == gt_slice.shape
        gt_slice *= 63
        assert gt_slice.dtype == np.uint8, gt_slice.dtype
        # assert set(np.unique(gt_slice)) <= set(range(5))
        assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)

        arrays: list[np.ndarray] = [img_slice, gt_slice]

        subfolders: list[str] = ["img", "gt"]
        assert len(arrays) == len(subfolders)
        for save_subfolder, data in zip(subfolders,
                                        arrays):
            filename = f"{id_}_{idz:04d}.png"

            save_path: Path = Path(dest_path, save_subfolder)
            save_path.mkdir(parents=True, exist_ok=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(str(save_path / filename), data)

    return nib_obj.header.get_zooms()


def get_splits(src_path: Path, retains: int, fold: int) -> tuple[list[str], list[str], list[str]]:
    ids: list[str] = sorted(map_(lambda p: p.name, (src_path / 'train').glob('*')))
    print(f"Founds {len(ids)} in the id list")
    print(ids[:10])
    assert len(ids) > retains

    random.shuffle(ids)  # Shuffle before to avoid any problem if the patients are sorted in any way
    validation_slice = slice(fold * retains, (fold + 1) * retains)
    validation_ids: list[str] = ids[validation_slice]
    assert len(validation_ids) == retains

    training_ids: list[str] = [e for e in ids if e not in validation_ids]
    assert (len(training_ids) + len(validation_ids)) == len(ids)

    test_ids: list[str] = sorted(map_(lambda p: Path(p.stem).stem, (src_path / 'test').glob('*')))
    print(f"Founds {len(test_ids)} test ids")
    print(test_ids[:10])

    return training_ids, validation_ids, test_ids


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the clean up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    training_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    training_ids, validation_ids, test_ids = get_splits(src_path, args.retains, args.fold)

    norm_stats = compute_norm_stats(src_path, training_ids) # only use training ids
    target_spacing = compute_target_spacing(src_path, training_ids) # only use training ids

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    split_ids: list[str]
    for mode, split_ids in zip(["train", "val"], [training_ids, validation_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=tuple(args.shape),
                                 norm_stats=norm_stats,
                                 target_spacing=target_spacing,
                                 crop_size=args.crop_size,
                                 test_mode=mode == 'test')
        resolutions: list[tuple[float, float, float]]
        iterator = tqdm_(split_ids)
        match args.process:
            case 1:
                resolutions = list(map(pfun, iterator))
            case -1:
                resolutions = Pool().map(pfun, iterator)
            case _ as p:
                resolutions = Pool(p).map(pfun, iterator)

        for key, val in zip(split_ids, resolutions):
            resolution_dict[key] = val

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")

    # Save preprocessing stats for stitching later
    stats = {
        "norm_stats": {"p_low": float(norm_stats[0]), "p_high": float(norm_stats[1])},
        "target_spacing": {"dx": float(target_spacing[0]), "dy": float(target_spacing[1]), "dz": float(target_spacing[2])},
        "crop_size": args.crop_size,
    }
    with open(dest_path / "preprocess_stats.pkl", 'wb') as f:
        pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved preprocessing stats to {f}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--crop_size', type=int, default=512,
                        help="Size to center crop / pad to (at target spacing), before resizing to --shape")
    parser.add_argument('--retains', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="The number of cores to use for processing")
    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
