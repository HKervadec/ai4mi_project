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
from skimage.io import imsave
from skimage.transform import resize

from utils import map_, tqdm_
from preprocessing import apply_hu_window, center_crop_pad, resample_volume, MEDIASTINAL
from experiments import EXPERIMENTS, get as get_experiment, SliceConfig


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    assert 0 == res.min(), res.min()
    assert res.max() == 255, res.max()

    return res.astype(np.uint8)


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


def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int],
                  test_mode: bool = False,
                  window: tuple[float, float] | None = MEDIASTINAL,
                  target_spacing: tuple[float, float, float] | None = None) -> tuple[float, float, float]:
    id_path: Path = source_path / ("train" if not test_mode else "test") / id_

    ct_path: Path = (id_path / f"{id_}.nii.gz") if not test_mode else (source_path / "test" / f"{id_}.nii.gz")
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    # dx, dy, dz = nib_obj.header.get_zooms()
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()

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

    # Spatial preprocessing: resample to a common voxel spacing (see
    # preprocessing.resample_volume) so the same physical organ maps to the same
    # grid in every patient. Done on the raw HU CT (linear) and label GT (nearest)
    # *before* windowing; when target_spacing is None this is a no-op and the
    # original per-patient axial slicing is preserved.
    if target_spacing is not None:
        ct = resample_volume(ct, (dx, dy, dz), target_spacing, order=1)
        gt = resample_volume(gt, (dx, dy, dz), target_spacing, order=0)

    # Intensity normalization: fixed HU windowing (see preprocessing.py) by
    # default, or the legacy per-volume min-max when `window` is None (for A/B).
    norm_ct: np.ndarray
    if window is not None:
        level, width = window
        norm_ct = apply_hu_window(ct, level=level, width=width)
    else:
        norm_ct = norm_arr(ct)

    to_slice_ct = norm_ct
    to_slice_gt = gt
    # z may have changed after resampling; slice over the current depth.
    depth: int = to_slice_ct.shape[2]

    for idz in range(depth):
        if target_spacing is not None:
            # Resampling already set the mm/pixel; crop/pad (not resize) keeps it.
            img_slice = center_crop_pad(to_slice_ct[:, :, idz], shape, pad_value=0).astype(np.uint8)
            gt_slice = center_crop_pad(to_slice_gt[:, :, idz], shape, pad_value=0).astype(np.uint8)
        else:
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

    # Record the spacing of the *saved* slices: after resampling + crop/pad the
    # output voxels are exactly target_spacing (crop/pad does not rescale), which
    # is what downstream stitching / 3D metrics need. Otherwise report the source
    # spacing as before.
    return target_spacing if target_spacing is not None else (dx, dy, dz)


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


def resolve_slice_config(args: argparse.Namespace) -> tuple[Path, Path, tuple[int, int], int,
                                                            tuple[float, float] | None,
                                                            tuple[float, float, float] | None]:
    """Resolve the effective slicing options.

    Precedence: explicit CLI flag > experiment config (``--experiment``) > built-in
    default. This keeps every experiment reproducible from its name while still
    allowing ad-hoc one-off overrides, and never removes the manual code path.
    """
    sc: SliceConfig = get_experiment(args.experiment).slice if args.experiment else SliceConfig()

    source_dir = Path(args.source_dir) if args.source_dir else Path(sc.source_dir)
    if args.dest_dir:
        dest_dir = Path(args.dest_dir)
    elif args.experiment:
        dest_dir = Path(get_experiment(args.experiment).data_dir)
    else:
        raise SystemExit("provide --dest_dir, or --experiment to derive it")

    shape = tuple(args.shape) if args.shape else sc.shape
    retains = args.retains if args.retains is not None else sc.retains

    # Intensity normalization: None -> legacy min-max, else a fixed (level, width) HU window.
    window: tuple[float, float] | None
    if args.legacy_norm:
        window = None
    elif args.window_level is not None or args.window_width is not None:
        level = args.window_level if args.window_level is not None else MEDIASTINAL[0]
        width = args.window_width if args.window_width is not None else MEDIASTINAL[1]
        window = (level, width)
    else:
        window = sc.window

    # Spatial resampling: explicit --target_spacing > experiment config > None.
    target_spacing: tuple[float, float, float] | None
    if args.target_spacing is not None:
        target_spacing = tuple(args.target_spacing)
    else:
        target_spacing = sc.target_spacing

    return source_dir, dest_dir, shape, retains, window, target_spacing


def main(args: argparse.Namespace):
    src_path, dest_path, shape, retains, window, target_spacing = resolve_slice_config(args)

    if args.experiment:
        print(f">>> Experiment '{args.experiment}'")
    if window is None:
        print("Intensity normalization: legacy per-volume min-max")
    else:
        lo, hi = window[0] - window[1] / 2, window[0] + window[1] / 2
        print(f"Intensity normalization: HU window level={window[0]} width={window[1]} "
              f"-> clip [{lo:.0f}, {hi:.0f}] HU")
    if target_spacing is None:
        print("Spatial: raw axial slices, resize to shape (no resampling)")
    else:
        print(f"Spatial: resample to spacing {target_spacing} mm, then center crop/pad to {shape}")
    print(f"Source: {src_path}  ->  Dest: {dest_path}  (shape={shape}, retains={retains})")

    # Assume the clean up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    training_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    training_ids, validation_ids, test_ids = get_splits(src_path, retains, args.fold)

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    split_ids: list[str]
    for mode, split_ids in zip(["train", "val"], [training_ids, validation_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=shape,
                                 test_mode=mode == 'test',
                                 window=window,
                                 target_spacing=target_spacing)
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


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    # --experiment supplies defaults from experiments.py; any explicit flag below
    # overrides it. Source/dest fall back to the experiment when omitted.
    parser.add_argument('--experiment', default=None, choices=list(EXPERIMENTS),
                        help="Named config from experiments.py (sets source/dest/window/shape/retains).")
    parser.add_argument('--source_dir', type=str, default=None)
    parser.add_argument('--dest_dir', type=str, default=None)

    parser.add_argument('--shape', type=int, nargs="+", default=None)
    parser.add_argument('--retains', type=int, default=None,
                        help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="The number of cores to use for processing")

    # Intensity normalization (see preprocessing.py). Left unset (None) so the
    # value comes from the experiment config; when no experiment is given the
    # default is the clinical mediastinal / soft-tissue window.
    parser.add_argument('--window_level', type=float, default=None,
                        help="HU window center (level). Overrides the experiment window.")
    parser.add_argument('--window_width', type=float, default=None,
                        help="HU window width. Clip range is level +/- width/2.")
    parser.add_argument('--legacy_norm', action='store_true',
                        help="Force legacy per-volume min-max normalization (overrides windowing).")
    # Spatial resampling (see preprocessing.resample_volume). Left unset (None) so
    # the value comes from the experiment config; None means no resampling (raw
    # axial slices resized to --shape, the original behavior).
    parser.add_argument('--target_spacing', type=float, nargs=3, default=None,
                        metavar=('SX', 'SY', 'SZ'),
                        help="Resample every volume to this mm spacing, then center "
                             "crop/pad to --shape. Overrides the experiment spacing.")
    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
