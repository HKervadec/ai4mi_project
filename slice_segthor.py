import argparse
import os
import pickle
import random
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
from PIL import Image
from skimage.transform import resize

from utils import map_, tqdm_


def norm_arr(img: np.ndarray) -> np.ndarray:
    """
    CT images have a wide range of intensity values (Hounsfield units)
    Goal: normalize an image array to the range [0, 255]  and return it as a dtype=uint8
    Which is compatible with standard image formats (PNG)

    Args:
        img: numpy array of CT in Hounsfield units

    Returns:
        uint8 numpy array with same shape
    """
    hu_lower, hu_upper = -1000.0, 400.0

    clipped = np.clip(img, hu_lower, hu_upper)
    scaled = (clipped - hu_lower) / (hu_upper - hu_lower) * 255.0
    return np.rint(scaled).astype(np.uint8)


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
    assert set(np.unique(gt)) == set(range(5))

    return True


def slice_patient(
    id_: str,
    dest_path: Path,
    source_path: Path,
    shape: tuple[int, int],
    test_mode=False,
) -> tuple[float, float, float]:
    """
    Patient slicing.
    Context:
      - [x] Given an ID and paths, load the NIfTI CT volume and (if not test_mode) the GT volume.
      - [x] Validate with sanity_ct / sanity_gt.
      - [x] Normalise CT with norm_arr().
      - [x] Slice the 3D volumes into 2D slices, resize to `shape`, and save PNGs.
      - [x] Currently we have groundtruth masks marked as {0,1,2,3,4} but those values are hard to distinguish in a grayscale png.
        Multiplying by 63 maps them to {0,63,126,189,252}, which keeps labels visually distinct in a grayscale PNG.
        You can use the following code, which works for already sliced 2d images:
        gt_slice *= 63
        assert gt_slice.dtype == np.uint8, gt_slice.dtype
        assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)
      - [x] Return the original voxel spacings (dx, dy, dz).

    Hints:
      - [x] Use nibabel to load NIfTI images.
      - [x] Use skimage.transform.resize (tip: anti_aliasing might be useful)
      - [x] The PNG files should be stored in the dest_path, organised into separate subfolders: train/img, train/gt, val/img, and val/gt
      - [x] Use consistent filenames: e.g. f"{id_}_{idz:04d}.png" inside subfolders "img" and "gt"; where idz is the slice index.
    """

    id_path: Path = source_path / ("train" if not test_mode else "test") / id_
    ct_path: Path = id_path / f"{id_}.nii.gz"
    assert id_path.exists()
    assert ct_path.exists()

    # load CT nifti
    ct_nib = nib.nifti1.load(str(ct_path))
    x, y, z = ct_nib.shape
    dx, dy, dz = ct_nib.header.get_zooms()[:3]
    ct_arr = np.array(ct_nib.dataobj)

    # sanity check on CT
    sanity_ct(ct_arr, x, y, z, dx, dy, dz)

    # normalize CT -> uint8 [0,255]
    normal_ct_img = norm_arr(ct_arr)

    # prepare destination directories
    img_dir = dest_path / "img"
    gt_dir = dest_path / "gt"
    img_dir.mkdir(parents=True, exist_ok=True)
    if not test_mode:
        gt_dir.mkdir(parents=True, exist_ok=True)

    # slice, resize and save CT slices
    # loop over z dimension preserving slice index
    for iz in range(normal_ct_img.shape[2]):
        ct_slice = normal_ct_img[:, :, iz]

        resized_ct = resize(
            ct_slice,
            output_shape=tuple(shape),
            preserve_range=True,
            anti_aliasing=True,
        )

        resized_ct_u8 = np.rint(resized_ct).astype(np.uint8)

        out_name = img_dir / f"{id_}_{iz:04d}.png"
        Image.fromarray(resized_ct_u8.T).convert("L").save(out_name)

    if not test_mode:
        # load GT nifti
        gt_path: Path = id_path / "GT.nii.gz"
        assert gt_path.exists()
        gt_nib = nib.nifti1.load(str(gt_path))
        gt_arr = np.array(gt_nib.dataobj)

        # sanity check on GT
        sanity_gt(gt_arr, ct_arr)

        # slice, resize labels with nearest neighbor and convert to {0,63,126,189,252}
        for iz in range(gt_arr.shape[2]):
            gt_slice = gt_arr[:, :, iz]

            # resize labels: preserve_range
            resized_gt = resize(
                gt_slice,
                output_shape=tuple(shape),
                preserve_range=True,
                anti_aliasing=False,
            )

            resized_gt = np.rint(resized_gt).astype(np.uint8)

            # safety: clamp labels to 0..4
            resized_gt = np.clip(resized_gt, 0, 4)

            # multiply by 63 to space them out visually
            resized_gt = (resized_gt * 63).astype(np.uint8)
            assert set(np.unique(resized_gt)) <= set([0, 63, 126, 189, 252]), np.unique(
                resized_gt
            )

            out_name = gt_dir / f"{id_}_{iz:04d}.png"
            Image.fromarray(resized_gt.T).convert("L").save(out_name)

    return dx, dy, dz


def get_splits(src_path: Path, retains: int) -> tuple[list[str], list[str]]:
    """
    Simple train/val split.

    Requirements:
      - List patient IDs from <src_path>/train (folder names).
      - Shuffle them (respect a seed set in main()).
      - Take the first `retains` as validation, and the rest as training.
      - Return (training_ids, validation_ids).


    Args:
        src_path (Path): Path to raw data.
        retains (int): Number of datapoints to be used for val. Rest for train.

    Returns:
        (train_ids, val_ids)
    """
    all_ids: list[str] = []

    train_dir = src_path / "train"
    file_names = train_dir.glob("*")
    for file_name in file_names:
        all_ids.append(file_name.name)

    random.shuffle(all_ids)

    val_ids = all_ids[:retains]
    train_ids = all_ids[retains:]

    print(
        f"\nSplit {len(all_ids)} datapoints from '{train_dir}' into {len(train_ids)}"
        f" datapoints for training and {len(val_ids)} datapoints for validation.\n"
    )

    return train_ids, val_ids


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)
    if not dest_path.exists():
        dest_path.mkdir(parents=True, exist_ok=True)

    assert src_path.exists()
    assert dest_path.exists()

    training_ids: list[str]
    validation_ids: list[str]
    training_ids, validation_ids = get_splits(src_path, args.retains)

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    for mode, split_ids in zip(["train", "val"], [training_ids, validation_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(
            slice_patient,
            dest_path=dest_mode,
            source_path=src_path,
            shape=tuple(args.shape),
        )

        resolutions: list[tuple[float, float, float]]
        iterator = tqdm_(split_ids)
        resolutions = list(map(pfun, iterator))

        for key, val in zip(split_ids, resolutions):
            resolution_dict[key] = val

    with open(dest_path / "spacing.pkl", "wb") as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slicing parameters")

    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--dest_dir", type=str, required=True)
    parser.add_argument("--shape", type=int, nargs="+", default=[256, 256])
    parser.add_argument(
        "--retains",
        type=int,
        default=10,
        help="Number of retained patient for the validation data",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)
    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
