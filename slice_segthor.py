import argparse
import pickle
import random
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np

from utils import map_, tqdm_

"""
TODO: Implement image normalisation.
CT images have a wide range of intensity values (Hounsfield units)
Goal: normalize an image array to the range [0, 255]  and return it as a dtype=uint8
Which is compatible with standard image formats (PNG)
"""


def norm_arr(img: np.ndarray) -> np.ndarray:
    # TODO: your code here

    raise NotImplementedError("Implement norm_arr")


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
      - Normalise CT with norm_arr().
      - Slice the 3D volumes into 2D slices, resize to `shape`, and save PNGs.
      - Currently we have groundtruth masks marked as {0,1,2,3,4} but those values are hard to distinguish in a grayscale png.
        Multiplying by 63 maps them to {0,63,126,189,252}, which keeps labels visually distinct in a grayscale PNG.
        You can use the following code, which works for already sliced 2d images:
        gt_slice *= 63
        assert gt_slice.dtype == np.uint8, gt_slice.dtype
        assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)
      - Return the original voxel spacings (dx, dy, dz).

    Hints:
      - Use nibabel to load NIfTI images.
      - Use skimage.transform.resize (tip: anti_aliasing might be useful)
      - The PNG files should be stored in the dest_path, organised into separate subfolders: train/img, train/gt, val/img, and val/gt
      - Use consistent filenames: e.g. f"{id_}_{idz:04d}.png" inside subfolders "img" and "gt"; where idz is the slice index.
    """

    id_path: Path = source_path / ("train" if not test_mode else "test") / id_
    ct_path: Path = id_path / f"{id_}.nii.gz"
    assert id_path.exists()
    assert ct_path.exists()

    # --------- FILL FROM HERE -----------
    ct_img = nib.nifti1.load(ct_path)
    x, y, z = ct_img.shape
    print(ct_img.header.get_zooms())
    dx, dy, dz = ct_img.header.get_zooms()[:3]
    sanity_ct(ct_img, x, y, z, dx, dy, dz)

    if not test_mode:
        gt_path: Path = id_path / "GT.nii.gz"
        gt_img = nib.nifti1.load(gt_path)
        sanity_gt(gt_img, ct_img)

    raise NotImplementedError("Implement slice_patient")


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
