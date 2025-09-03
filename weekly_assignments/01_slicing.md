# AI for medical imaging: individual assignment 1
## Data extraction, slicing and visualization
The goal of this assignment is to practice data extraction, slicing, and visualisation.
###Overview:
The purpose of this assignment is to unpack the dataset, split it into individual .png slices (for both the images and their labels), and then visualise the results using two different tools: the provided viewer and 3D Slicer. It also provides a chance to practice working with NIfTI files and 3D arrays in Python. Additionally, during this assignment, you should explore the dataset by examining data from multiple patients in order to better understand its structure and characteristics.

# 1. Environment setup and data download
First, set up your environment and download the required data by following the instructions from the project repository: https://github.com/HKervadec/ai4mi_project/tree/master 
Once your environment and data are ready, continue with the tasks below.

# 2. Create the slicing script (slice_segthor.py)
Fill the missing parts in this Python script called slice_segthor.py (see code below, can also be found in the GitHub repo), which will slice patients’ CT into individual .png slices (for both the images and their labels)

A library for handling medical imaging data is Nibabel: https://nipy.org/nibabel/Links 

During this process, you need to to:
- Implement a train/validation split at the patient level—ensuring that all slices from a single patient belong exclusively to either the training or validation set
- Normalize CT to greyscale from 0 to 255
- Adapt the greyscale of the segmentation masks (see the code below)
- Resize 2D slices to the following shape - 256 x 256 (useful function: resize from skimage.transform and its anti_aliasing parameter)

Your code needs to work when run:
```
python slice_segthor.py --source_dir "c/user/projects/ai4mi_project/data/segthor_train" --dest_dir "c/user/projects/ai4mi_project/data/SEGTHOR_tmp" --retains 5
```
# Deliverable:
Deliverable -zip file containing (please upload it to canvas):

- The executable slice_segthor.py script:
    - Code is running and files are correctly ordered and named (including train/val split; naming convention - see the comments in the code) (3 pt)
    - 2D CT images are rescaled and normalised properly (2 pt)
    - Sliced segmentation masks are correct (2 pt)
- A screenshot of the data visualised using the provided viewer (Segthor dataset, not the TOY dataset) (1 pt)
- A screenshot of the data visualised using 3D Slicer (Segthor dataset, not the TOY dataset) (1 pt)
- A .txt file describing any interesting observations you made about the data, or providing your description of the datasets (1 pt)
- 
Please use the following convention when naming your zip folder: student-nnn.zip (nnn being your number)

Contact
Maria Galanty (m.galanty@uva.nl) (lead), Caroline Magg (c.magg@uva.nl) (helper).

```
import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable

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


"""
TODO: Implement patient slicing.
Context:
  - Given an ID and paths, load the NIfTI CT volume and (if not test_mode) the GT volume.
  - Validate with sanity_ct / sanity_gt.
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

def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int], test_mode=False)\
        -> tuple[float, float, float]:

    id_path: Path = source_path / ("train" if not test_mode else "test") / id_
    ct_path: Path = (id_path / f"{id_}.nii.gz")
    assert id_path.exists()
    assert ct_path.exists()

    # --------- FILL FROM HERE -----------

    raise NotImplementedError("Implement slice_patient")


"""
TODO: Implement a simple train/val split.
Requirements:
  - List patient IDs from <src_path>/train (folder names).
  - Shuffle them (respect a seed set in main()).
  - Take the first `retains` as validation, and the rest as training.
  - Return (training_ids, validation_ids).
"""

def get_splits(src_path: Path, retains: int) -> tuple[list[str], list[str]]:
    # TODO: your code here

    raise NotImplementedError("Implement get_splits")

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

        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=tuple(args.shape))

        resolutions: list[tuple[float, float, float]]
        iterator = tqdm_(split_ids)
        resolutions = list(map(pfun, iterator))

        for key, val in zip(split_ids, resolutions):
            resolution_dict[key] = val

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")




def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Slicing parameters")

    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retains', type=int, default=10, help="Number of retained patient for the validation data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)
    print(args)

    return args

if __name__ == "__main__":
    main(get_args())
```
