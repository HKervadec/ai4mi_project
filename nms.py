"""
Perform non-maximum suppression in 3D on predicted images
Since this is a post-processing technique, it's in a seperate file
"""

import argparse
import numpy as np
from skimage.measure import label
import nibabel as nib
import torch
from pathlib import Path
from utils.tensor_utils import tqdm_
import os

def main(args):
    # Select only files ending in .nii.gz
    target_volumes = os.listdir(args.src)
    target_volumes = [x for x in target_volumes if '.nii.gz' in x ]
    print('Found volumes:', target_volumes)

    for vol_path in tqdm_(target_volumes):
        result = nms(f'{args.src}/{vol_path}')

def nms(path):
    vol_nib = nib.load(path)
    vol = np.asarray(vol_nib.dataobj)

    # Split and label per class
    classes = [0,1,2,3,4]
    split = [np.where(vol == x, x, 0) for x in classes]
    split_labeled = [label(x) for x in split]

    # Keep only the largest area per class
    split_nms = [np.zeros(vol.shape)] * 5
    for i, l in enumerate(split_labeled):
        indices, counts = np.unique(l, return_counts=True)

        if len(counts) > 1:
            indices, counts = indices[1:], counts[1:]
        most_occuring = indices[np.argmax(counts)]
        selected = np.where(l == most_occuring, 1, 0)

        # Put the nms processed volume in its right spot
        # and make sure the values match the class index
        split_nms[i] = selected * i

    result = np.vstack(split_nms)
    print(result.shape)
    print(np.hstack(split_nms.shape), np.stack(split_nms.shape))
    quit()
    # split_nibs = [nib.nifti1.Nifti1Image(x, affine=vol_nib.affine, header=vol_nib.header) for x in gt_split_labeled]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src',
        type=Path,
        help='Path to the folder containing stitched volumes to perform NMS on')
    parser.add_argument(
        '--dest',
        type=Path,
        help='Path to folder to save processed volumes in.'
    )
    args = parser.parse_args()

    if args.dest is None:
        args.dest = Path(f"{args.src}/nms")

    args.dest.mkdir(parents=True, exist_ok=True)

    main(args)
