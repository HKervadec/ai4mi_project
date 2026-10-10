"""Round-trip check for FoV normalisation: GT -> forward -> 256 -> back -> compare."""

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from skimage.transform import resize

from pixel_space_norm import fov_target_size, center_crop_or_pad

R = dict(mode="constant", preserve_range=True, anti_aliasing=False, order=0)


def roundtrip(gt, dx, target_fov):
    X, Y, Z = gt.shape
    t = fov_target_size(dx, target_fov) if target_fov else X
    fwd = center_crop_or_pad(gt, (t, t), 0)
    lost = [(gt == k).sum() - (fwd == k).sum() for k in range(1, 5)]
    rec = np.zeros_like(gt)
    for z in range(Z):
        small = resize(fwd[:, :, z], (256, 256), **R)
        back = resize(small, (t, t), **R)
        rec[:, :, z] = center_crop_or_pad(back, (X, Y), 0)
    return rec, lost


def dice(a, b, k):
    a, b = a == k, b == k
    return 2 * (a & b).sum() / (a.sum() + b.sum())


root = Path(sys.argv[1])
for pid in sys.argv[2:]:
    nib_obj = nib.load(root / "train" / pid / "GT.nii.gz")
    gt = np.asarray(nib_obj.dataobj)mat
    dx = nib_obj.header.get_zooms()[0]
    base, _ = roundtrip(gt, dx, None)
    fov, lost = roundtrip(gt, dx, 500)
    print(f"{pid}  dx={dx:.3f}  FoV={dx * gt.shape[0]:.0f}mm  organ voxels lost by crop={lost}")
    for k, name in enumerate(["esophagus", "heart", "trachea", "aorta"], start=1):
        print(f"   {name:10s} baseline={dice(gt, base, k):.4f}  fov={dice(gt, fov, k):.4f}")