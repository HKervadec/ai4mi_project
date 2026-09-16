#!/usr/bin/env python3
"""Simple script to split the merged class 1 back into esophagus (1) and aorta (4). 
Uses a size heuristic to set seeds, then watershed on the 3D image.

The aorta is the thicker structure, so the deepest seed should be its
core. The threshold can be raised if the aorta share comes out near 100%:
that would mean the seed bridged the two organs.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

ROOT = Path("data/segthor_part1/train")
OUT = "GT_recovered.nii.gz"
THRESHOLD = 4.0  # seed depth in mm
OVERRIDES = {"Patient_15": 5.0}  # patients needing a deeper seed
PATIENTS = [
    "Patient_01", "Patient_02", "Patient_03", "Patient_04", "Patient_05", "Patient_06",
    "Patient_07", "Patient_08", "Patient_09", "Patient_10", "Patient_11", "Patient_12", 
    "Patient_13", "Patient_14", "Patient_15", "Patient_16", "Patient_17", "Patient_18", 
    "Patient_19", "Patient_20"
]

for name in PATIENTS:
    path = ROOT / name / "GT.nii.gz"
    im = nib.load(path)
    gt = np.asanyarray(im.dataobj).copy()
    merged = gt == 1

    dist = ndi.distance_transform_edt(merged, sampling=im.header.get_zooms()[:3])
    seeds, _ = ndi.label(dist > OVERRIDES.get(name, THRESHOLD))
    basins = watershed(-dist, seeds, mask=merged)
    gt[basins == np.bincount(seeds.ravel())[1:].argmax() + 1] = 4

    nib.save(nib.Nifti1Image(gt, im.affine, im.header), path.parent / OUT)
    aorta = (gt == 4).sum()
    print(f"{name}: aorta {aorta} ({100 * aorta / merged.sum():.0f}%), esophagus {(gt == 1).sum()}")

    # Patient 7 has the correct segmentation, so we can score against it.
    if name == "Patient_07":
        reference = path.parent / "GT2.nii.gz"
        if reference.exists():
            ref = np.asanyarray(nib.load(reference).dataobj)
            wrong = gt[merged] != ref[merged]
            eso = ((ref == 1) & (gt == 4)).sum() / (ref == 1).sum()
            aor = ((ref == 4) & (gt == 1)).sum() / (ref == 4).sum()
            print(f"  vs GT: {wrong.sum()}/{merged.sum()} wrong ({100 * wrong.mean():.3f}%)"
                f" | mislabeled as: esophagus {100 * eso:.3f}%, aorta {100 * aor:.3f}%")
