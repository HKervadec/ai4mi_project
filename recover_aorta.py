#!/usr/bin/env python3
"""Simple script to split the merged class 1 back into esophagus (1) and aorta (4). 
Uses a size heuristic to set seeds, then watershed on the 3D image.

The aorta is the thicker structure, so the deepest seed should be its
core. The threshold can be raised if the aorta share comes out near 100%:
that would mean the seed bridged the two organs.

Patient_03 needs an extra intensity-based repair, see repair_patient_03.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed

ROOT = Path("data/...") #path to the original folder with the wrong data
OUT = "GT_recovered.nii.gz"
THRESHOLD = 4.0  # seed depth in mm
OVERRIDES = {"Patient_15": 5.0}  # patients needing a deeper seed
RIM = 2  # in-plane erosion in voxels before placing markers (for patient 3)
MARGIN = 0.25  # fraction of the HU gap around each organ's median (for patient 3)
S26 = np.ones((3, 3, 3), dtype=bool)
IN_PLANE = np.zeros((3, 3, 3), dtype=bool)  # 4-connected cross in the axial plane
IN_PLANE[:, :, 1] = ndi.generate_binary_structure(2, 1)
PATIENTS = [
    "Patient_01", "Patient_02", "Patient_03", "Patient_04", "Patient_05", "Patient_06",
    "Patient_07", "Patient_08", "Patient_09", "Patient_10", "Patient_11", "Patient_12",
    "Patient_13", "Patient_14", "Patient_15", "Patient_16", "Patient_17", "Patient_18",
    "Patient_19", "Patient_20"
]


def repair_patient_03(gt, merged, seeds, aorta_seed, ct):
    """A stretch of Patient_03's esophagus gets no seed and is flooded by the aorta.
    The scan has contrast, so intensity markers and a watershed on the CT gradient
    replace the depth-based split.
    """
    ct = ct.astype(np.float32)
    smooth = ndi.gaussian_filter(ct, sigma=(1, 1, 0))  # in-plane only, slices are 2.5mm apart
    aorta_hu = np.median(ct[seeds == aorta_seed])
    eso_hu = np.median(ct[gt == 1])
    gap = aorta_hu - eso_hu

    # skip the mask border, where partial volume makes the aorta wall look dark
    inner = ndi.binary_erosion(merged, structure=IN_PLANE, iterations=RIM)
    eso = inner & (smooth < eso_hu + MARGIN * gap)
    eso |= (seeds > 0) & (seeds != aorta_seed) & (smooth < eso_hu + gap / 2)
    aorta = inner & (smooth > aorta_hu - MARGIN * gap) & ~eso

    markers = np.zeros(merged.shape, dtype=np.int32)
    markers[eso] = 1
    markers[aorta] = 4
    gradient = np.stack([sobel(smooth[:, :, z]) for z in range(smooth.shape[2])], axis=2)
    aorta = watershed(gradient, markers, mask=merged) == 4

    # keep only the main aorta, cut-off bright spots go back to the esophagus
    pieces, _ = ndi.label(aorta, structure=S26)
    gt[merged] = 1
    gt[pieces == np.bincount(pieces.ravel())[1:].argmax() + 1] = 4


for name in PATIENTS:
    path = ROOT / name / "GT.nii.gz"
    im = nib.load(path)
    gt = np.asanyarray(im.dataobj).copy()
    merged = gt == 1

    dist = ndi.distance_transform_edt(merged, sampling=im.header.get_zooms()[:3])
    seeds, _ = ndi.label(dist > OVERRIDES.get(name, THRESHOLD))
    basins = watershed(-dist, seeds, mask=merged)
    aorta_seed = np.bincount(seeds.ravel())[1:].argmax() + 1
    gt[basins == aorta_seed] = 4

    if name == "Patient_03":
        ct = np.asanyarray(nib.load(ROOT / name / f"{name}.nii.gz").dataobj)
        repair_patient_03(gt, merged, seeds, aorta_seed, ct)

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
