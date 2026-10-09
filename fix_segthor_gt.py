#!/usr/bin/env python3

# Repairs the provided SegTHOR ground truths before slicing:
#  * label 1 is esophagus and aorta merged together; split it back into
#    1 (esophagus) and 4 (aorta), the original SegTHOR convention
#  * GT.nii.gz carries an identity affine; copy the affine/header of the CT
# When a patient has a GT2.nii.gz (already split) it is used as-is, and also
# serves as a check of the automatic split.
#
# Method (selected by comparing against Patient_07/GT2.nii.gz):
#  1. distance map (in mm) inside the merged mask
#  2. aorta seed: largest 3D component of voxels deeper than `radius` mm; the
#     aorta is a much thicker tube (max radius 13-20mm) than the esophagus (6-10mm)
#  3. esophagus seed: largest component of the mask farther than radius+1+margin mm
#     from the aorta seed
#  4. seeded watershed on the inverted distance map, so the split follows the
#     narrow "neck" where the two tubes touch
#  5. per axial slice, tiny fragments (< crumb_mm2) left on the wrong side of the
#     interface are handed to the neighbouring organ
# Patient_07: 87/115229 voxels wrong (Dice 0.998 esophagus / 0.9995 aorta), all on
# the interface; 80-120 wrong when resampled to the other spacings of the dataset.
# CT intensities do not help: the scans are mostly non-contrast.

import shutil
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.segmentation import watershed

S26 = np.ones((3, 3, 3), dtype=bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask, structure=S26)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (np.argmax(sizes) + 1)


def reassign_crumbs(eso: np.ndarray, aorta: np.ndarray, spacing: np.ndarray, max_mm2: float) -> None:
    # In place. Only fragments that are not the organ's main blob in that slice and that
    # touch the other organ move, so the ascending/descending aorta pair is never affected.
    px_area = spacing[0] * spacing[1]
    for z in range(eso.shape[2]):
        for src, dst in [(eso[:, :, z], aorta[:, :, z]), (aorta[:, :, z], eso[:, :, z])]:
            lab, n = ndi.label(src)
            if n < 2:
                continue
            sizes = ndi.sum(src, lab, range(1, n + 1))
            main = np.argmax(sizes) + 1
            touching = ndi.binary_dilation(dst)
            for k in range(1, n + 1):
                comp = lab == k
                if k != main and sizes[k - 1] * px_area <= max_mm2 and (comp & touching).any():
                    src[comp] = False
                    dst[comp] = True


def split_esophagus_aorta(merged: np.ndarray, spacing: np.ndarray, radius: float = 6.0,
                          margin: float = 3.0, crumb_mm2: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    # Work on a padded bounding box: much faster than on the full volume
    bbox = ndi.find_objects(merged.astype(np.uint8))[0]
    bbox = tuple(slice(max(s.start - 12, 0), s.stop + 12) for s in bbox)
    m = merged[bbox]

    dist = ndi.distance_transform_edt(m, sampling=spacing)
    aorta_seed = largest_component(dist > radius)
    from_aorta = ndi.distance_transform_edt(~aorta_seed, sampling=spacing)
    eso_seed = largest_component(m & (from_aorta > radius + 1 + margin))

    markers = np.zeros(m.shape, dtype=np.int32)
    markers[eso_seed] = 1
    markers[aorta_seed] = 2
    lab = watershed(-dist, markers, mask=m)
    eso, aorta = lab == 1, lab == 2
    reassign_crumbs(eso, aorta, spacing, crumb_mm2)

    eso_full = np.zeros_like(merged)
    aorta_full = np.zeros_like(merged)
    eso_full[bbox] = eso
    aorta_full[bbox] = aorta
    return eso_full, aorta_full


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def fix_patient(src: Path, dest: Path, args: argparse.Namespace) -> None:
    ct_path = src / f"{src.name}.nii.gz"
    ct_nib = nib.load(str(ct_path))
    spacing = np.array(ct_nib.header.get_zooms()[:3], dtype=float)

    gt = np.asarray(nib.load(str(src / "GT.nii.gz")).dataobj).astype(np.uint8)
    assert set(np.unique(gt)) <= {0, 1, 2, 3}, np.unique(gt)
    eso, aorta = split_esophagus_aorta(gt == 1, spacing, args.radius, args.margin, args.crumb_mm2)
    assert not (eso & aorta).any() and np.array_equal(eso | aorta, gt == 1)

    top = lambda x: np.where(x.any(axis=(0, 1)))[0].max()
    if top(eso) <= top(aorta):
        print(f"WARNING {src.name}: esophagus does not extend above the aorta, check the split visually")

    if (src / "GT2.nii.gz").exists():
        gt2 = np.asarray(nib.load(str(src / "GT2.nii.gz")).dataobj).astype(np.uint8)
        assert np.array_equal(np.where(gt2 == 4, 1, gt2), gt), "GT2 is not a split of GT"
        msg = f"used GT2 (automatic split: Dice esophagus {dice(eso, gt2 == 1):.4f}, aorta {dice(aorta, gt2 == 4):.4f})"
        gt = gt2
    else:
        gt[eso] = 1
        gt[aorta] = 4
        msg = f"split: esophagus {eso.sum()} vox, aorta {aorta.sum()} vox"

    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(ct_path, dest / ct_path.name)
        header = ct_nib.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(gt, ct_nib.affine, header), str(dest / "GT.nii.gz"))
    print(f"{src.name}: {msg}")


def main(args: argparse.Namespace) -> None:
    src_path = Path(args.source_dir)
    dest_path = Path(args.dest_dir)
    assert src_path.exists()
    assert args.dry_run or not dest_path.exists()

    for patient in sorted(p for p in (src_path / "train").iterdir() if p.is_dir()):
        fix_patient(patient, dest_path / "train" / patient.name, args)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split merged esophagus/aorta labels and fix GT affines")
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--radius', type=float, default=6.0,
                        help="Depth (mm) inside the merged mask that only the aorta reaches")
    parser.add_argument('--margin', type=float, default=3.0,
                        help="Extra distance (mm) from the aorta seed for the esophagus seed")
    parser.add_argument('--crumb_mm2', type=float, default=10.0,
                        help="Max area of stray per-slice fragments reassigned across the interface")
    parser.add_argument('--dry_run', action='store_true', help="Only report, write nothing")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
