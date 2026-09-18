from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.segmentation import watershed

MERGED_LABEL = 1
MAX_EROSION = 8


def ap_axis_and_sign(affine: np.ndarray) -> tuple[int, int]:
    codes = nib.aff2axcodes(affine)
    for axis, code in enumerate(codes[:2]):
        if code in ("A", "P"):
            return axis, (1 if code == "A" else -1)
    raise ValueError(f"expected an anterior/posterior in-plane axis, got orientation {codes}")


def split_fused_slice(m: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:

    for k in range(1, MAX_EROSION + 1):
        eroded = ndi.binary_erosion(m, iterations=k)
        lbl, n = ndi.label(eroded)
        if n < 2:
            continue
        sizes = [(lbl == c).sum() for c in range(1, n + 1)]
        trachea_c = int(np.argmin(sizes)) + 1
        markers = np.zeros(m.shape, dtype=np.int32)
        markers[lbl == trachea_c] = 1
        markers[(lbl != trachea_c) & eroded] = 2
        labels = watershed(np.zeros(m.shape), markers=markers, mask=m)
        return labels == 1, labels == 2
    return None, None


def split_trachea_from_esophagus_v2(merged: np.ndarray, ct_affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trachea = np.zeros_like(merged)
    esophagus = np.zeros_like(merged)
    for z in range(merged.shape[2]):
        m = merged[:, :, z]
        if not m.any():
            continue
        lbl, n = ndi.label(m)
        if n >= 2:
            comps = list(range(1, n + 1))
            areas = [(lbl == c).sum() for c in comps]
            trachea_c = comps[int(np.argmin(areas))]
            trachea[:, :, z] = lbl == trachea_c
            esophagus[:, :, z] = (lbl != trachea_c) & m
        else:
            t, e = split_fused_slice(m)
            if t is None:
                esophagus[:, :, z] = m  # never separated -- overwhelmingly esophagus alone
            else:
                trachea[:, :, z] = t
                esophagus[:, :, z] = e
    return esophagus, trachea


def report(name: str, mask: np.ndarray) -> None:
    if not mask.any():
        print(f"  {name:<10} not present")
        return
    zs = np.where(mask.any(axis=(0, 1)))[0]
    print(f"  {name:<10} {len(zs):>4} slices   {int(mask.sum()):>7} px")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("segthor_part1/data/segthor_part1/train"))
    ap.add_argument("--patient", default="Patient_01")
    args = ap.parse_args()

    patient_dir = args.data_dir / args.patient
    gt_img = nib.load(patient_dir / "GT.nii.gz")
    ct_img = nib.load(patient_dir / f"{args.patient}.nii.gz")
    lab = np.asanyarray(gt_img.dataobj).astype(np.uint8)

    print(f"{args.patient}  shape={lab.shape}")
    print(f"  GT affine is identity (no orientation info): {(gt_img.affine == np.eye(4)).all()}")
    print(f"  CT orientation={nib.aff2axcodes(ct_img.affine)}  spacing={ct_img.header.get_zooms()[:3]}")

    merged = lab == MERGED_LABEL
    print("\nBEFORE (label 1 = esophagus+trachea merged):")
    report("label 1", merged)

    esophagus, trachea = split_trachea_from_esophagus_v2(merged, ct_img.affine)
    assert ((esophagus | trachea) == merged).all(), "esophagus + trachea should exactly cover merged"
    assert not (esophagus & trachea).any(), "esophagus/trachea should not overlap"

    out = lab.copy()
    out[lab == 3] = 4
    out[merged] = 0
    out[esophagus] = 1
    out[trachea] = 3

    untouched = (lab != MERGED_LABEL) & (lab != 3)
    assert (out[untouched] == lab[untouched]).all(), "background/heart labels should be unchanged"

    print("\nAFTER (split):")
    report("esophagus", out == 1)
    report("trachea", out == 3)
    report("aorta", out == 4)

    ap_axis, ap_sign = ap_axis_and_sign(ct_img.affine)
    zs = np.where(merged.any(axis=(0, 1)))[0]
    consistent = checked = 0
    for z in zs:
        t, e = trachea[:, :, z], esophagus[:, :, z]
        if not (t.any() and e.any()):
            continue
        checked += 1
        consistent += np.mean(np.where(t)[ap_axis]) * ap_sign > np.mean(np.where(e)[ap_axis]) * ap_sign
    print(f"  anatomy check: trachea anterior to esophagus in {consistent}/{checked} slices with both present")

    tra_zs = np.where(trachea.any(axis=(0, 1)))[0]
    tra_areas = np.array([trachea[:, :, z].sum() for z in tra_zs])
    print(f"  trachea area: median={np.median(tra_areas):.0f} max={tra_areas.max()} "
          f"max/median={tra_areas.max() / max(np.median(tra_areas), 1):.1f}x")

    out_path = patient_dir / "GT_4label_v2.nii.gz"
    nib.save(nib.Nifti1Image(out, gt_img.affine, gt_img.header), out_path)
    print(f"\nwrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())