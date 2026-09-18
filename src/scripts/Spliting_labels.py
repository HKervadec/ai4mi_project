from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.segmentation import watershed

MERGED_LABEL = 1
STRUCT_3D = np.ones((3, 3, 3), dtype=bool)


def ap_axis_and_sign(affine: np.ndarray) -> tuple[int, int]:
    codes = nib.aff2axcodes(affine)
    for axis, code in enumerate(codes[:2]):
        if code in ("A", "P"):
            return axis, (1 if code == "A" else -1)
    raise ValueError(f"expected an anterior/posterior in-plane axis, got orientation {codes}")


def classify_separated_slices(merged: np.ndarray, ap_axis: int, ap_sign: int) -> tuple[np.ndarray, np.ndarray]:

    trachea_marker = np.zeros_like(merged)
    esophagus_marker = np.zeros_like(merged)
    prev_centroid = None
    prev_area = None
    for z in range(merged.shape[2]):
        m = merged[:, :, z]
        if not m.any():
            continue
        lbl, n = ndi.label(m)
        if n < 2:
            continue
        comps = list(range(1, n + 1))
        centroids = [np.array(ndi.center_of_mass(lbl == c)) for c in comps]
        areas = [float((lbl == c).sum()) for c in comps]

        if prev_centroid is None:
            # first confident slice: no history yet, trust anatomy alone
            ap_pos = [centroids[i][ap_axis] * ap_sign for i in range(n)]
            trachea_i = int(np.argmax(ap_pos))
        else:
            dist = [np.linalg.norm(centroids[i] - prev_centroid) for i in range(n)]
            area_ratio = [max(areas[i], prev_area) / max(min(areas[i], prev_area), 1.0) for i in range(n)]
            cost = [dist[i] + 20.0 * (area_ratio[i] - 1.0) for i in range(n)]
            trachea_i = int(np.argmin(cost))

        trachea_c = comps[trachea_i]
        trachea_marker[:, :, z] = lbl == trachea_c
        esophagus_marker[:, :, z] = (lbl != trachea_c) & m
        prev_centroid = centroids[trachea_i]
        prev_area = areas[trachea_i]
    return trachea_marker, esophagus_marker


def split_v2(merged: np.ndarray, ct_affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ap_axis, ap_sign = ap_axis_and_sign(ct_affine)
    spacing = np.abs(np.diag(ct_affine))[:3]  

    trachea_marker, esophagus_marker = classify_separated_slices(merged, ap_axis, ap_sign)

    markers = np.zeros(merged.shape, dtype=np.int32)
    markers[trachea_marker] = 1
    markers[esophagus_marker] = 2

    dist = ndi.distance_transform_edt(merged, sampling=spacing)
    labels3d = watershed(-dist, markers=markers, mask=merged)


    lbl3d, n3d = ndi.label(merged, structure=STRUCT_3D)
    for c in range(1, n3d + 1):
        comp = lbl3d == c
        if (labels3d[comp] != 0).any():
            continue  # already resolved by the watershed
        ap_pos = np.where(comp)[ap_axis] * ap_sign
        labels3d[comp] = np.where(ap_pos > np.median(ap_pos), 1, 2)

    return labels3d == 2, labels3d == 1  # esophagus, trachea


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

    esophagus, t = split_v2(merged, ct_img.affine)
    assert ((esophagus | t) == merged).all(), "esophagus + trachea should exactly cover merged"
    assert not (esophagus & t).any(), "esophagus/trachea should not overlap"

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
   

    out_path = patient_dir / "GT_4label_v2.nii.gz"
    nib.save(nib.Nifti1Image(out, gt_img.affine, gt_img.header), out_path)
    print(f"\nwrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())