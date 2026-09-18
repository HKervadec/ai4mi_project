from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.segmentation import watershed

MERGED_LABEL = 1
TRACHEA_LABEL = 3  
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
        aorta_c = int(np.argmin(sizes)) + 1
        markers = np.zeros(m.shape, dtype=np.int32)
        markers[lbl == aorta_c] = 1
        markers[(lbl != aorta_c) & eroded] = 2
        labels = watershed(np.zeros(m.shape), markers=markers, mask=m)
        return labels == 1, labels == 2
    return None, None


def reclaim_stray_esophagus_fragments(aorta: np.ndarray, esophagus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    lbl, n = ndi.label(esophagus, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return aorta, esophagus

    aorta_zs = np.where(aorta.any(axis=(0, 1)))[0]
    aorta_median_area = np.median([aorta[:, :, z].sum() for z in aorta_zs]) if len(aorta_zs) else 0
    area_cap = max(aorta_median_area * 2, 50)  

    sizes = ndi.sum(esophagus, lbl, range(1, n + 1))
    main_c = int(np.argmax(sizes)) + 1
    aorta_dilated = ndi.binary_dilation(aorta, iterations=2)
    for c in range(1, n + 1):
        if c == main_c:
            continue
        frag = lbl == c
        frag_zs = np.where(frag.any(axis=(0, 1)))[0]
        max_slice_area = max((frag[:, :, z].sum() for z in frag_zs), default=0)
        if max_slice_area > area_cap:
            continue
        if (aorta_dilated & frag).any():
            aorta = aorta | frag
            esophagus = esophagus & ~frag
    return aorta, esophagus


def split_aorta_from_esophagus(merged: np.ndarray, ct_affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aorta = np.zeros_like(merged)
    esophagus = np.zeros_like(merged)
    for z in range(merged.shape[2]):
        m = merged[:, :, z]
        if not m.any():
            continue
        lbl, n = ndi.label(m)
        if n >= 2:
            comps = list(range(1, n + 1))
            areas = [(lbl == c).sum() for c in comps]
            aorta_c = comps[int(np.argmin(areas))]
            aorta[:, :, z] = lbl == aorta_c
            esophagus[:, :, z] = (lbl != aorta_c) & m
        else:
            a, e = split_fused_slice(m)
            if a is None:
                esophagus[:, :, z] = m  
            else:
                aorta[:, :, z] = a
                esophagus[:, :, z] = e
    aorta, esophagus = reclaim_stray_esophagus_fragments(aorta, esophagus)
    return esophagus, aorta


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
    trachea = lab == TRACHEA_LABEL  
    print("\nBEFORE (label 1 = esophagus+aorta merged; label 3 = trachea, already separate):")
    report("label 1", merged)
    report("trachea", trachea)

    esophagus, aorta = split_aorta_from_esophagus(merged, ct_img.affine)
    assert ((esophagus | aorta) == merged).all(), "esophagus + aorta should exactly cover merged"
    assert not (esophagus & aorta).any(), "esophagus/aorta should not overlap"

    out = lab.copy() 
    out[merged] = 0
    out[esophagus] = 1
    out[aorta] = 4

    untouched = (lab != MERGED_LABEL)
    assert (out[untouched] == lab[untouched]).all(), "background/heart/trachea labels should be unchanged"

    print("\nAFTER (split):")
    report("esophagus", out == 1)
    report("trachea", out == 3)
    report("aorta", out == 4)


    touches_trachea = (ndi.binary_dilation(aorta, iterations=2) & trachea).any()
    print(f"  sanity check: aorta contacts the true trachea somewhere: {touches_trachea}")

    aorta_zs = np.where(aorta.any(axis=(0, 1)))[0]
    aorta_areas = np.array([aorta[:, :, z].sum() for z in aorta_zs])
    print(f"  aorta area: median={np.median(aorta_areas):.0f} max={aorta_areas.max()} "
          f"max/median={aorta_areas.max() / max(np.median(aorta_areas), 1):.1f}x")

    out_path = patient_dir / "GT_4label_v2.nii.gz"
    nib.save(nib.Nifti1Image(out, gt_img.affine, gt_img.header), out_path)
    print(f"\nwrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
