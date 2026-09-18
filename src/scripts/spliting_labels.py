"""Splitting the merged label.

Usage:

  For a single patient (datapoint):
    python spliting_labels.py --patient Patient_01
  For all patients in a folder:
    python spliting_labels.py --all --data-dir segthor_part1/data/segthor_part1/train
  Running just python spliting_labels.py will default to Patient_01 in the default data-dir. (if it is set as such)

Writes GT_4label_v2.nii.gz into each patient folder.

if you want to call this in another python script

from spliting_labels import process_patient

  stats = process_patient(Path("Data_path"),"Patient_01", verbose=False)
  print(stats["touches_trachea"], stats["aorta_max_median_ratio"]) # You can do this as well

or loop over every patient:

  root = Path("segthor_part1/data/segthor_part1/train")
  results = [process_patient(p, p.name, verbose=False) for p in sorted(root.glob("Patient_*"))]
"""
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

# helper
def ap_axis_and_sign(affine: np.ndarray) -> tuple[int, int]:
    codes = nib.aff2axcodes(affine)
    for axis, code in enumerate(codes[:2]):
        if code in ("A", "P"):
            return axis, (1 if code == "A" else -1)
    raise ValueError(f"expected an anterior/posterior in-plane axis, got orientation {codes}")

# changed to erosion in case of connected components, then watershed to split them
def split_fused_slice(m: np.ndarray, prev_centroid: np.ndarray | None = None, size_cap: float | None = None, aorta_hint: float | None = None) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:

    best, best_score = None, None
    for k in range(1, MAX_EROSION + 1):
        eroded = ndi.binary_erosion(m, iterations=k)
        lbl, n = ndi.label(eroded)
        if n < 2:
            continue
        if prev_centroid is None:
            sizes = [(lbl == c).sum() for c in range(1, n + 1)]
            aorta_c = int(np.argmin(sizes)) + 1
        else:
            centroids = [np.array(ndi.center_of_mass(lbl == c)) for c in range(1, n + 1)]
            aorta_c = min(range(n), key=lambda i: np.linalg.norm(centroids[i] - prev_centroid)) + 1
        markers = np.zeros(m.shape, dtype=np.int32)
        markers[lbl == aorta_c] = 1
        markers[(lbl != aorta_c) & eroded] = 2
        labels = watershed(np.zeros(m.shape), markers=markers, mask=m)
        a, e = labels == 1, labels == 2
        if size_cap is not None and a.sum() > size_cap:
            continue
        if aorta_hint is None:
            return a, e
        score = abs(int(a.sum()) - aorta_hint)
        if best_score is None or score < best_score:
            best, best_score = (a, e), score
    return best if best is not None else (None, None)

# post hoc 3D clean up
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

# spliting the merged label into aorta and esophagus
def split_aorta_from_esophagus(merged: np.ndarray, ct_affine: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    aorta = np.zeros_like(merged)
    esophagus = np.zeros_like(merged)
    prev_centroid = None
    trusted_areas: list[int] = []
    flagged_zs: list[int] = []
    pending: list[tuple[int, np.ndarray]] = []

    for z in range(merged.shape[2]):
        m = merged[:, :, z]
        if not m.any():
            continue
        lbl, n = ndi.label(m)
        confident = False

        if n >= 2:
            comps = list(range(1, n + 1))
            centroids = [np.array(ndi.center_of_mass(lbl == c)) for c in comps]
            areas = [(lbl == c).sum() for c in comps]

            if prev_centroid is None:
                if pending:
                    last_c = np.array(ndi.center_of_mass(pending[-1][1]))
                    aorta_i = min(range(n), key=lambda i: np.linalg.norm(centroids[i] - last_c))
                else:
                    aorta_i = int(np.argmin(areas))
                a_mask = lbl == comps[aorta_i]
                e_mask = (lbl != comps[aorta_i]) & m
                confident = True
            else:
                cap = max(np.median(trusted_areas[-15:]) * 2, 500)
                candidates = [i for i in range(n) if areas[i] <= cap]
                if candidates:
                    aorta_i = min(candidates, key=lambda i: np.linalg.norm(centroids[i] - prev_centroid))
                    a_mask = lbl == comps[aorta_i]
                    e_mask = (lbl != comps[aorta_i]) & m
                    confident = True
                else:
                    nearest_i = min(range(n), key=lambda i: np.linalg.norm(centroids[i] - prev_centroid))
                    nearest_comp = lbl == comps[nearest_i]
                    rest = m & ~nearest_comp
                    a_local, e_local = split_fused_slice(nearest_comp, prev_centroid, cap, np.median(trusted_areas[-15:]))
                    if a_local is None:
                        a_mask = np.zeros_like(m)
                        e_mask = m.copy()
                    else:
                        a_mask = a_local
                        e_mask = e_local | rest
                        confident = True
        else:
            size_cap = max(np.median(trusted_areas[-15:]) * 2, 500) if trusted_areas else None
            aorta_hint = np.median(trusted_areas[-15:]) if trusted_areas else None
            anchor = np.array(ndi.center_of_mass(pending[-1][1])) if pending and prev_centroid is None else prev_centroid
            a_mask, e_mask = split_fused_slice(m, anchor, size_cap, aorta_hint)
            if a_mask is None:
                if prev_centroid is None:
                    pending.append((z, m))
                    continue
                e_mask = m
                a_mask = np.zeros_like(m)
            else:
                confident = True

        if pending:
            last_c = np.array(ndi.center_of_mass(pending[-1][1]))
            if a_mask.any() and (not e_mask.any() or np.linalg.norm(last_c - np.array(ndi.center_of_mass(a_mask))) < np.linalg.norm(last_c - np.array(ndi.center_of_mass(e_mask)))):
                for pz, pm in pending:
                    aorta[:, :, pz] = pm
                    trusted_areas.append(int(pm.sum()))
            else:
                for pz, pm in pending:
                    esophagus[:, :, pz] = pm
            pending = []

        aorta[:, :, z] = a_mask
        esophagus[:, :, z] = e_mask
        if not confident:
            flagged_zs.append(z)
        if a_mask.any():
            prev_centroid = np.array(ndi.center_of_mass(a_mask))
            if confident:
                trusted_areas.append(int(a_mask.sum()))

    if pending:
        for pz, pm in pending:
            esophagus[:, :, pz] = pm
            flagged_zs.append(pz)

    aorta, esophagus = reclaim_stray_esophagus_fragments(aorta, esophagus)
    return esophagus, aorta, flagged_zs


def disconnected_gap_zs(mask: np.ndarray, spacing: np.ndarray, min_gap_mm: float = 2.0) -> list[int]:
    lbl, n = ndi.label(mask, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return []
    comps = [lbl == c for c in range(1, n + 1)]
    edts = [ndi.distance_transform_edt(~comps[i], sampling=spacing) for i in range(n)]
    dists = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if i != j:
                dists[i, j] = edts[j][comps[i]].min()
    in_tree = [0]
    remaining = set(range(1, n))
    edges = []
    while remaining:
        i, j, d = min(((i, j, dists[i, j]) for i in in_tree for j in remaining), key=lambda x: x[2])
        edges.append((i, j, d))
        in_tree.append(j)
        remaining.remove(j)

    gap_zs = set()
    for i, j, d in edges:
        if d < min_gap_mm:
            continue
        zi = np.where(comps[i].any(axis=(0, 1)))[0]
        zj = np.where(comps[j].any(axis=(0, 1)))[0]
        lo, hi = sorted((zi.max(), zj.min())) if zi.max() < zj.min() else sorted((zj.max(), zi.min()))
        gap_zs.update(range(lo, hi + 1))
    return sorted(gap_zs)

def report(name: str, mask: np.ndarray) -> None:
    if not mask.any():
        print(f"  {name:<10} not present")
        return
    zs = np.where(mask.any(axis=(0, 1)))[0]
    print(f"  {name:<10} {len(zs):>4} slices   {int(mask.sum()):>7} px")

def process_patient(patient_dir: Path, patient_name: str, verbose: bool = True) -> dict:

    gt_img = nib.load(patient_dir / "GT.nii.gz")
    ct_img = nib.load(patient_dir / f"{patient_name}.nii.gz")
    lab = np.asanyarray(gt_img.dataobj).astype(np.uint8)

    merged = lab == MERGED_LABEL
    trachea = lab == TRACHEA_LABEL

    if verbose:
        print(f"{patient_name}  shape={lab.shape}")
        print(f"  GT affine is identity (no orientation info): {(gt_img.affine == np.eye(4)).all()}")
        print(f"  CT orientation={nib.aff2axcodes(ct_img.affine)}  spacing={ct_img.header.get_zooms()[:3]}")
        print("\nBEFORE (label 1 = esophagus+aorta merged; label 3 = trachea, already separate):")
        report("label 1", merged)
        report("trachea", trachea)

    esophagus, aorta, flagged_zs = split_aorta_from_esophagus(merged, ct_img.affine)
    assert ((esophagus | aorta) == merged).all(), "esophagus + aorta should exactly cover merged"
    assert not (esophagus & aorta).any(), "esophagus/aorta should not overlap"

    out = lab.copy()
    out[merged] = 0
    out[esophagus] = 1
    out[aorta] = 4

    untouched = (lab != MERGED_LABEL)
    assert (out[untouched] == lab[untouched]).all(), "background/heart/trachea labels should be unchanged"


    touches_trachea = bool((ndi.binary_dilation(aorta, iterations=2) & trachea).any())

    aorta_zs = np.where(aorta.any(axis=(0, 1)))[0]
    aorta_areas = np.array([aorta[:, :, z].sum() for z in aorta_zs])
    max_median_ratio = float(aorta_areas.max() / max(np.median(aorta_areas), 1))

    spacing = np.abs(np.diag(ct_img.affine))[:3]
    review_zs = disconnected_gap_zs(aorta, spacing)

    if verbose:
        print("\nAFTER (split):")
        report("esophagus", out == 1)
        report("trachea", out == 3)
        report("aorta", out == 4)
        print(f"  sanity check: aorta contacts the true trachea somewhere: {touches_trachea}")
        print(f"  aorta area: median={np.median(aorta_areas):.0f} max={aorta_areas.max()} "
              f"max/median={max_median_ratio:.1f}x")
        if review_zs:
            print(f"  slices to review manually: {review_zs}")

    out_path = patient_dir / "GT_4label_v2.nii.gz"
    nib.save(nib.Nifti1Image(out, gt_img.affine, gt_img.header), out_path)
    if verbose:
        print(f"\nwrote {out_path}")

    return {
        "patient": patient_name,
        "touches_trachea": touches_trachea,
        "aorta_max_median_ratio": max_median_ratio,
        "review_zs": review_zs,
        "out_path": str(out_path),
    }

# Main console function to run the script
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("segthor_part1/data/segthor_part1/train"))
    ap.add_argument("--patient", default="Patient_01")
    ap.add_argument("--all", action="store_true", help="run every Patient_* folder under --data-dir instead")
    args = ap.parse_args()

    if not args.all:
        process_patient(args.data_dir / args.patient, args.patient)
        return 0

    patient_dirs = sorted(args.data_dir.glob("Patient_*"))
    results = []
    for patient_dir in patient_dirs:
        stats = process_patient(patient_dir, patient_dir.name, verbose=False)
        results.append(stats)
        flag = f"  REVIEW slices {stats['review_zs']}" if stats["review_zs"] else ""
        print(f"{stats['patient']}: touches_trachea={stats['touches_trachea']}  "
              f"aorta_max/median={stats['aorta_max_median_ratio']:.1f}x  -> wrote {Path(stats['out_path']).name}{flag}")

    n_ok = sum(r["touches_trachea"] for r in results)
    print(f"\n{n_ok}/{len(results)} patients: aorta contacts the true trachea")
    worst = max(results, key=lambda r: r["aorta_max_median_ratio"])
    print(f"worst-case aorta max/median ratio: {worst['aorta_max_median_ratio']:.1f}x ({worst['patient']})")

    need_review = [r for r in results if r["review_zs"]]
    print(f"\n{len(need_review)}/{len(results)} patients need manual review:")
    for r in need_review:
        print(f"  {r['patient']}: z = {r['review_zs']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
