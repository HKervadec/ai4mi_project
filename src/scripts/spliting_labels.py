from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.measure import regionprops

MERGED_LABEL = 1
TRACHEA_LABEL = 3
MAX_EROSION = 8
MIN_SPLIT_FRACTION = 0.05
MIN_SPLIT_PIXELS = 15
SIZE_SIMILAR_RATIO = 0.3
POSITION_AMBIGUOUS_PX = 20
DISAGREE_AT_START_THRESHOLD = 0.5

# helper
def ap_axis_and_sign(affine: np.ndarray) -> tuple[int, int]:
    codes = nib.aff2axcodes(affine)
    for axis, code in enumerate(codes[:2]):
        if code in ("A", "P"):
            return axis, (1 if code == "A" else -1)
    raise ValueError(f"expected an anterior/posterior in-plane axis, got orientation {codes}")




def split_fused_slice_fwd(m: np.ndarray, prev_centroid: np.ndarray | None = None, size_cap: float | None = None, aorta_hint: float | None = None) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:

    min_piece = max(m.sum() * MIN_SPLIT_FRACTION, MIN_SPLIT_PIXELS)
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
        if min(a.sum(), e.sum()) < min_piece:
            continue
        if size_cap is not None and a.sum() > size_cap:
            continue
        if aorta_hint is None:
            return a, e
        score = abs(int(a.sum()) - aorta_hint)
        if best_score is None or score < best_score:
            best, best_score = (a, e), score
    return best if best is not None else (None, None)

def reclaim_stray_esophagus_fragments_fwd(aorta: np.ndarray, esophagus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

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

def split_aorta_from_esophagus_fwd(merged: np.ndarray, ct_affine: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
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
                aorta_i = int(np.argmin(areas))
                a_mask = lbl == comps[aorta_i]
                e_mask = (lbl != comps[aorta_i]) & m
                confident = True
            else:
                trusted_median = np.median(trusted_areas[-15:])
                cap = max(trusted_median * 2, 500)
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
                    a_local, e_local = split_fused_slice_fwd(nearest_comp, prev_centroid, cap, trusted_median)
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
            a_mask, e_mask = split_fused_slice_fwd(m, prev_centroid, size_cap, aorta_hint)
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

    aorta, esophagus = reclaim_stray_esophagus_fragments_fwd(aorta, esophagus)
    return esophagus, aorta, flagged_zs



def split_fused_slice_bwd(m: np.ndarray, anchor_centroid: np.ndarray | None = None, size_cap: float | None = None, size_hint: float | None = None) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:

    min_piece = max(m.sum() * MIN_SPLIT_FRACTION, MIN_SPLIT_PIXELS)
    best, best_score = None, None
    for k in range(1, MAX_EROSION + 1):
        eroded = ndi.binary_erosion(m, iterations=k)
        lbl, n = ndi.label(eroded)
        if n < 2:
            continue
        if anchor_centroid is None:
            sizes = [(lbl == c).sum() for c in range(1, n + 1)]
            anchor_c = int(np.argmin(sizes)) + 1
        else:
            centroids = [np.array(ndi.center_of_mass(lbl == c)) for c in range(1, n + 1)]
            anchor_c = min(range(n), key=lambda i: np.linalg.norm(centroids[i] - anchor_centroid)) + 1
        markers = np.zeros(m.shape, dtype=np.int32)
        markers[lbl == anchor_c] = 1
        markers[(lbl != anchor_c) & eroded] = 2
        labels = watershed(np.zeros(m.shape), markers=markers, mask=m)
        anchored, rest = labels == 1, labels == 2
        if min(anchored.sum(), rest.sum()) < min_piece:
            continue
        if size_cap is not None and anchored.sum() > size_cap:
            continue
        if size_hint is None:
            return anchored, rest
        score = abs(int(anchored.sum()) - size_hint)
        if best_score is None or score < best_score:
            best, best_score = (anchored, rest), score
    return best if best is not None else (None, None)

def reclaim_stray_esophagus_fragments_bwd(aorta: np.ndarray, esophagus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

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

def reclaim_stray_aorta_fragments_bwd(aorta: np.ndarray, esophagus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    lbl, n = ndi.label(aorta, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return aorta, esophagus

    eso_zs = np.where(esophagus.any(axis=(0, 1)))[0]
    eso_median_area = np.median([esophagus[:, :, z].sum() for z in eso_zs]) if len(eso_zs) else 0
    area_cap = max(eso_median_area * 2, 50)

    sizes = ndi.sum(aorta, lbl, range(1, n + 1))
    main_c = int(np.argmax(sizes)) + 1
    eso_dilated = ndi.binary_dilation(esophagus, iterations=2)
    for c in range(1, n + 1):
        if c == main_c:
            continue
        frag = lbl == c
        frag_zs = np.where(frag.any(axis=(0, 1)))[0]
        max_slice_area = max((frag[:, :, z].sum() for z in frag_zs), default=0)
        if max_slice_area > area_cap:
            continue
        if (eso_dilated & frag).any():
            esophagus = esophagus | frag
            aorta = aorta & ~frag
    return aorta, esophagus

def eccentricity_of(mask: np.ndarray) -> float:
    props = regionprops(mask.astype(np.uint8))
    if not props:
        return 0.0
    return props[0].eccentricity

def pick_esophagus_candidate(candidate_idxs: list, centroids: list, areas: list, comps_masks: list, prev_eso_centroid: np.ndarray, trusted_median: float) -> int:
    order_by_dist = sorted(candidate_idxs, key=lambda i: np.linalg.norm(centroids[i] - prev_eso_centroid))
    close = [i for i in order_by_dist if np.linalg.norm(centroids[i] - prev_eso_centroid) < POSITION_AMBIGUOUS_PX]
    if len(close) <= 1:
        return order_by_dist[0]
    by_size = sorted(close, key=lambda i: abs(areas[i] - trusted_median))
    best, second = by_size[0], by_size[1]
    if areas[second] == 0 or abs(areas[best] - areas[second]) / max(areas[best], areas[second]) > SIZE_SIMILAR_RATIO:
        return best
    ecc_best = eccentricity_of(comps_masks[best])
    ecc_second = eccentricity_of(comps_masks[second])
    return best if ecc_best >= ecc_second else second

def split_aorta_from_esophagus_bwd(merged: np.ndarray, ct_affine: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    aorta = np.zeros_like(merged)
    esophagus = np.zeros_like(merged)
    prev_eso_centroid = None
    trusted_eso_areas: list[int] = []
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
            comps_masks = [lbl == c for c in comps]

            if prev_eso_centroid is None:
                if pending:
                    last_c = np.array(ndi.center_of_mass(pending[-1][1]))
                    eso_i = min(range(n), key=lambda i: np.linalg.norm(centroids[i] - last_c))
                else:
                    eso_i = int(np.argmin(areas))
                e_mask = lbl == comps[eso_i]
                a_mask = (lbl != comps[eso_i]) & m
                confident = True
            else:
                trusted_median = np.median(trusted_eso_areas[-15:])
                cap = max(trusted_median * 2, 500)
                candidates = [i for i in range(n) if areas[i] <= cap]
                if candidates:
                    eso_i = pick_esophagus_candidate(candidates, centroids, areas, comps_masks, prev_eso_centroid, trusted_median)
                    e_mask = lbl == comps[eso_i]
                    a_mask = (lbl != comps[eso_i]) & m
                    confident = True
                else:
                    nearest_i = min(range(n), key=lambda i: np.linalg.norm(centroids[i] - prev_eso_centroid))
                    nearest_comp = lbl == comps[nearest_i]
                    rest = m & ~nearest_comp
                    e_local, a_local = split_fused_slice_bwd(nearest_comp, prev_eso_centroid, cap, trusted_median)
                    if e_local is None:
                        a_mask = m.copy()
                        e_mask = np.zeros_like(m)
                    else:
                        e_mask = e_local
                        a_mask = a_local | rest
                        confident = True
        else:
            size_cap = max(np.median(trusted_eso_areas[-15:]) * 2, 500) if trusted_eso_areas else None
            eso_hint = np.median(trusted_eso_areas[-15:]) if trusted_eso_areas else None
            e_mask, a_mask = split_fused_slice_bwd(m, prev_eso_centroid, size_cap, eso_hint)
            if e_mask is None:
                if prev_eso_centroid is None:
                    pending.append((z, m))
                    continue
                whole_centroid = np.array(ndi.center_of_mass(m))
                if (size_cap is None or m.sum() <= size_cap) and np.linalg.norm(whole_centroid - prev_eso_centroid) < 20:
                    e_mask = m
                    a_mask = np.zeros_like(m)
                    confident = True
                else:
                    a_mask = m
                    e_mask = np.zeros_like(m)
            else:
                confident = True

        if pending:
            last_c = np.array(ndi.center_of_mass(pending[-1][1]))
            if e_mask.any() and (not a_mask.any() or np.linalg.norm(last_c - np.array(ndi.center_of_mass(e_mask))) < np.linalg.norm(last_c - np.array(ndi.center_of_mass(a_mask)))):
                for pz, pm in pending:
                    esophagus[:, :, pz] = pm
                    trusted_eso_areas.append(int(pm.sum()))
            else:
                for pz, pm in pending:
                    aorta[:, :, pz] = pm
            pending = []

        aorta[:, :, z] = a_mask
        esophagus[:, :, z] = e_mask
        if not confident:
            flagged_zs.append(z)
        if e_mask.any():
            prev_eso_centroid = np.array(ndi.center_of_mass(e_mask))
            if confident:
                trusted_eso_areas.append(int(e_mask.sum()))

    if pending:
        for pz, pm in pending:
            aorta[:, :, pz] = pm
            flagged_zs.append(pz)

    aorta, esophagus = reclaim_stray_esophagus_fragments_bwd(aorta, esophagus)
    aorta, esophagus = reclaim_stray_aorta_fragments_bwd(aorta, esophagus)
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

def disagreement_zs(esophagus_fwd: np.ndarray, esophagus_bwd: np.ndarray, min_px: int = 10) -> list[int]:
    zs = np.where((esophagus_fwd | esophagus_bwd).any(axis=(0, 1)))[0]
    flagged = []
    for z in zs:
        diff = (esophagus_fwd[:, :, z] != esophagus_bwd[:, :, z]).sum()
        if diff >= min_px:
            flagged.append(int(z))
    return flagged

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

    esophagus_fwd, aorta_fwd, _ = split_aorta_from_esophagus_fwd(merged, ct_img.affine)

    head_first = merged[:, :, ::-1]
    esophagus_bwd_r, aorta_bwd_r, _ = split_aorta_from_esophagus_bwd(head_first, ct_img.affine)
    esophagus_bwd = esophagus_bwd_r[:, :, ::-1]

    zs_all = np.where(merged.any(axis=(0, 1)))[0]
    first_zs = zs_all[:10]
    disagree_at_start = sum(
        1 for z in first_zs if (esophagus_fwd[:, :, z] != esophagus_bwd[:, :, z]).sum() >= 10
    ) / max(len(first_zs), 1)

    spacing = np.abs(np.diag(ct_img.affine))[:3]
    if disagree_at_start > DISAGREE_AT_START_THRESHOLD:
        disagree_zs = disconnected_gap_zs(aorta_fwd, spacing)
        backward_unreliable = True
    else:
        disagree_zs = disagreement_zs(esophagus_fwd, esophagus_bwd)
        backward_unreliable = False

    esophagus, aorta = esophagus_fwd, aorta_fwd
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

    if verbose:
        print("\nAFTER (split, forward/v3 method used for the saved labels):")
        report("esophagus", out == 1)
        report("trachea", out == 3)
        report("aorta", out == 4)
        print(f"  sanity check: aorta contacts the true trachea somewhere: {touches_trachea}")
        print(f"  aorta area: median={np.median(aorta_areas):.0f} max={aorta_areas.max()} "
              f"max/median={max_median_ratio:.1f}x")
        print(f"  disagree_at_start={disagree_at_start:.0%}  backward_unreliable={backward_unreliable}")
        print(f"  review method: {'forward gap-based fallback' if backward_unreliable else 'forward/backward disagreement'}")
        if disagree_zs:
            print(f"  slices to review manually: {disagree_zs}")

    out_path = patient_dir / "GT_4label_v6.nii.gz"
    nib.save(nib.Nifti1Image(out, gt_img.affine, gt_img.header), out_path)
    if verbose:
        print(f"\nwrote {out_path}")

    return {
        "patient": patient_name,
        "touches_trachea": touches_trachea,
        "aorta_max_median_ratio": max_median_ratio,
        "disagree_zs": disagree_zs,
        "backward_unreliable": backward_unreliable,
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
        flag = f"  DISAGREE slices {stats['disagree_zs']}" if stats["disagree_zs"] else ""
        print(f"{stats['patient']}: touches_trachea={stats['touches_trachea']}  "
              f"aorta_max/median={stats['aorta_max_median_ratio']:.1f}x  -> wrote {Path(stats['out_path']).name}{flag}")

    n_ok = sum(r["touches_trachea"] for r in results)
    print(f"\n{n_ok}/{len(results)} patients: aorta contacts the true trachea")
    worst = max(results, key=lambda r: r["aorta_max_median_ratio"])
    print(f"worst-case aorta max/median ratio: {worst['aorta_max_median_ratio']:.1f}x ({worst['patient']})")

    need_review = [r for r in results if r["disagree_zs"]]
    print(f"\n{len(need_review)}/{len(results)} patients need manual review (forward/backward disagreement):")
    for r in need_review:
        print(f"  {r['patient']}: z = {r['disagree_zs']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
