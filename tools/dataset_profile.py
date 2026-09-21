#!/usr/bin/env python3
"""
One pass over the SegTHOR training volumes that writes tidy per-patient,
per-label, per-slice and per-label-pair tables, plus exact 1 HU intensity
histograms. Figure scripts read these tables instead of reloading volumes.

Labels are referred to by number only.

Usage:
    python tools/dataset_profile.py --data-dir data/segthor_part1/train --out-dir figures/profile
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label

LABELS = [1, 2, 3, 4]
INTENSITY_GROUPS = {"background": [0], "label 1": [1], "label 2": [2], "label 3": [3], "label 4": [4], "all labels": [1, 2, 3, 4]}
HU_OFFSET = 4096  # added before np.bincount so every stored CT value maps to a non-negative bin
NNUNET_SAMPLES_PER_CASE = 5_000_000  # fingerprint_extractor.py: 10e7 // 20 training cases
NNUNET_SEED = 1234  # fingerprint_extractor.py: analyze_case(..., seed=1234)
FACE = generate_binary_structure(3, 1)
FACE_EDGE_CORNER = generate_binary_structure(3, 3)

TABLES = ("patients", "labels", "label_slices", "scan_slices", "label_pairs")
HIST_FILE = "intensity_histograms.npz"


def hist_stats(counts: np.ndarray, start: int) -> dict | None:
    """Exact statistics of integer values given as a 1-unit histogram.

    Percentiles use the same linear interpolation as np.percentile.

    Args:
        counts: Number of values in each unit bin.
        start: Value of the first bin.

    Returns:
        min, p0.5, median, p99.5, max, mean, std and n, or None if the histogram is empty.
    """
    n = int(counts.sum())
    if n == 0:
        return None
    values = np.arange(start, start + len(counts), dtype=np.float64)
    cum = np.cumsum(counts)

    def value_at(rank: int) -> float:
        return float(values[np.searchsorted(cum, rank + 1)])

    def percentile(q: float) -> float:
        pos = q / 100 * (n - 1)
        lo, hi = int(np.floor(pos)), int(np.ceil(pos))
        return value_at(lo) + (value_at(hi) - value_at(lo)) * (pos - lo)

    mean = float((values * counts).sum() / n)
    nonzero = np.flatnonzero(counts)
    return {
        "min": float(values[nonzero[0]]),
        "p0.5": percentile(0.5),
        "median": percentile(50),
        "p99.5": percentile(99.5),
        "max": float(values[nonzero[-1]]),
        "mean": mean,
        "std": float(np.sqrt((((values - mean) ** 2) * counts).sum() / n)),
        "n": n,
    }


def binned_pcts(counts: np.ndarray, start: int, edges: np.ndarray) -> np.ndarray:
    """Percentage of voxels in each of len(edges)-1 HU bins, plus below/above-range columns.

    Args:
        counts: 1 HU histogram counts.
        start: HU value of the first bin in counts.
        edges: Bin edges spanning the in-range portion (edges[0]..edges[-1]).

    Returns:
        Array of length len(edges)+1: [below edges[0], one per bin, above edges[-1]], in percent.
    """
    n = counts.sum()
    if n == 0:
        return np.full(len(edges) + 1, np.nan)
    values = start + np.arange(len(counts))
    bin_idx = np.digitize(values, edges)  # 0 = below edges[0], len(edges) = at/above edges[-1]
    out = np.zeros(len(edges) + 1)
    for i in range(len(edges) + 1):
        out[i] = counts[bin_idx == i].sum()
    return 100 * out / n


def _trimmed_hist(vals: np.ndarray) -> tuple[np.ndarray, int]:
    if vals.size == 0:
        return np.zeros(0, dtype=np.int64), 0
    counts = np.bincount(vals.ravel().astype(np.int64) + HU_OFFSET)
    nonzero = np.flatnonzero(counts)
    return counts[nonzero[0] : nonzero[-1] + 1], int(nonzero[0] - HU_OFFSET)


def _bbox(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = [], []
    for axis in range(mask.ndim):
        hits = np.flatnonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != axis)))
        lo.append(hits[0])
        hi.append(hits[-1])
    return np.array(lo), np.array(hi)


def label_row(mask: np.ndarray, zooms: np.ndarray) -> dict:
    """Size, extent, position and piece counts of one label in one patient.

    Positions are in mm from the first voxel of the array (index * spacing).

    Args:
        mask: Boolean mask of the label.
        zooms: Voxel spacing (x, y, z) in mm.

    Returns:
        Flat dict of measurements; only voxel counts are filled for an empty mask.
    """
    n = int(mask.sum())
    row = {"voxels": n, "volume_ml": n * float(np.prod(zooms)) / 1000, "pct_of_scan": 100 * n / mask.size}
    if n == 0:
        return row
    lo, hi = _bbox(mask)
    idx = np.argwhere(mask)
    for k, axis in enumerate("xyz"):
        row[f"{axis}_min_mm"] = lo[k] * zooms[k]
        row[f"{axis}_max_mm"] = hi[k] * zooms[k]
        row[f"{axis}_size_mm"] = (hi[k] - lo[k] + 1) * zooms[k]
        row[f"{axis}_center_mm"] = idx[:, k].mean() * zooms[k]
    present = mask.any(axis=(0, 1))
    row["first_slice"], row["last_slice"] = int(lo[2]), int(hi[2])
    row["slices_with_label"] = int(present.sum())
    row["empty_slices_inside_range"] = int((~present[lo[2] : hi[2] + 1]).sum())
    sub = mask[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1]
    faces, n_faces = label(sub, structure=FACE)
    row["pieces_sharing_a_face"] = int(n_faces)
    row["pieces_sharing_face_edge_or_corner"] = int(label(sub, structure=FACE_EDGE_CORNER)[1])
    row["largest_piece_pct_face"] = 100 * np.bincount(faces.ravel())[1:].max() / n
    return row


def slice_rows(mask: np.ndarray, zooms: np.ndarray) -> list[dict]:
    """Per axial slice measurements of one label, for slices that contain it.

    Args:
        mask: Boolean mask of the label, shape (x, y, z).
        zooms: Voxel spacing (x, y, z) in mm.

    Returns:
        One dict per slice with index, height, area, left-right and anterior–posterior size, and piece count.
    """
    rows = []
    for z in np.flatnonzero(mask.any(axis=(0, 1))):
        sl = mask[:, :, z]
        (x0, y0), (x1, y1) = _bbox(sl)
        rows.append(
            {
                "slice": int(z),
                "z_mm": z * zooms[2],
                "area_mm2": sl.sum() * zooms[0] * zooms[1],
                "x_size_mm": (x1 - x0 + 1) * zooms[0],
                "y_size_mm": (y1 - y0 + 1) * zooms[1],
                "pieces_sharing_an_edge": int(label(sl)[1]),
            }
        )
    return rows


def pair_row(a: np.ndarray, b: np.ndarray, zooms: np.ndarray) -> dict:
    """Closest distance and shared face area between two label masks.

    Args:
        a: Boolean mask of the first label.
        b: Boolean mask of the second label.
        zooms: Voxel spacing (x, y, z) in mm.

    Returns:
        closest_voxel_centers_mm (NaN if either mask is empty; one voxel spacing when
        they share a face) and shared_face_area_mm2 (0 when no voxel faces meet).
    """
    if not a.any() or not b.any():
        return {"closest_voxel_centers_mm": np.nan, "shared_face_area_mm2": 0.0}
    face_area = [zooms[1] * zooms[2], zooms[0] * zooms[2], zooms[0] * zooms[1]]
    shared = 0.0
    for axis in range(3):
        head = [slice(None)] * 3
        tail = [slice(None)] * 3
        head[axis], tail[axis] = slice(None, -1), slice(1, None)
        head, tail = tuple(head), tuple(tail)
        shared += face_area[axis] * int((a[head] & b[tail]).sum() + (b[head] & a[tail]).sum())
    lo, hi = _bbox(a | b)
    box = tuple(slice(s, e + 1) for s, e in zip(lo, hi))
    dist_to_a = distance_transform_edt(~a[box], sampling=zooms)
    return {"closest_voxel_centers_mm": float(dist_to_a[b[box]].min()), "shared_face_area_mm2": shared}


def occupied_box(ct: np.ndarray, zooms: np.ndarray) -> dict:
    """Box around every voxel above the scan's lowest value.

    Args:
        ct: CT array (x, y, z).
        zooms: Voxel spacing (x, y, z) in mm.

    Returns:
        Box size per axis in mm, box volume as % of the image, and voxels outside the box
        on each side; sizes are 0 if every voxel equals the minimum.
    """
    above = ct > ct.min()
    row = {}
    if not above.any():
        return {f"occupied_{axis}_size_mm": 0.0 for axis in "xyz"} | {"occupied_pct_of_image": 0.0}
    lo, hi = _bbox(above)
    for k, axis in enumerate("xyz"):
        row[f"occupied_{axis}_size_mm"] = (hi[k] - lo[k] + 1) * zooms[k]
        row[f"outside_box_{axis}_low_voxels"] = int(lo[k])
        row[f"outside_box_{axis}_high_voxels"] = int(ct.shape[k] - 1 - hi[k])
    row["occupied_pct_of_image"] = 100 * float(np.prod(hi - lo + 1)) / ct.size
    return row


def identical_neighbor_slices(ct: np.ndarray) -> int:
    """Number of axial slices that are voxel-for-voxel identical to the next slice.

    Args:
        ct: CT array (x, y, z).

    Returns:
        Count of k where slice k equals slice k + 1.
    """
    return int(sum(np.array_equal(ct[:, :, k], ct[:, :, k + 1]) for k in range(ct.shape[2] - 1)))


def split_by_patient(processed_dir: Path) -> dict:
    """Train/validation membership from the sliced PNG folders.

    Args:
        processed_dir: Folder with train/img and val/img PNGs named Patient_XX_NNNN.png.

    Returns:
        {patient id: "train" or "val"}.
    """
    return {
        png.name.rsplit("_", 1)[0]: split
        for split in ("train", "val")
        for png in (processed_dir / split / "img").glob("Patient_*.png")
    }


def profile_patient(patient_dir: Path) -> dict:
    """All tables and histograms for one patient.

    Args:
        patient_dir: Folder with Patient_XX.nii.gz and GT.nii.gz.

    Returns:
        Dict with one "patient" row, lists of label, slice, scan-slice and pair rows,
        and {group: (counts, start)} intensity histograms including the nnU-Net sample.
    """
    pid = patient_dir.name
    img = nib.load(str(patient_dir / f"{pid}.nii.gz"))
    gt = nib.load(str(patient_dir / "GT.nii.gz"))
    ct = np.asarray(img.dataobj)
    seg = np.asarray(gt.dataobj)
    zooms = np.array(img.header.get_zooms()[:3], dtype=float)

    patient = {"patient": pid}
    for k, axis in enumerate("xyz"):
        patient[f"{axis}_spacing_mm"] = zooms[k]
        patient[f"{axis}_voxels"] = ct.shape[k]
        patient[f"{axis}_extent_mm"] = ct.shape[k] * zooms[k]
    patient["voxels"] = ct.size
    patient["label_grid_shape_matches"] = ct.shape == seg.shape
    patient["label_grid_spacing_matches"] = bool(np.allclose(zooms, gt.header.get_zooms()[:3]))
    patient["max_affine_difference"] = float(np.abs(img.affine - gt.affine).max())
    patient["image_axis_codes"] = "".join(nib.aff2axcodes(img.affine))
    patient["label_axis_codes"] = "".join(nib.aff2axcodes(gt.affine))
    patient["label_values_found"] = " ".join(str(v) for v in np.unique(seg))
    patient["identical_neighbor_slices"] = identical_neighbor_slices(ct)
    patient.update(occupied_box(ct, zooms))

    hists = {}
    for name, values in INTENSITY_GROUPS.items():
        hists[name] = _trimmed_hist(ct[np.isin(seg, values)])
    hists["scan"] = _trimmed_hist(ct)
    rs = np.random.RandomState(NNUNET_SEED)
    fg = ct[seg > 0]
    hists["nnunet sample"] = _trimmed_hist(rs.choice(fg, NNUNET_SAMPLES_PER_CASE, replace=True))

    flat = ct.reshape(-1, ct.shape[2])
    scan_slices = pd.DataFrame(
        {
            "patient": pid,
            "slice": np.arange(ct.shape[2]),
            "z_mm": np.arange(ct.shape[2]) * zooms[2],
            "median_hu": np.median(flat, axis=0),
        }
    )
    above = ct > ct.min()
    for k, axis in enumerate("xy"):
        other = 1 - k
        filled = above.any(axis=other)
        first = np.where(filled.any(axis=0), filled.argmax(axis=0), -1)
        last = np.where(filled.any(axis=0), ct.shape[k] - 1 - filled[::-1].argmax(axis=0), -2)
        scan_slices[f"occupied_{axis}_size_mm"] = (last - first + 1) * zooms[k]
    for c in LABELS:
        scan_slices[f"label_{c}_voxels"] = (seg == c).reshape(-1, seg.shape[2]).sum(axis=0)

    masks = {c: seg == c for c in LABELS}
    labels = [{"patient": pid, "label": c, **label_row(masks[c], zooms)} for c in LABELS]
    slices = [{"patient": pid, "label": c, **r} for c in LABELS for r in slice_rows(masks[c], zooms)]
    pairs = [
        {"patient": pid, "label_a": a, "label_b": b, **pair_row(masks[a], masks[b], zooms)}
        for i, a in enumerate(LABELS)
        for b in LABELS[i + 1 :]
    ]
    return {
        "patient": patient,
        "labels": labels,
        "slices": slices,
        "scan_slices": scan_slices,
        "pairs": pairs,
        "hists": hists,
    }


def load_tables(out_dir: Path) -> dict:
    """Load the tables and histograms written by this script.

    Args:
        out_dir: Directory passed as --out-dir.

    Returns:
        {table name: DataFrame} plus "histograms": {(patient, group): (counts, start)}.
    """
    tables = {name: pd.read_csv(out_dir / f"{name}.csv") for name in TABLES}
    npz = np.load(out_dir / HIST_FILE)
    tables["histograms"] = {
        tuple(key.split("|")[:2]): (npz[key], int(npz[key.replace("|counts", "|start")]))
        for key in npz.files
        if key.endswith("|counts")
    }
    return tables


def main():
    """Profile every patient and write the tables and histograms."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_part1/train"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures/profile"))
    ap.add_argument("--processed-dir", type=Path, default=Path("data/SEGTHOR"))
    ap.add_argument("--max-patients", type=int, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(p for p in args.data_dir.iterdir() if p.is_dir() and p.name.startswith("Patient_"))
    split = split_by_patient(args.processed_dir)
    patients, labels, slices, scan_slices, pairs, hist_arrays = [], [], [], [], [], {}
    for d in patient_dirs[: args.max_patients]:
        result = profile_patient(d)
        patients.append({**result["patient"], "split": split.get(d.name, "not in split")})
        labels += result["labels"]
        slices += result["slices"]
        scan_slices.append(result["scan_slices"])
        pairs += result["pairs"]
        for group, (counts, start) in result["hists"].items():
            hist_arrays[f"{d.name}|{group}|counts"] = counts
            hist_arrays[f"{d.name}|{group}|start"] = np.array(start)
        print(f"profiled {d.name}", flush=True)

    pd.DataFrame(patients).to_csv(args.out_dir / "patients.csv", index=False)
    pd.DataFrame(labels).to_csv(args.out_dir / "labels.csv", index=False)
    pd.DataFrame(slices).to_csv(args.out_dir / "label_slices.csv", index=False)
    pd.concat(scan_slices).to_csv(args.out_dir / "scan_slices.csv", index=False)
    pd.DataFrame(pairs).to_csv(args.out_dir / "label_pairs.csv", index=False)
    np.savez_compressed(args.out_dir / HIST_FILE, **hist_arrays)
    print(f"wrote {len(patients)} patients to {args.out_dir}")


if __name__ == "__main__":
    main()
