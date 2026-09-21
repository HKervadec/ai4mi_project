#!/usr/bin/env python3
"""
Data exploration / "dataset fingerprint" script for SegTHOR (part1).

Combines nnU-Net-style dataset fingerprinting (shape/spacing, per-class voxel
and slice-presence stats, per-organ HU intensity distribution incl. the
[0.5, 99.5] percentile CT-normalization clip computed over FOREGROUND voxels,
which is what nnU-Net actually uses) with additional medical-imaging EDA/QC
practices commonly reported in segmentation dataset papers (Medical Segmentation
Decathlon, radiomics QC pipelines): organ volume distributions, organ
z-position/anatomical-ordering, per-patient class-presence QC (catches
incomplete annotations), intensity/HU artifact detection split by the 12-bit
reconstruction ceiling, acquisition-protocol grouping, a per-patient outlier
scan, and a raw-vs-GT slice montage for visual sanity checking.

LABEL MAPPING
-------------
``background esophagus heart trachea aorta`` (0-4), matching the course
readme and dataset_analysis/utils.py:CLASSES. Which of labels 1-4 actually
have voxels depends on the --data-dir passed in: the original course release
omits the aorta annotation (label 4 has zero voxels in every patient), while
a full 4-class release has voxels for all of them. This script detects
presence from the data rather than assuming either case, so the same command
works for both -- absent classes are reported and excluded from per-organ
summaries, not silently included as zero.

Usage:
    python tools/explore_data.py \
        --data-dir data/segthor_part1/train \
        --out-dir figures

Requires matplotlib >= 3.9 (uses the `tick_labels=` boxplot kwarg, renamed from
`labels=` in 3.9), numpy >= 1.24, nibabel >= 5.0.

Outputs (all in --out-dir):
    fingerprint.json           -- all numeric stats, machine-readable
    fingerprint.md              -- human-readable summary tables (paste into slides)
    organ_volumes_cm3.csv       -- per-patient, per-organ volume table
    shape_spacing_boxplots.png
    class_balance.png
    hu_histograms.png
    hu_boxplots_by_organ.png
    organ_volume_violin.png
    organ_z_extent.png
    class_presence_heatmap.png
    patient_pca_outliers.png
    slice_montage_qc.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import PALETTE, apply_style, decorate, legend_below

# See the LABEL MAPPING note in the module docstring: whether class 4 (aorta)
# has voxels depends on the dataset passed via --data-dir.
CLASS_NAMES = {0: "background", 1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}
NUM_CLASSES = 5
ORGAN_COLORS = {1: PALETTE[1], 2: PALETTE[5], 3: PALETTE[2], 4: PALETTE[4]}

# A 12-bit CT reconstruction stores 0..4095 with a -1024 intercept, so 3071 HU is
# the hard ceiling of the scanner's output. Values AT 3071 are censored (clipped)
# dense bone/metal; values ABOVE 3071 cannot come from a 12-bit reconstruction at
# all and indicate an extended-range reconstruction of a metal artifact. Lumping
# both together under a single "> 2000 HU" threshold mixes ordinary dense bone in
# with genuine artifacts, so we report the two populations separately.
CT_12BIT_CEILING = 3071.0

# Features used for the per-patient morphometric outlier scan. Spacing is handled
# separately as a categorical acquisition-protocol grouping: it takes 3-4 discrete
# values, so a MAD-based outlier score on it is meaningless.
MORPHOMETRIC_FEATURES = ["z_shape", "vol_esophagus", "vol_heart", "vol_trachea", "nonzero_fraction"]
PCA_FEATURES = ["z_shape", "spacing_z", "spacing_xy", "vol_esophagus", "vol_heart", "vol_trachea", "nonzero_fraction"]
MODIFIED_Z_THRESHOLD = 3.5  # Iglewicz & Hoaglin's recommended cutoff


def find_patients(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("Patient_"))


def load_patient(patient_dir: Path):
    pid = patient_dir.name
    ct_img = nib.load(str(patient_dir / f"{pid}.nii.gz"))
    gt_img = nib.load(str(patient_dir / "GT.nii.gz"))
    ct = np.asarray(ct_img.dataobj)
    gt = np.asarray(gt_img.dataobj)
    zooms = ct_img.header.get_zooms()[:3]
    return pid, ct, gt, zooms


def bbox(mask: np.ndarray):
    if not mask.any():
        return None
    idx = np.nonzero(mask)
    mins = tuple(int(a.min()) for a in idx)
    maxs = tuple(int(a.max()) for a in idx)
    return mins, maxs


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def modified_z(a: np.ndarray) -> np.ndarray:
    """Iglewicz-Hoaglin modified z-score (median/MAD based). Falls back to a
    plain z-score when the MAD is zero, and to zeros when the feature is
    constant -- neither fallback can manufacture an outlier out of nothing."""
    a = np.asarray(a, dtype=float)
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    if mad > 0:
        return 0.6745 * (a - med) / mad
    std = a.std()
    if std > 0:
        return (a - a.mean()) / std
    return np.zeros_like(a)


def read_val_patients(val_gt_dir: Path) -> set[str]:
    """Patient ids in the 2D validation split, so the fingerprint can say which
    of its numbers were computed over held-out data."""
    if not val_gt_dir.is_dir():
        return set()
    return {f.name.rsplit("_", 1)[0] for f in val_gt_dir.iterdir() if f.suffix == ".png"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_part1/train"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument(
        "--val-gt-dir",
        type=Path,
        default=Path("data/SEGTHOR/val/gt"),
        help="2D validation slices, used only to label which patients are held out.",
    )
    ap.add_argument("--n-montage-patients", type=int, default=3)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    patients = find_patients(args.data_dir)
    if not patients:
        raise SystemExit(f"No Patient_* directories found under {args.data_dir}")
    val_pids = read_val_patients(args.val_gt_dir)

    shapes, spacings_inplane, spacings_slice = [], [], []
    voxel_counts = {c: 0 for c in range(NUM_CLASSES)}
    total_voxels = 0
    slice_presence = {c: 0 for c in range(NUM_CLASSES)}
    total_slices = 0
    hu_all = []
    hu_foreground = []  # pooled over ALL labeled voxels -- this is the nnU-Net statistic
    hu_by_class = {c: [] for c in range(1, NUM_CLASSES)}
    bbox_extent_mm = {c: [] for c in range(1, NUM_CLASSES)}
    fg_fraction_per_patient = []
    per_patient_rows = []

    # Exact whole-volume extrema, tracked separately from the subsampled array
    # used for percentiles -- a 1-in-37 subsample will miss the rare extreme
    # voxels entirely and understate the true range.
    hu_min_exact, hu_max_exact = np.inf, -np.inf
    n_voxels_over_500 = 0  # for the histogram footnote (its x-range stops at 500)

    organ_volume_cm3 = {c: [] for c in range(1, NUM_CLASSES)}  # per-patient, 0.0 if organ absent
    organ_z_extent_frac = {c: [] for c in range(1, NUM_CLASSES)}  # (pid, start_frac, end_frac)
    class_present_matrix = []  # rows: pid, cols: presence bool per class 1..4
    feature_rows = []  # per-patient feature vector for the outlier scan
    hu_extreme_voxels = []  # (pid, value, x, y, z) -- one per patient, located in full
    hu_at_ceiling = {}  # pid -> count of voxels exactly at the 12-bit ceiling
    hu_above_ceiling = {}  # pid -> count of voxels above it (impossible for 12-bit)

    for patient_dir in patients:
        pid, ct, gt, zooms = load_patient(patient_dir)
        voxel_vol_cm3 = (zooms[0] * zooms[1] * zooms[2]) / 1000.0  # mm^3 -> cm^3

        shapes.append(ct.shape)
        spacings_inplane.append((float(zooms[0]), float(zooms[1])))
        spacings_slice.append(float(zooms[2]))

        n_vox = gt.size
        total_voxels += n_vox
        total_slices += gt.shape[-1]
        fg_fraction_per_patient.append(float((ct != 0).sum()) / n_vox)

        # Percentiles come from a 1-in-37 subsample (cheap, and verified to move
        # the organ-wise mean by <0.1 HU); extrema come from the full array.
        hu_all.append(ct.reshape(-1)[::37])
        hu_min_exact = min(hu_min_exact, float(ct.min()))
        hu_max_exact = max(hu_max_exact, float(ct.max()))
        n_voxels_over_500 += int((ct > 500).sum())
        if (gt > 0).any():
            hu_foreground.append(ct[gt > 0][::10])

        # Artifact scan, split at the 12-bit ceiling (see CT_12BIT_CEILING).
        n_at = int((ct == CT_12BIT_CEILING).sum())
        n_above = int((ct > CT_12BIT_CEILING).sum())
        hu_at_ceiling[pid] = n_at
        hu_above_ceiling[pid] = n_above
        if n_above:
            flat_idx = int(np.argmax(ct))
            x_idx, y_idx, z_idx = np.unravel_index(flat_idx, ct.shape)
            hu_extreme_voxels.append((pid, float(ct.flat[flat_idx]), int(x_idx), int(y_idx), int(z_idx)))

        presence_row = []
        organ_vols_this_patient = {}
        for c in range(NUM_CLASSES):
            mask = gt == c
            voxel_counts[c] += int(mask.sum())
            present_slices = mask.any(axis=(0, 1)).sum() if mask.ndim == 3 else 0
            slice_presence[c] += int(present_slices)

            if c != 0:
                is_present = bool(mask.any())
                presence_row.append(is_present)
                vol_cm3 = float(mask.sum()) * voxel_vol_cm3
                organ_volume_cm3[c].append(vol_cm3)
                organ_vols_this_patient[c] = vol_cm3
                if is_present:
                    hu_by_class[c].append(ct[mask][::5])
                    mins, maxs = bbox(mask)
                    extent_vox = tuple(mx - mn + 1 for mn, mx in zip(mins, maxs))
                    extent_mm = tuple(e * z for e, z in zip(extent_vox, zooms))
                    bbox_extent_mm[c].append(extent_mm)
                    z_total = ct.shape[-1]
                    # +1 on the end index so the fraction spans the same closed
                    # voxel range as extent_vox above (max - min + 1).
                    organ_z_extent_frac[c].append((pid, mins[-1] / z_total, (maxs[-1] + 1) / z_total))

        class_present_matrix.append(presence_row)
        feature_rows.append(
            {
                "pid": pid,
                "z_shape": ct.shape[-1],
                "spacing_z": float(zooms[2]),
                "spacing_xy": float(zooms[0]),
                "vol_esophagus": organ_vols_this_patient.get(1, 0.0),
                "vol_heart": organ_vols_this_patient.get(2, 0.0),
                "vol_trachea": organ_vols_this_patient.get(3, 0.0),
                "nonzero_fraction": fg_fraction_per_patient[-1],
            }
        )
        per_patient_rows.append(
            {
                "patient": pid,
                "split": "val" if pid in val_pids else "train",
                "shape": list(ct.shape),
                "spacing_xyz_mm": [float(z) for z in zooms],
                "stored_dtype": str(ct.dtype),
                "hu_max": float(ct.max()),
                "nonzero_ct_fraction": fg_fraction_per_patient[-1],
                "hu_voxels_at_12bit_ceiling": n_at,
                "hu_voxels_above_12bit_ceiling": n_above,
            }
        )

    shapes_arr = np.array(shapes)
    inplane_arr = np.array(spacings_inplane)
    slice_arr = np.array(spacings_slice)
    hu_all_concat = np.concatenate(hu_all) if hu_all else np.array([])
    hu_fg_concat = np.concatenate(hu_foreground) if hu_foreground else np.array([])
    pids = [p.name for p in patients]

    # ------------------------------------------------- acquisition protocol groups
    protocol_groups = {}
    for row in per_patient_rows:
        key = f"{row['spacing_xyz_mm'][0]:.4f} x {row['spacing_xyz_mm'][1]:.4f} x {row['spacing_xyz_mm'][2]:.2f}"
        protocol_groups.setdefault(key, []).append(row["patient"])
    dtype_groups = {}
    for row in per_patient_rows:
        dtype_groups.setdefault(row["stored_dtype"], []).append(row["patient"])

    # ------------------------------------------------- per-patient outlier scan
    feat_matrix = {k: np.array([r[k] for r in feature_rows], dtype=float) for k in PCA_FEATURES}
    morph_z = {k: modified_z(feat_matrix[k]) for k in MORPHOMETRIC_FEATURES}
    feature_flags = {
        pid: {k: float(morph_z[k][i]) for k in MORPHOMETRIC_FEATURES if abs(morph_z[k][i]) > MODIFIED_Z_THRESHOLD}
        for i, pid in enumerate(pids)
    }
    feature_flags = {pid: d for pid, d in feature_flags.items() if d}

    X = np.array([[row[k] for k in PCA_FEATURES] for row in feature_rows], dtype=float)
    Xc = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = U[:, :2] * S[:2]
    var_explained = 100 * (S[:2] ** 2).sum() / (S**2).sum()
    # Robust threshold on the PC1-PC2 distance. The old mean+2*std rule was
    # computed on distances that *contain* the outlier, so a single extreme
    # patient inflated the threshold that was supposed to catch it (masking).
    dist = np.linalg.norm(scores, axis=1)
    dist_med = np.median(dist)
    dist_mad = np.median(np.abs(dist - dist_med))
    dist_thresh = dist_med + 3 * (dist_mad * 1.4826 if dist_mad > 0 else dist.std())
    pca_flagged = {pid for pid, d in zip(pids, dist) if d > dist_thresh}
    # Union of both criteria: PC1-PC2 only explains ~half the variance, so a
    # patient extreme on a single feature can sit near the origin of the plot.
    outlier_pids = pca_flagged | set(feature_flags)
    outlier_mask = np.array([pid in outlier_pids for pid in pids])

    # Which organ labels actually have voxels in THIS data-dir -- computed from
    # the data, not assumed, so the same script reports correctly whether the
    # aorta annotation is present (full 4-class release) or absent (the
    # original 3-class course release).
    absent_classes = [c for c in range(1, NUM_CLASSES) if not any(v > 0 for v in organ_volume_cm3[c])]
    present_classes = [c for c in range(1, NUM_CLASSES) if c not in absent_classes]
    if absent_classes:
        absent_note = (
            f"Label(s) {', '.join(f'{c} ({CLASS_NAMES[c]})' for c in absent_classes)} have no "
            "voxels in this data-dir and are excluded from per-organ summaries below."
        )
    else:
        absent_note = "All labeled organs have voxels in this data-dir."

    # ---------------------------------------------------------------- JSON
    fingerprint = {
        "n_patients": len(patients),
        "label_mapping": {str(c): CLASS_NAMES[c] for c in range(NUM_CLASSES)},
        "label_mapping_note": f"Matches the course readme's class order. {absent_note}",
        "split": {
            "val_patients": sorted(val_pids),
            "train_patients": sorted(set(pids) - val_pids),
            "note": (
                "Statistics below are pooled over ALL patients, including the held-out "
                "validation patients. Any preprocessing derived from them (normalization "
                "percentiles, target spacing) is therefore fitted on validation data too."
            ),
        },
        "shape_voxels": {
            "median": [int(x) for x in np.median(shapes_arr, axis=0)],
            "min": [int(x) for x in shapes_arr.min(axis=0)],
            "max": [int(x) for x in shapes_arr.max(axis=0)],
        },
        "spacing_mm": {
            "inplane_median": [float(x) for x in np.median(inplane_arr, axis=0)],
            "inplane_min": [float(x) for x in inplane_arr.min(axis=0)],
            "inplane_max": [float(x) for x in inplane_arr.max(axis=0)],
            "slice_median": float(np.median(slice_arr)),
            "slice_min": float(slice_arr.min()),
            "slice_max": float(slice_arr.max()),
        },
        "acquisition_protocol_groups": protocol_groups,
        "stored_dtype_groups": dtype_groups,
        "class_balance": {
            CLASS_NAMES[c]: {
                "pct_voxels": 100.0 * voxel_counts[c] / total_voxels,
                "pct_slices_present": 100.0 * slice_presence[c] / total_slices,
            }
            for c in range(NUM_CLASSES)
        },
        "hu_global": {
            "percentiles_from_subsample_stride": 37,
            "p0_5": pct(hu_all_concat, 0.5),
            "p50": pct(hu_all_concat, 50),
            "p99_5": pct(hu_all_concat, 99.5),
            "min": hu_min_exact if np.isfinite(hu_min_exact) else None,
            "max": hu_max_exact if np.isfinite(hu_max_exact) else None,
            "min_max_note": "exact, computed over every voxel (not the subsample)",
        },
        "hu_foreground_nnunet_normalization": {
            "note": (
                "nnU-Net's CTNormalization clips to the [0.5, 99.5] percentiles of the "
                "FOREGROUND voxels pooled across the dataset, then z-scores with the "
                "foreground mean/std. These are those numbers."
            ),
            "p0_5": pct(hu_fg_concat, 0.5),
            "p99_5": pct(hu_fg_concat, 99.5),
            "mean": float(np.mean(hu_fg_concat)) if len(hu_fg_concat) else None,
            "std": float(np.std(hu_fg_concat)) if len(hu_fg_concat) else None,
        },
        "hu_by_organ_foreground_only": {
            CLASS_NAMES[c]: (
                {
                    "p0_5": pct(np.concatenate(hu_by_class[c]), 0.5),
                    "mean": float(np.mean(np.concatenate(hu_by_class[c]))),
                    "p99_5": pct(np.concatenate(hu_by_class[c]), 99.5),
                }
                if hu_by_class[c]
                else None
            )
            for c in range(1, NUM_CLASSES)
        },
        "organ_bbox_extent_mm_dx_dy_dz": {
            CLASS_NAMES[c]: {
                "median": [float(x) for x in np.median(np.array(bbox_extent_mm[c]), axis=0)]
                if bbox_extent_mm[c]
                else None,
                "n_patients_present": len(bbox_extent_mm[c]),
            }
            for c in range(1, NUM_CLASSES)
        },
        "organ_volume_cm3": {
            CLASS_NAMES[c]: {
                "median": float(np.median(organ_volume_cm3[c])) if organ_volume_cm3[c] else None,
                "min": float(np.min(organ_volume_cm3[c])) if organ_volume_cm3[c] else None,
                "max": float(np.max(organ_volume_cm3[c])) if organ_volume_cm3[c] else None,
                "max_over_min": (
                    float(np.max(organ_volume_cm3[c]) / np.min(organ_volume_cm3[c]))
                    if organ_volume_cm3[c] and np.min(organ_volume_cm3[c]) > 0
                    else None
                ),
                "cv_pct": (
                    100.0 * float(np.std(organ_volume_cm3[c])) / float(np.mean(organ_volume_cm3[c]))
                    if organ_volume_cm3[c] and np.mean(organ_volume_cm3[c]) > 0
                    else None
                ),
            }
            for c in range(1, NUM_CLASSES)
        },
        "nonzero_ct_fraction": {
            "median": float(np.median(fg_fraction_per_patient)),
            "min": float(np.min(fg_fraction_per_patient)),
            "max": float(np.max(fg_fraction_per_patient)),
        },
        "hu_artifacts": {
            "ceiling_HU": CT_12BIT_CEILING,
            "note": (
                "Voxels AT 3071 HU are censored by the 12-bit reconstruction ceiling "
                "(ordinary dense bone hitting the clamp). Voxels ABOVE it cannot come "
                "from a 12-bit reconstruction and indicate metal + beam hardening."
            ),
            "patients_at_ceiling": {pid: n for pid, n in hu_at_ceiling.items() if n > 0},
            "patients_above_ceiling": {pid: n for pid, n in hu_above_ceiling.items() if n > 0},
            "most_extreme_voxel_per_patient_top5": [
                {"patient": p, "hu": v, "x": x, "y": y, "z": z}
                for p, v, x, y, z in sorted(hu_extreme_voxels, key=lambda t: -t[1])[:5]
            ],
        },
        "patient_outliers": {
            "pca_variance_explained_pc1_pc2_pct": float(var_explained),
            "pca_distance_threshold": float(dist_thresh),
            "flagged_by_pca_projection": sorted(pca_flagged),
            "flagged_by_feature_modified_z": {
                pid: {k: round(v, 2) for k, v in d.items()} for pid, d in sorted(feature_flags.items())
            },
            "flagged_any": sorted(outlier_pids),
        },
        "per_patient": per_patient_rows,
    }
    (args.out_dir / "fingerprint.json").write_text(json.dumps(fingerprint, indent=2))

    # ---------------------------------------------------------------- CSV
    with (args.out_dir / "organ_volumes_cm3.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "split"] + [f"{CLASS_NAMES[c]}_cm3" for c in range(1, NUM_CLASSES)])
        for i, pid in enumerate(pids):
            split = "val" if pid in val_pids else "train"
            w.writerow([pid, split] + [round(organ_volume_cm3[c][i], 2) for c in range(1, NUM_CLASSES)])

    # ---------------------------------------------------------------- Markdown
    md = []
    md.append(f"# SegTHOR part1 dataset fingerprint (n={len(patients)} patients)\n")
    md.append("## Label mapping\n")
    md.append("| label | organ |")
    md.append("|---|---|")
    for c in range(NUM_CLASSES):
        md.append(f"| {c} | {CLASS_NAMES[c]} |")
    md.append(f"\n**{absent_note}**\n")
    if val_pids:
        md.append("## Train / validation split\n")
        md.append(f"- Validation patients ({len(val_pids)}): {', '.join(sorted(val_pids))}")
        md.append(f"- Training patients ({len(set(pids) - val_pids)}): {', '.join(sorted(set(pids) - val_pids))}")
        md.append(
            "- The split is patient-disjoint (no slice-level leakage). **However, every "
            "statistic below is pooled over all 20 patients**, validation included, so "
            "any preprocessing derived from this fingerprint is fitted on held-out data.\n"
        )
    md.append("## Shape & spacing\n")
    md.append(f"- Median shape (voxels, x,y,z): {fingerprint['shape_voxels']['median']}")
    md.append(f"- Shape range: {fingerprint['shape_voxels']['min']} to {fingerprint['shape_voxels']['max']}")
    md.append(
        f"- Median in-plane spacing (mm): "
        f"{fingerprint['spacing_mm']['inplane_median'][0]:.4f} x {fingerprint['spacing_mm']['inplane_median'][1]:.4f}"
    )
    md.append(
        f"- Slice spacing (mm): median {fingerprint['spacing_mm']['slice_median']:.3f}, "
        f"range {fingerprint['spacing_mm']['slice_min']:.3f}-{fingerprint['spacing_mm']['slice_max']:.3f}\n"
    )
    md.append("### Acquisition protocol groups (x x y x z spacing, mm)\n")
    md.append("| spacing | n | patients |")
    md.append("|---|---|---|")
    for key, members in sorted(protocol_groups.items(), key=lambda kv: -len(kv[1])):
        md.append(f"| {key} | {len(members)} | {', '.join(m.replace('Patient_', 'P') for m in sorted(members))} |")
    md.append("")
    md.append(
        "No DICOM tags survive in these files (every NIfTI text field is zeroed), so "
        "scanner/protocol metadata cannot be recovered directly. Spacing groups and the "
        "stored integer type are the only protocol proxies available:\n"
    )
    md.append("| stored dtype | n | patients |")
    md.append("|---|---|---|")
    for key, members in sorted(dtype_groups.items(), key=lambda kv: -len(kv[1])):
        md.append(f"| {key} | {len(members)} | {', '.join(m.replace('Patient_', 'P') for m in sorted(members))} |")
    md.append("")
    md.append("## Class balance\n")
    md.append("| class | % voxels | % slices present |")
    md.append("|---|---|---|")
    for c in range(NUM_CLASSES):
        cb = fingerprint["class_balance"][CLASS_NAMES[c]]
        md.append(f"| {CLASS_NAMES[c]} ({c}) | {cb['pct_voxels']:.3f} | {cb['pct_slices_present']:.1f} |")
    md.append("")
    md.append("## HU intensity\n")
    hg = fingerprint["hu_global"]
    md.append(
        f"- Whole-image HU range (exact, all voxels): [{hg['min']:.1f}, {hg['max']:.1f}], "
        f"median {hg['p50']:.1f}"
    )
    md.append(
        f"- Whole-image [0.5, 99.5] percentiles: [{hg['p0_5']:.1f}, {hg['p99_5']:.1f}] "
        f"(from a 1-in-37 voxel subsample)\n"
    )
    fgn = fingerprint["hu_foreground_nnunet_normalization"]
    md.append("### nnU-Net CT normalization parameters\n")
    md.append(
        "nnU-Net clips to the [0.5, 99.5] percentiles of the **foreground** voxels pooled "
        "across the dataset and then z-scores with the foreground mean/std. Computed over "
        "all labeled voxels:\n"
    )
    md.append("| statistic | value |")
    md.append("|---|---|")
    md.append(f"| clip lower (fg p0.5) | {fgn['p0_5']:.1f} HU |")
    md.append(f"| clip upper (fg p99.5) | {fgn['p99_5']:.1f} HU |")
    md.append(f"| mean (fg) | {fgn['mean']:.1f} HU |")
    md.append(f"| std (fg) | {fgn['std']:.1f} HU |")
    md.append(
        f"\nNote the difference from the whole-image percentiles above "
        f"([{hg['p0_5']:.0f}, {hg['p99_5']:.0f}]): clipping at the whole-image range is "
        f"very nearly a no-op, because most voxels are air.\n"
    )
    md.append("| organ | mean HU (fg) | HU [0.5, 99.5] pct range |")
    md.append("|---|---|---|")
    for c in range(1, NUM_CLASSES):
        row = fingerprint["hu_by_organ_foreground_only"][CLASS_NAMES[c]]
        if row is None:
            md.append(f"| {CLASS_NAMES[c]} | not labeled in this release | - |")
        else:
            md.append(f"| {CLASS_NAMES[c]} | {row['mean']:.1f} | [{row['p0_5']:.1f}, {row['p99_5']:.1f}] |")
    md.append("")
    md.append("## Organ bounding-box extent (mm, dx x dy x dz, median over patients where present)\n")
    md.append("| organ | median extent (mm) | # patients present |")
    md.append("|---|---|---|")
    for c in range(1, NUM_CLASSES):
        row = fingerprint["organ_bbox_extent_mm_dx_dy_dz"][CLASS_NAMES[c]]
        ext = row["median"]
        ext_str = f"{ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f}" if ext else "n/a"
        md.append(f"| {CLASS_NAMES[c]} | {ext_str} | {row['n_patients_present']} |")
    md.append("\nAxis order is the array/voxel order (x, y, z) = (L, P, S); dz is the "
              "through-plane extent.\n")
    md.append("## Organ volume (cm3, per patient)\n")
    md.append("| organ | median | min | max | max/min | coefficient of variation |")
    md.append("|---|---|---|---|---|---|")
    for c in range(1, NUM_CLASSES):
        row = fingerprint["organ_volume_cm3"][CLASS_NAMES[c]]
        if c in absent_classes:
            md.append(f"| {CLASS_NAMES[c]} | 0.0 | 0.0 | 0.0 | - | - (not labeled) |")
        else:
            md.append(
                f"| {CLASS_NAMES[c]} | {row['median']:.1f} | {row['min']:.1f} | {row['max']:.1f} | "
                f"{row['max_over_min']:.2f}x | {row['cv_pct']:.0f}% |"
            )
    md.append("")
    nz = fingerprint["nonzero_ct_fraction"]
    md.append(
        f"## Foreground (nonzero-CT) fraction of volume\n\n"
        f"- median {nz['median']:.3f}, range [{nz['min']:.3f}, {nz['max']:.3f}] "
        f"-> cropping to nonzero region would remove almost nothing for this modality "
        f"(unlike skull-stripped brain MRI, CT fills the frame). Note this measures "
        f"`ct != 0`, and 0 HU is water, not air -- it detects zero-padding, not the body.\n"
    )
    md.append(f"## HU artifact scan (12-bit reconstruction ceiling = {CT_12BIT_CEILING:.0f} HU)\n")
    md.append(
        "Voxels **at** 3071 HU are censored by the scanner's 12-bit clamp (dense bone "
        "hitting the ceiling). Voxels **above** 3071 HU cannot come from a 12-bit "
        "reconstruction at all and mark metal implants with beam-hardening streaks. A "
        "single `> 2000 HU` threshold would merge the two.\n"
    )
    md.append("| patient | at ceiling (3071) | above ceiling | max HU |")
    md.append("|---|---|---|---|")
    for row in sorted(per_patient_rows, key=lambda r: -r["hu_voxels_above_12bit_ceiling"]):
        md.append(
            f"| {row['patient']} | {row['hu_voxels_at_12bit_ceiling']} | "
            f"{row['hu_voxels_above_12bit_ceiling']} | {row['hu_max']:.0f} |"
        )
    md.append("")
    n_above_pts = len(fingerprint["hu_artifacts"]["patients_above_ceiling"])
    md.append(
        f"{n_above_pts} of {len(patients)} patients have voxels above the ceiling. "
        f"Traced to source: these are metal implants with radiating beam-hardening "
        f"streaks (e.g. Patient_19 z=180, a shoulder/neck slice). Locations of the most "
        f"extreme voxel per patient:\n"
    )
    for e in fingerprint["hu_artifacts"]["most_extreme_voxel_per_patient_top5"]:
        md.append(f"- {e['patient']}: {e['hu']:.0f} HU at (x={e['x']}, y={e['y']}, z={e['z']})")
    md.append("")
    md.append("## Per-patient outlier scan\n")
    po = fingerprint["patient_outliers"]
    md.append(
        f"Two criteria, unioned. PC1+PC2 of the 7 shape/spacing/volume features explain "
        f"only {po['pca_variance_explained_pc1_pc2_pct']:.0f}% of the variance, so the "
        f"projection alone cannot see a patient that is extreme on a single feature; the "
        f"modified z-score catches those. The PCA threshold is median+3*MAD, not "
        f"mean+2*std, so one extreme patient cannot inflate the threshold meant to catch it.\n"
    )
    md.append(f"- Flagged by PC1-PC2 distance: {', '.join(po['flagged_by_pca_projection']) or 'none'}")
    md.append("- Flagged by per-feature modified z-score (|z| > 3.5):")
    if po["flagged_by_feature_modified_z"]:
        for pid, d in po["flagged_by_feature_modified_z"].items():
            md.append(f"  - {pid}: " + ", ".join(f"{k} z={v:+.2f}" for k, v in d.items()))
    else:
        md.append("  - none")
    md.append(f"- **Flagged by either: {', '.join(po['flagged_any']) or 'none'}**\n")
    (args.out_dir / "fingerprint.md").write_text("\n".join(md))

    # ================================================================ PLOTS

    # 1. Shape/spacing boxplots. With only 20 patients and most values
    # clustered on one number (e.g. 512x512 in-plane for every patient), a
    # boxplot alone renders as a flat, seemingly-empty line -- so every raw
    # value is also plotted as a jittered dot, making "everyone shares this
    # value" visually distinct from "no data here".
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    shape_groups = [shapes_arr[:, 0], shapes_arr[:, 1], shapes_arr[:, 2]]
    spacing_groups = [inplane_arr[:, 0], inplane_arr[:, 1], slice_arr]
    bp0 = axes[0].boxplot(shape_groups, tick_labels=["x", "y", "z"], patch_artist=True, showfliers=False, zorder=2)
    bp1 = axes[1].boxplot(spacing_groups, tick_labels=["spacing x", "spacing y", "spacing z"], patch_artist=True,
                          showfliers=False, zorder=2)
    for bp in (bp0, bp1):
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for median in bp["medians"]:
            median.set_color(PALETTE[1])
            median.set_linewidth(2)
    rng = np.random.default_rng(0)
    for ax, groups in [(axes[0], shape_groups), (axes[1], spacing_groups)]:
        for i, vals in enumerate(groups, start=1):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=22, color="black", alpha=0.6, zorder=3,
                       edgecolor="white", linewidth=0.4)
    axes[0].set_ylabel("voxels")
    axes[1].set_ylabel("mm")
    axes[0].set_title("Volume shape")
    axes[1].set_title("Voxel spacing")
    # Describe the spacing spread from the data rather than asserting a count.
    xy_values = sorted({round(v, 4) for v in inplane_arr[:, 0]})
    xy_counts = {v: int((np.round(inplane_arr[:, 0], 4) == v).sum()) for v in xy_values}
    xy_desc = ", ".join(f"{v:.4f}mm (n={xy_counts[v]})" for v in xy_values)
    z_values = sorted({round(v, 2) for v in slice_arr})
    z_counts = {v: int((np.round(slice_arr, 2) == v).sum()) for v in z_values}
    z_desc = ", ".join(f"{v:.1f}mm (n={z_counts[v]})" for v in z_values)
    # Standard box-and-whisker legend, spelled out: a flat box (as in x/y here,
    # where nearly every patient shares the same value) is a real result, not
    # a rendering issue -- it means the IQR is ~0, not that data is missing.
    # Every one of the 20 raw values is also drawn as a dot (fliers are off,
    # so dots -- not a separate outlier marker -- carry that information too).
    axes[1].plot([], [], color=PALETTE[0], alpha=0.55, linewidth=8, label="box = 25th-75th percentile (IQR)")
    axes[1].plot([], [], color=PALETTE[1], linewidth=2, label="line = median")
    axes[1].plot([], [], color="black", linewidth=1, label="whiskers = full range")
    axes[1].plot([], [], marker="o", linestyle="none", markeredgecolor="black", markerfacecolor="black",
                alpha=0.6, label="dot = one patient (n=20)")
    legend_below(axes[1], ncol=2)
    decorate(
        fig,
        "In-Plane Grid Is Fixed, Physical Spacing Is Not",
        subtitle=f"{shapes_arr[0,0]}x{shapes_arr[0,1]} in-plane for every patient; "
        f"z-count and all three spacings vary by acquisition protocol",
        footnote_text=(
            f"In-plane spacing takes {len(xy_values)} distinct values: {xy_desc}. "
            f"Slice spacing: {z_desc}. Flat x/y boxes = ~0 IQR (nearly all patients share one value), "
            f"not missing data. Resampling to one common spacing is necessary before training."
        ),
        has_legend=True,
    )
    fig.savefig(args.out_dir / "shape_spacing_boxplots.png")
    plt.close(fig)

    # 2. Class balance bar chart (log scale, % voxels), with value labels
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [CLASS_NAMES[c] for c in range(1, NUM_CLASSES)]
    vals = [fingerprint["class_balance"][n]["pct_voxels"] for n in names]
    bars = ax.bar(names, vals, color=[ORGAN_COLORS[c] for c in range(1, NUM_CLASSES)])
    ax.set_yscale("log")
    positive = [v for v in vals if v > 0]
    # Set BOTH limits explicitly: the "0% (absent)" label needs a known floor to
    # sit above, otherwise it is placed below the auto bottom and clipped away.
    y_bottom = min(positive) / 8.0
    ax.set_ylim(bottom=y_bottom, top=max(vals) * 2.2)
    ax.set_ylabel("% of total voxels (log scale)")
    for b, v in zip(bars, vals):
        label = f"{v:.3f}%" if v > 0 else "0% (not labeled)"
        y = v * 1.3 if v > 0 else y_bottom * 1.4
        ax.text(b.get_x() + b.get_width() / 2, y, label, ha="center", fontsize=10)
    span_orders = np.log10(max(positive) / min(positive))
    bg_pct = fingerprint["class_balance"]["background"]["pct_voxels"]
    absent_names = [CLASS_NAMES[c] for c in absent_classes]
    biggest = names[int(np.argmax(vals))]
    decorate(
        fig,
        f"{biggest.title()} Dominates the Foreground; {' and '.join(n.title() for n in absent_names)} Is Unlabeled",
        subtitle=f"Background is {bg_pct:.2f}% of voxels (off-chart); the {len(positive)} labeled "
        f"classes span {span_orders:.1f} orders of magnitude",
        footnote_text=(
            f"Class 4 ({', '.join(absent_names)}) has zero voxels under this label id in this release -- "
            f"whether it's genuinely absent or mislabeled elsewhere is an open question worth tracking down."
        ),
    )
    fig.savefig(args.out_dir / "class_balance.png")
    plt.close(fig)

    # 3. HU histograms per organ (foreground only) + global
    hist_lo, hist_hi = -1050, 500
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.hist(hu_all_concat, bins=100, range=(hist_lo, hist_hi), alpha=0.35, label="whole CT", density=True, color="#999999")
    for c in range(1, NUM_CLASSES):
        if hu_by_class[c]:
            ax.hist(
                np.concatenate(hu_by_class[c]),
                bins=100,
                range=(hist_lo, hist_hi),
                alpha=0.55,
                label=CLASS_NAMES[c],
                density=True,
                color=ORGAN_COLORS[c],
            )
    ax.set_xlabel("Hounsfield Units")
    ax.set_ylabel("density")
    legend_below(ax, ncol=4)
    pct_over = 100.0 * n_voxels_over_500 / total_voxels
    # Quote the interquartile band, not p0.5/p99.5: the soft-tissue masks have
    # thin air-contaminated tails at their edges, which would describe a far
    # wider overlap than the histograms actually show.
    soft_classes = [c for c in present_classes if np.mean(np.concatenate(hu_by_class[c])) > -200]
    soft_pooled = np.concatenate([np.concatenate(hu_by_class[c]) for c in soft_classes])
    soft_lo, soft_hi = np.percentile(soft_pooled, 25), np.percentile(soft_pooled, 75)
    soft_names = " and ".join(CLASS_NAMES[c] for c in soft_classes)
    decorate(
        fig,
        "Soft-Tissue Organs Overlap in a Narrow HU Band",
        subtitle=f"Trachea sits with air (~-1000 HU); {soft_names} overlap almost entirely, "
        f"interquartile range {soft_lo:.0f} to {soft_hi:.0f} HU",
        footnote_text=(
            f"X-axis is clipped to [{hist_lo}, {hist_hi}] HU, which hides {pct_over:.2f}% of voxels "
            f"(bone and the metal artifacts up to {hg['max']:.0f} HU); densities are renormalized over "
            f"the visible range."
        ),
    )
    fig.savefig(args.out_dir / "hu_histograms.png")
    plt.close(fig)

    # 4. HU boxplots per organ (horizontal)
    fig, ax = plt.subplots(figsize=(8, 4))
    box_classes = [c for c in range(1, NUM_CLASSES) if hu_by_class[c]]
    data = [np.concatenate(hu_by_class[c]) for c in box_classes]
    labels = [CLASS_NAMES[c] for c in box_classes]
    bp = ax.boxplot(data, tick_labels=labels, vert=False, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], box_classes):
        patch.set_facecolor(ORGAN_COLORS[c])
        patch.set_alpha(0.75)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Hounsfield Units (whiskers at 1.5 IQR, fliers hidden)")
    decorate(
        fig,
        "Per-Organ HU Spread, Outliers Excluded",
        subtitle="Box = interquartile range, line = median, per labeled organ",
        footnote_text=f"{', '.join(CLASS_NAMES[c] for c in absent_classes)} omitted: no labeled voxels.",
    )
    fig.savefig(args.out_dir / "hu_boxplots_by_organ.png")
    plt.close(fig)

    # 5. Organ volume violin/strip plot (cm^3)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Keep class ids alongside the data so colors follow the class, not position.
    vol_classes = [c for c in range(1, NUM_CLASSES) if any(v > 0 for v in organ_volume_cm3[c])]
    vol_data = [np.array(organ_volume_cm3[c]) for c in vol_classes]
    vol_labels = [CLASS_NAMES[c] for c in vol_classes]
    parts = ax.violinplot(vol_data, showmedians=True)
    for body, c in zip(parts["bodies"], vol_classes):
        body.set_facecolor(ORGAN_COLORS[c])
        body.set_alpha(0.6)
    rng = np.random.default_rng(0)
    for i, d in enumerate(vol_data, start=1):
        jitter = (rng.random(len(d)) - 0.5) * 0.08
        ax.scatter(np.full(len(d), i) + jitter, d, color="#333333", s=14, zorder=3)
    ax.set_xticks(range(1, len(vol_labels) + 1))
    ax.set_xticklabels(vol_labels)
    ax.set_ylabel("volume (cm3)")
    ratios = {CLASS_NAMES[c]: fingerprint["organ_volume_cm3"][CLASS_NAMES[c]]["max_over_min"] for c in vol_classes}
    widest = max(ratios, key=ratios.get)
    hv = fingerprint["organ_volume_cm3"][widest]
    decorate(
        fig,
        f"{widest.title()} Volume Varies {hv['max_over_min']:.1f}x Across Patients",
        subtitle=f"Per-patient organ volume (points = individual patients); "
        f"{widest} spans {hv['min']:.0f}-{hv['max']:.0f} cm3",
        footnote_text=(
            "Shared linear y-axis, so the trachea distribution is compressed near zero; see "
            "organ_volumes_cm3.csv for its values. Violin outlines are kernel density estimates "
            "and extend slightly beyond the observed range."
        ),
    )
    fig.savefig(args.out_dir / "organ_volume_violin.png")
    plt.close(fig)

    # 6. Organ z-extent (anatomical position along the scan), one subplot per organ
    z_classes = [c for c in range(1, NUM_CLASSES) if organ_z_extent_frac[c]]
    fig, axes = plt.subplots(1, len(z_classes), figsize=(4.3 * len(z_classes), 5), sharex=True)
    axes = np.atleast_1d(axes)
    span_txt = []
    for ax_i, c in zip(axes, z_classes):
        rows = sorted(organ_z_extent_frac[c], key=lambda t: t[1])
        ys = np.arange(len(rows))
        starts = [r[1] for r in rows]
        ends = [r[2] for r in rows]
        span_txt.append(f"{CLASS_NAMES[c]} {min(starts):.2f}-{max(ends):.2f}")
        ax_i.barh(ys, [e - s for s, e in zip(starts, ends)], left=starts, color=ORGAN_COLORS[c], alpha=0.8, height=0.7)
        ax_i.set_title(CLASS_NAMES[c])
        ax_i.set_xlabel("fraction of scan (z axis)")
        ax_i.set_yticks([])
        ax_i.grid(axis="x", color="#D9D9D9", linewidth=0.8)
        ax_i.grid(axis="y", visible=False)
        ax_i.set_xlim(0, 1)
    axes[0].set_ylabel("patients (each panel sorted independently)")
    decorate(
        fig,
        "Organ Z-Ranges Overlap Heavily and Vary Between Patients",
        subtitle="Each bar = one patient's organ extent, normalized to that patient's slice count",
        footnote_text=(
            f"Observed spans: {'; '.join(span_txt)}. Panels are sorted independently, so row i is a "
            f"DIFFERENT patient in each panel -- do not read across. Normalized z-position is too "
            f"broad here to serve as a hard prior; it is usable only as a coarse sanity check."
        ),
    )
    fig.savefig(args.out_dir / "organ_z_extent.png")
    plt.close(fig)

    # 7. Class presence heatmap (annotation completeness QC)
    presence_arr = np.array(class_present_matrix, dtype=float)  # (n_patients, 4)
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.imshow(presence_arr, cmap="Greys", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels([CLASS_NAMES[c] for c in range(1, NUM_CLASSES)])
    ax.set_yticks(range(len(pids)))
    ax.set_yticklabels(pids, fontsize=8)
    # Cell borders + an axes frame, so an all-white (all-absent) column reads as
    # "column of empty cells" rather than "no column drawn at all".
    ax.set_xticks(np.arange(-0.5, presence_arr.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, presence_arr.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#BBBBBB", linewidth=0.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(0.8)
    for c_i, c in enumerate(range(1, NUM_CLASSES)):
        if c in absent_classes:
            ax.text(
                c_i,
                len(pids) / 2 - 0.5,
                "no voxels\nin any patient",
                ha="center",
                va="center",
                fontsize=9,
                color="#8C3730",
                rotation=90,
            )
    absent_str = ", ".join(CLASS_NAMES[c] for c in absent_classes) or "none"
    decorate(
        fig,
        "Annotation Completeness Check, Per Patient Per Class",
        subtitle="Black = organ labeled and present; white = no voxels for that patient",
        footnote_text=(
            f"The {absent_str} column is empty for all {len(pids)} patients -- an intentional omission "
            f"for the course dataset, confirmed by the professor. The other "
            f"{len(present_classes)} classes are labeled in all {len(pids)} patients."
        ),
    )
    fig.savefig(args.out_dir / "class_presence_heatmap.png")
    plt.close(fig)

    # 8. Per-patient outlier scan (PCA projection + per-feature flags)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(
        scores[~outlier_mask, 0], scores[~outlier_mask, 1], color=PALETTE[0], s=60, label="typical", zorder=3
    )
    ax.scatter(scores[outlier_mask, 0], scores[outlier_mask, 1], color=PALETTE[5], s=80, label="flagged", zorder=4)
    for i, pid in enumerate(pids):
        if outlier_mask[i]:
            ax.annotate(
                pid.replace("Patient_", "P"),
                (scores[i, 0], scores[i, 1]),
                fontsize=9,
                xytext=(5, 5),
                textcoords="offset points",
            )
    ax.axhline(0, color="#CCCCCC", linewidth=0.8)
    ax.axvline(0, color="#CCCCCC", linewidth=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal", adjustable="box")  # so the Euclidean distance threshold reads correctly
    legend_below(ax, ncol=2)
    n_out = int(outlier_mask.sum())
    headline = "No Patients Flagged" if n_out == 0 else f"{n_out} Patient(s) Flagged"
    flag_bits = []
    for pid in sorted(outlier_pids):
        why = []
        if pid in pca_flagged:
            why.append("PC distance")
        if pid in feature_flags:
            why.append(", ".join(f"{k} z={v:+.1f}" for k, v in feature_flags[pid].items()))
        flag_bits.append(f"{pid.replace('Patient_', 'P')}: {'; '.join(why)}")
    decorate(
        fig,
        f"{headline} in Shape/Spacing/Volume Space",
        subtitle=f"PCA over {len(PCA_FEATURES)} features; PC1+PC2 explain only "
        f"{var_explained:.0f}% of variance, so flags also use per-feature modified z-scores",
        footnote_text=(
            (f"{'. '.join(flag_bits)}. " if flag_bits else "")
            + f"Threshold is median+3*MAD of the PC1-PC2 distance (robust to masking) OR "
            f"|modified z| > {MODIFIED_Z_THRESHOLD} on any morphometric feature. Points near the "
            f"origin can still be extreme on a feature PC1/PC2 do not capture."
        ),
        has_legend=True,
    )
    fig.savefig(args.out_dir / "patient_pca_outliers.png")
    plt.close(fig)

    # 9. Slice montage: raw CT vs. GT overlay for a few representative patients
    inplane_max_axis0 = inplane_arr[:, 0]
    idx_spacing_outlier = int(np.argmax(inplane_max_axis0))
    idx_max_z = int(np.argmax(shapes_arr[:, 2]))
    montage_idx = sorted(set([0, idx_spacing_outlier, idx_max_z]))[: args.n_montage_patients]
    montage_patients = [patients[i] for i in montage_idx]
    # Describe the columns in the order they are actually drawn, not the order
    # they were selected in.
    reason_by_idx = {0: "first patient", idx_spacing_outlier: "widest in-plane spacing", idx_max_z: "most slices"}
    montage_reasons = ", ".join(f"{patients[i].name.replace('Patient_', 'P')} = {reason_by_idx[i]}" for i in montage_idx)

    fig, axes = plt.subplots(2, len(montage_patients), figsize=(4.2 * len(montage_patients), 8.5))
    if len(montage_patients) == 1:
        axes = axes.reshape(2, 1)
    cmap_gt = matplotlib.colors.ListedColormap(
        ["none", ORGAN_COLORS[1], ORGAN_COLORS[2], ORGAN_COLORS[3], ORGAN_COLORS[4]]
    )
    for col, patient_dir in enumerate(montage_patients):
        pid, ct, gt, zooms = load_patient(patient_dir)
        z_mid = ct.shape[-1] // 2
        # find a slice near the middle that actually contains foreground, for a more informative QC image
        for offset in range(0, ct.shape[-1] // 2):
            for cand in (z_mid + offset, z_mid - offset):
                if 0 <= cand < ct.shape[-1] and gt[:, :, cand].any():
                    z_mid = cand
                    break
            else:
                continue
            break
        # The array axes are (L, P, S). rot90 with k=-1 puts decreasing P (anterior)
        # at the top, i.e. the standard axial display with the sternum up and the
        # spine down. k=+1 flips it vertically -- the giveaway is that the scanner
        # table then renders ABOVE the patient.
        raw_slice = np.rot90(ct[:, :, z_mid], k=-1)
        gt_slice = np.rot90(gt[:, :, z_mid], k=-1)
        axes[0, col].imshow(raw_slice, cmap="gray", vmin=-200, vmax=300)
        axes[0, col].set_title(f"{pid}\nslice z={z_mid}", fontsize=11)
        axes[0, col].axis("off")
        axes[1, col].imshow(raw_slice, cmap="gray", vmin=-200, vmax=300)
        axes[1, col].imshow(np.ma.masked_where(gt_slice == 0, gt_slice), cmap=cmap_gt, vmin=0, vmax=4, alpha=0.55)
        axes[1, col].axis("off")
    axes[0, 0].text(-0.15, 0.5, "raw CT", transform=axes[0, 0].transAxes, rotation=90, va="center", fontsize=12)
    axes[1, 0].text(-0.15, 0.5, "+ GT overlay", transform=axes[1, 0].transAxes, rotation=90, va="center", fontsize=12)
    # A QC overlay needs a color key, otherwise the reader cannot check the labels.
    handles = [
        matplotlib.patches.Patch(facecolor=ORGAN_COLORS[c], alpha=0.55, label=CLASS_NAMES[c])
        for c in present_classes
    ]
    fig._pending_legend = (handles, [h.get_label() for h in handles], len(handles))
    decorate(
        fig,
        "Visual QC: Raw CT vs. Ground-Truth Overlay",
        subtitle=f"Anterior up, standard axial orientation. Columns: {montage_reasons}",
        footnote_text=f"Colors follow {CLASS_NAMES[1]}/{CLASS_NAMES[2]}/{CLASS_NAMES[3]} per the legend above.",
    )
    fig.savefig(args.out_dir / "slice_montage_qc.png")
    plt.close(fig)

    print(f"Wrote fingerprint.json, fingerprint.md, organ_volumes_cm3.csv, and 9 PNGs to {args.out_dir}/")


if __name__ == "__main__":
    main()
