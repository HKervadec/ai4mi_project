#!/usr/bin/env python3

"""
Data integrity audit for the SegTHOR project.

Checks that the data is correct *before* spending compute on training. It runs
two independent passes:

  --raw_dir     : the 3D NIfTI source (e.g. data/segthor_part1/train), one folder
                  per patient containing <ID>.nii.gz (CT) and GT.nii.gz (labels).
                  A patient with a second ground truth (GT2.nii.gz) is treated as
                  a *reference pair* and used to characterise any corruption.

  --sliced_dir  : the 2D sliced dataset used by the dataloader (e.g. data/SEGTHOR),
                  with {train,val}/{img,gt}/*.png.

Findings are reported at two severities:
  [ERROR] blocks correct training/evaluation (missing class, shape mismatch, leakage).
  [WARN ] does not break the training loop but breaks metrics/stitching or hints at
          corruption (affine mismatch, inflated label, spacing out of range).

Exit code is non-zero if any ERROR is found, so it can gate a Makefile / CI step:
    python audit_data.py --raw_dir data/segthor_part1/train --sliced_dir data/SEGTHOR

The audit is read-only: it never modifies the data.
"""

import re
import sys
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image


# SegTHOR label convention. Index = integer label in the raw GT.
CLASS_NAMES: list[str] = ["background", "esophagus", "heart", "trachea", "aorta"]
K: int = len(CLASS_NAMES)
# gt PNGs store class k as k * SLICE_SCALE (see slice_segthor.py / gt_transform).
SLICE_SCALE: int = 63
PATIENT_RE = re.compile(r"(Patient_\d+)")


class Report:
    """Collects findings so the audit reports *all* problems rather than crashing
    on the first one (assertions would stop at the first failure)."""

    def __init__(self) -> None:
        self.errors: int = 0
        self.warns: int = 0

    def ok(self, msg: str) -> None:
        print(f"  [ ok ] {msg}")

    def warn(self, msg: str) -> None:
        self.warns += 1
        print(f"  [WARN] {msg}")

    def error(self, msg: str) -> None:
        self.errors += 1
        print(f"  [ERROR] {msg}")

    def section(self, title: str) -> None:
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def class_counts(arr: np.ndarray) -> dict[int, int]:
    """Voxel count per label 0..K-1 (labels outside the range are ignored here;
    they are reported separately by the label-set check)."""
    counts = np.bincount(arr.ravel(), minlength=K)
    return {k: int(counts[k]) for k in range(K)}


# --------------------------------------------------------------------------- #
# Raw 3D checks                                                               #
# --------------------------------------------------------------------------- #
def audit_raw(raw_dir: Path, rep: Report) -> None:
    rep.section(f"RAW 3D data: {raw_dir}")
    patient_dirs = sorted(p for p in raw_dir.glob("Patient_*") if p.is_dir())
    if not patient_dirs:
        rep.error(f"No Patient_* folders found under {raw_dir}")
        return
    print(f"Found {len(patient_dirs)} patients.\n")

    class1_sizes: dict[str, int] = {}
    reference_pairs: list[Path] = []

    for pdir in patient_dirs:
        pid = pdir.name
        ct_path = pdir / f"{pid}.nii.gz"
        gt_path = pdir / "GT.nii.gz"
        print(f"- {pid}")

        if not ct_path.exists():
            rep.error(f"{pid}: missing CT {ct_path.name}")
            continue
        if not gt_path.exists():
            rep.error(f"{pid}: missing GT.nii.gz")
            continue

        ct_obj = nib.load(str(ct_path))          # header only, no pixel load
        gt_obj = nib.load(str(gt_path))
        gt = np.asarray(gt_obj.dataobj)

        # 1) shape: GT must be voxel-aligned with the CT
        if ct_obj.shape != gt_obj.shape:
            rep.error(f"{pid}: GT shape {gt_obj.shape} != CT shape {ct_obj.shape}")
        else:
            rep.ok(f"{pid}: shape {gt_obj.shape}")

        # 2) affine / orientation: needed for stitching + 3D metric spacing
        if not np.allclose(ct_obj.affine, gt_obj.affine):
            ct_ax = "".join(nib.aff2axcodes(ct_obj.affine))
            gt_ax = "".join(nib.aff2axcodes(gt_obj.affine))
            rep.warn(f"{pid}: GT affine != CT affine (CT {ct_ax} / GT {gt_ax}); "
                     f"breaks 3D metrics + stitching until re-stamped with the CT affine")

        # 3) spacing sanity (SegTHOR: ~1mm in-plane, 2-3.7mm through-plane)
        dx, dy, dz = ct_obj.header.get_zooms()[:3]
        if not (0.8 <= dx <= 1.4 and 0.8 <= dy <= 1.4 and 1.5 <= dz <= 4.0):
            rep.warn(f"{pid}: unusual CT spacing ({dx:.2f}, {dy:.2f}, {dz:.2f}) mm")

        # 4) label set: must be exactly {0..K-1}
        labels = set(int(v) for v in np.unique(gt))
        unexpected = labels - set(range(K))
        missing = set(range(K)) - labels
        if unexpected:
            rep.error(f"{pid}: unexpected label values {sorted(unexpected)} "
                      f"(expected subset of {list(range(K))})")
        if missing:
            names = ", ".join(f"{m}={CLASS_NAMES[m]}" for m in sorted(missing))
            rep.error(f"{pid}: labels absent from GT: {names}")

        # 5) per-class sizes; remember esophagus for the inflation cross-check
        counts = class_counts(gt)
        class1_sizes[pid] = counts[1]

        # collect reference pairs (a second ground truth exists)
        if (pdir / "GT2.nii.gz").exists():
            reference_pairs.append(pdir)

    # 6) inflated-esophagus cross-check: a label-1 region far above the cohort
    #    median means another organ (the aorta) has been merged into it.
    if class1_sizes:
        vals = np.array(list(class1_sizes.values()))
        median = float(np.median(vals))
        for pid, size in class1_sizes.items():
            if median > 0 and size > 2.0 * median:
                rep.warn(f"{pid}: esophagus label is {size} voxels "
                         f"({size / median:.1f}x cohort median {median:.0f}) "
                         f"-> likely aorta merged into esophagus")

    for pdir in reference_pairs:
        audit_reference_pair(pdir, rep)


def audit_reference_pair(pdir: Path, rep: Report) -> None:
    """Compare GT.nii.gz against GT2.nii.gz for a patient that has both, and decide
    which is the trustworthy label (full class set + affine matching the CT)."""
    pid = pdir.name
    rep.section(f"REFERENCE PAIR: {pid} (GT.nii.gz vs GT2.nii.gz)")
    ct = nib.load(str(pdir / f"{pid}.nii.gz"))
    o1 = nib.load(str(pdir / "GT.nii.gz"))
    o2 = nib.load(str(pdir / "GT2.nii.gz"))
    g1 = np.asarray(o1.dataobj)
    g2 = np.asarray(o2.dataobj)

    l1 = sorted(int(v) for v in np.unique(g1))
    l2 = sorted(int(v) for v in np.unique(g2))
    print(f"  GT  labels {l1}, affine==CT: {np.allclose(o1.affine, ct.affine)}")
    print(f"  GT2 labels {l2}, affine==CT: {np.allclose(o2.affine, ct.affine)}")

    if g1.shape != g2.shape:
        rep.error(f"{pid}: GT and GT2 have different shapes {g1.shape} vs {g2.shape}")
        return

    print("  Per-class voxel counts and disagreement (GT vs GT2):")
    for k in range(K):
        a, b = (g1 == k), (g2 == k)
        disagree = int((a != b).sum())
        print(f"    {k}={CLASS_NAMES[k]:<10} GT={int(a.sum()):>8} "
              f"GT2={int(b.sum()):>8} disagree={disagree:>8}")

    # Which file is correct? Full label set + affine matching the CT.
    def score(obj, arr):
        return (set(range(K)).issubset(np.unique(arr)), np.allclose(obj.affine, ct.affine))
    ok1, ok2 = score(o1, g1), score(o2, g2)
    correct = "GT2.nii.gz" if ok2 == (True, True) and ok1 != (True, True) else (
        "GT.nii.gz" if ok1 == (True, True) else "unclear")
    if correct == "unclear":
        rep.warn(f"{pid}: could not unambiguously pick the correct label; inspect manually")
    else:
        rep.ok(f"{pid}: '{correct}' is the trustworthy ground truth "
               f"(full class set + CT-matching affine); use it as the reference/fix key")


# --------------------------------------------------------------------------- #
# Sliced 2D checks                                                            #
# --------------------------------------------------------------------------- #
def audit_sliced(sliced_dir: Path, rep: Report) -> None:
    rep.section(f"SLICED 2D data: {sliced_dir}")
    valid_gt = {k * SLICE_SCALE for k in range(K)}
    split_patients: dict[str, set[str]] = {}

    for split in ("train", "val"):
        img_dir = sliced_dir / split / "img"
        gt_dir = sliced_dir / split / "gt"
        print(f"\n- split '{split}'")
        if not img_dir.is_dir():
            rep.warn(f"{split}: no img folder ({img_dir}), skipping")
            continue

        imgs = sorted(img_dir.glob("*.png"))
        gts = sorted(gt_dir.glob("*.png")) if gt_dir.is_dir() else []
        img_stems = {p.stem for p in imgs}
        gt_stems = {p.stem for p in gts}
        print(f"  {len(imgs)} images, {len(gts)} labels")

        # 1) every image has a matching label and vice versa
        only_img = img_stems - gt_stems
        only_gt = gt_stems - img_stems
        if only_img:
            rep.error(f"{split}: {len(only_img)} images without a matching gt "
                      f"(e.g. {sorted(only_img)[:3]})")
        if only_gt:
            rep.error(f"{split}: {len(only_gt)} gts without a matching image "
                      f"(e.g. {sorted(only_gt)[:3]})")
        if imgs and not only_img and not only_gt:
            rep.ok(f"{split}: img/gt pairing consistent")

        # 2) scan pixel values: img in [0,255], gt in the valid class set, and
        #    record which classes actually appear + image sizes + patient ids.
        seen_gt_vals: set[int] = set()
        sizes: set[tuple[int, int]] = set()
        img_range_bad = False
        for p in imgs:
            a = np.asarray(Image.open(p))
            sizes.add(a.shape[:2])
            if a.min() < 0 or a.max() > 255:
                img_range_bad = True
            split_patients.setdefault(split, set())
            m = PATIENT_RE.match(p.stem)
            if m:
                split_patients[split].add(m.group(1))
        for p in gts:
            seen_gt_vals |= set(int(v) for v in np.unique(np.asarray(Image.open(p))))

        if img_range_bad:
            rep.error(f"{split}: image pixels outside [0,255]")
        if len(sizes) > 1:
            rep.warn(f"{split}: mixed image sizes {sorted(sizes)}")
        elif sizes:
            rep.ok(f"{split}: uniform image size {sizes.pop()}")

        bad_vals = seen_gt_vals - valid_gt
        if bad_vals:
            rep.error(f"{split}: gt has values {sorted(bad_vals)} outside "
                      f"the valid class set {sorted(valid_gt)}")

        present = sorted(v // SLICE_SCALE for v in seen_gt_vals if v in valid_gt)
        absent = [k for k in range(K) if k not in present]
        if absent:
            names = ", ".join(f"{k}={CLASS_NAMES[k]}" for k in absent)
            rep.error(f"{split}: classes never appear in labels: {names} "
                      f"-> the network can never learn them")
        else:
            rep.ok(f"{split}: all {K} classes present in labels")

    # 3) no patient in both train and val (would leak into validation)
    if "train" in split_patients and "val" in split_patients:
        overlap = split_patients["train"] & split_patients["val"]
        if overlap:
            rep.error(f"train/val leakage: {len(overlap)} patients in both splits "
                      f"({sorted(overlap)})")
        else:
            rep.ok(f"train/val split is patient-disjoint "
                   f"({len(split_patients['train'])} train / {len(split_patients['val'])} val patients)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", type=Path, default=None,
                        help="3D source, e.g. data/segthor_part1/train")
    parser.add_argument("--sliced_dir", type=Path, default=None,
                        help="2D sliced dataset root, e.g. data/SEGTHOR")
    args = parser.parse_args()

    if not args.raw_dir and not args.sliced_dir:
        parser.error("provide at least one of --raw_dir / --sliced_dir")

    rep = Report()
    if args.raw_dir:
        audit_raw(args.raw_dir, rep)
    if args.sliced_dir:
        audit_sliced(args.sliced_dir, rep)

    rep.section("SUMMARY")
    print(f"  errors: {rep.errors}   warnings: {rep.warns}")
    if rep.errors:
        print("  VERDICT: data is NOT ready for training — fix the ERRORs above.")
    elif rep.warns:
        print("  VERDICT: trainable, but WARNINGs will affect metrics/stitching — review them.")
    else:
        print("  VERDICT: all checks passed.")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
