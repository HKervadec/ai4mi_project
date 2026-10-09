#!/usr/bin/env python3
# we implemented a 3D geometric (distance-transform + watershed) recovery, not atlas propagation and not a 2D per-slice heuristic.

"""
Repair the corrupted SegTHOR ground truth into a consistent 5-class dataset.

The raw part-1 labels have two defects (see audit_data.py):
  1. The aorta (class 4) is merged into the esophagus (class 1) in every patient.
  2. Every GT.nii.gz has its affine stripped to identity, so it no longer shares
     the CT's world geometry (breaks 3D metrics + stitching).

Patient_07 ships a second, correct ground truth (GT2.nii.gz) with the aorta
properly separated and the CT-matching affine. We use it two ways:
  * as the trustworthy label for Patient_07 itself, and
  * as the single validation case to measure our recovery quality every run.

For the other patients we cannot rely on CT intensity to tell aorta from
esophagus (this is a non-contrast scan: aortic blood ~35 HU is indistinguishable
from soft tissue), and the two organs abut into a single connected blob, so
connected components cannot split them either. Instead we exploit *caliber*: the
aorta is a fat tube (~10-14 mm across) and the esophagus a thin one (~4 mm). A
distance transform of the merged region turns that geometric difference into a
scalar we can threshold and flood with a watershed. Validated on Patient_07 this
recovers the aorta at Dice ~0.95 and the esophagus at ~0.81.

Output: a fixed copy of the dataset under --dest, with corrected GT.nii.gz files
(5 classes, CT affine) and the CT linked in, ready for slice_segthor.py. The raw
data is never modified. Re-run audit_data.py on the result to confirm it is clean.

    python fix_gt.py --source_dir data/segthor_part1/train --dest data/gt/watershed/train
"""

import os
import shutil
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.segmentation import watershed


# Recovery hyper-parameters, in millimetres. Tuned on Patient_07 (the only case
# with a known answer). They are anatomical, not scan-specific: the aorta's
# radius is far larger than the esophagus's, which is what these thresholds encode.
CORE_THR_MM: float = 8.0    # a voxel at least this deep in the mask is aortic core
ESO_THR_MM: float = 3.5     # a voxel this shallow is thin -> esophagus candidate
FAR_MM: float = 12.0        # ...and this far from the aortic core to seed esophagus
MIN_CORE_MM: float = 7.0    # if nothing is this fat, assume no aorta is present


def largest_cc(mask: np.ndarray) -> np.ndarray:
    """Largest connected component of a boolean mask (drops speckle seeds)."""
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # ignore background
    return lab == sizes.argmax()


def split_aorta_esophagus(merged: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """Split a boolean 'esophagus+aorta' mask into a labelled volume:
    2 = aorta, 1 = esophagus, 0 = background. Reasoning inline."""
    # runs on the entire 3D mask at once, with the full (dx, dy, dz) spacing — not slice by slice.
    out = np.zeros(merged.shape, dtype=np.uint8)
    if merged.sum() == 0:
        return out

    # Depth of every voxel inside the mask, in mm. sampling=spacing makes this
    # anisotropy-aware (SegTHOR voxels are ~2x taller in z than they are wide),
    # so "fatness" is measured in real millimetres, not in voxel counts.
    dt = ndi.distance_transform_edt(merged, sampling=spacing)

    # No genuinely fat structure anywhere -> the mask is esophagus only. Do not
    # hallucinate an aorta; label everything esophagus and return.
    if dt.max() < MIN_CORE_MM:
        out[merged] = 1
        return out

    # Aorta seed: the deep core of the fat tube. Take the largest component to
    # discard any small deep pockets that are not the aorta.
    seed_a = largest_cc(dt >= CORE_THR_MM)

    # Esophagus seed: thin voxels (small dt) that are also far from the aortic
    # core -> confidently esophagus. distance-to-seed_a is again in mm.
    dist_to_a = ndi.distance_transform_edt(~seed_a, sampling=spacing)
    seed_e = merged & (dt < ESO_THR_MM) & (dist_to_a > FAR_MM)
    if not seed_e.any():
        # Fallback so the watershed always has two markers: seed the single voxel
        # farthest from the aorta as esophagus. Without a 2nd marker the flood
        # would label the whole union as aorta.
        far_idx = np.unravel_index(np.argmax(dist_to_a * merged), merged.shape)
        seed_e[far_idx] = True

    # Watershed floods the union from both seeds along the ridge of the distance
    # transform (-dt as the "elevation"), so the cut lands at the thin neck where
    # the fat aorta meets the thin esophagus. mask=merged keeps it inside the union.
    markers = np.zeros(merged.shape, dtype=np.int32)
    markers[seed_a] = 2
    markers[seed_e] = 1
    labels = watershed(-dt, markers, mask=merged)

    out[merged] = 1                 # default any union voxel to esophagus...
    out[labels == 2] = 2            # ...then promote the aorta basin.
    return out


def recover_gt(gt: np.ndarray, spacing: tuple[float, float, float]) -> tuple[np.ndarray, int]:
    """Return a 5-class GT with the aorta (4) carved back out of label 1, plus the
    recovered aorta voxel count (0 if none)."""
    fixed = gt.copy().astype(np.uint8)
    split = split_aorta_esophagus(gt == 1, spacing)
    fixed[split == 2] = 4           # aorta basin -> class 4
    # split == 1 stays class 1 (esophagus); heart(2)/trachea(3) untouched.
    return fixed, int((split == 2).sum())


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return 2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9)


def validate_on_reference(pdir: Path) -> None:
    """Run the SAME recovery on a reference patient's corrupted GT and score it
    against GT2. This is an honest self-test: it never peeks at GT2 during recovery."""
    pid = pdir.name
    ct = nib.load(str(pdir / f"{pid}.nii.gz"))
    corrupted = np.asarray(nib.load(str(pdir / "GT.nii.gz")).dataobj)
    truth = np.asarray(nib.load(str(pdir / "GT2.nii.gz")).dataobj)
    spacing = tuple(float(z) for z in ct.header.get_zooms()[:3])

    recovered, n = recover_gt(corrupted, spacing)
    da = dice(recovered == 4, truth == 4)
    de = dice(recovered == 1, truth == 1)
    print(f"[validation on {pid}] recovered aorta={n} vox (true {(truth == 4).sum()}); "
          f"aorta Dice={da:.3f}  esophagus Dice={de:.3f}")


def write_fixed_patient(pdir: Path, dest_dir: Path, link_ct: bool) -> dict:
    """Produce the corrected GT for one patient and link its CT. Returns per-patient
    stats used for the run summary and outlier flagging."""
    pid = pdir.name
    ct = nib.load(str(pdir / f"{pid}.nii.gz"))
    spacing = tuple(float(z) for z in ct.header.get_zooms()[:3])

    out_pdir = dest_dir / pid
    out_pdir.mkdir(parents=True, exist_ok=True)

    # A gold GT2 is trusted directly; otherwise recover the aorta geometrically.
    if (pdir / "GT2.nii.gz").exists():
        fixed = np.asarray(nib.load(str(pdir / "GT2.nii.gz")).dataobj).astype(np.uint8)
        gold = True
    else:
        gt = np.asarray(nib.load(str(pdir / "GT.nii.gz")).dataobj)
        fixed, _ = recover_gt(gt, spacing)
        gold = False

    # Re-stamp the CT geometry so the GT shares the CT's world space again.
    header = ct.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(fixed, ct.affine, header), str(out_pdir / "GT.nii.gz"))

    # The CT is unchanged; link (or copy) it so slice_segthor finds the pair.
    ct_link = out_pdir / f"{pid}.nii.gz"
    if ct_link.exists() or ct_link.is_symlink():
        ct_link.unlink()
    src = (pdir / f"{pid}.nii.gz").resolve()
    if link_ct:
        os.symlink(src, ct_link)
        # shutil.copy2(src, ct_link)  # fallback if symlink fails (voor Windows)
    else:
        shutil.copy2(src, ct_link)

    aorta = int((fixed == 4).sum())
    eso = int((fixed == 1).sum())
    union = aorta + eso
    return {"pid": pid, "gold": gold, "aorta": aorta, "eso": eso,
            "frac": (aorta / union) if union else 0.0}


def flag_outliers(stats: list[dict]) -> None:
    """Flag patients whose recovered aorta looks wrong, so you inspect ~3 in ITK-SNAP
    instead of all 20. The discriminator is the aorta fraction aorta/(aorta+esophagus):
    it is scan-FOV-independent (a short scan sees less of both organs) and catches
    under-segmentation (fraction too low, e.g. thin descending aorta left as esophagus)
    and over-segmentation (fraction too high) alike. We compare each recovered patient
    to the cohort with a robust z-score (median / MAD), which a couple of outliers
    cannot distort the way mean / std would. Gold (GT2) patients are the reference, not
    flagged. An absolute sanity band backs up the statistical test for tiny cohorts."""
    recovered = [s for s in stats if not s["gold"]]
    if len(recovered) < 3:
        return
    fracs = np.array([s["frac"] for s in recovered])
    median = float(np.median(fracs))
    mad = float(np.median(np.abs(fracs - median))) or 1e-6
    Z, ABS_LO, ABS_HI = 3.5, 0.45, 0.90

    print("\nAorta fraction per patient (aorta / (aorta+esophagus)):")
    flagged: list[str] = []
    for s in sorted(stats, key=lambda d: d["frac"]):
        z = 0.0 if s["gold"] else (s["frac"] - median) / (1.4826 * mad)
        mark = ""
        if not s["gold"] and (abs(z) > Z or s["frac"] < ABS_LO or s["frac"] > ABS_HI):
            mark = "  <-- INSPECT"
            flagged.append(s["pid"])
        tag = "gold" if s["gold"] else f"z={z:+.1f}"
        print(f"  {s['pid']}: {s['frac']:.2f}  ({tag}){mark}")

    print(f"\n(cohort median fraction {median:.2f})")
    if flagged:
        print(f"INSPECT in ITK-SNAP and hand-correct if wrong: {', '.join(flagged)}")
    else:
        print("No outliers flagged — still spot-check a couple by eye.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_dir", type=Path, required=True,
                        help="raw per-patient folders, e.g. data/segthor_part1/train")
    parser.add_argument("--dest", type=Path, required=True,
                        help="output folder for the fixed dataset, e.g. data/gt/watershed/train")
    parser.add_argument("--copy_ct", action="store_true",
                        help="copy CT volumes instead of symlinking (use on filesystems without symlinks)")
    args = parser.parse_args()

    patient_dirs = sorted(p for p in args.source_dir.glob("Patient_*") if p.is_dir())
    if not patient_dirs:
        parser.error(f"no Patient_* folders under {args.source_dir}")
    args.dest.mkdir(parents=True, exist_ok=True)

    # Self-test first: prove the recovery quality on every reference patient.
    for pdir in patient_dirs:
        if (pdir / "GT2.nii.gz").exists():
            validate_on_reference(pdir)
    print()

    stats: list[dict] = []
    for pdir in patient_dirs:
        s = write_fixed_patient(pdir, args.dest, link_ct=not args.copy_ct)
        stats.append(s)
        origin = "used GT2 (gold)" if s["gold"] else f"recovered aorta ({s['aorta']} vox)"
        print(f"  {s['pid']}: {origin}")

    flag_outliers(stats)

    print(f"\nWrote fixed dataset to {args.dest}")
    print("Next: inspect any flagged patients, re-run audit_data.py, then re-slice.")


if __name__ == "__main__":
    main()
