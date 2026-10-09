"""Volumetric (3D) Dice per patient and per organ, from stitched NIfTI predictions."""

import sys
from pathlib import Path

import numpy as np
import nibabel as nib

NAMES = ["esophagus", "heart", "trachea", "aorta"]


def dice_3d(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    p, g = pred == k, gt == k
    denom = p.sum() + g.sum()
    return np.nan if denom == 0 else 2 * (p & g).sum() / denom


pred_dir, gt_pattern = Path(sys.argv[1]), sys.argv[2]
results: dict[str, np.ndarray] = {}

for pred_path in sorted(pred_dir.glob("*.nii.gz")):
    pid = pred_path.name.split(".")[0]
    pred = np.asarray(nib.load(pred_path).dataobj)
    gt = np.asarray(nib.load(gt_pattern.format(id_=pid)).dataobj)
    assert pred.shape == gt.shape, (pid, pred.shape, gt.shape)
    results[pid] = np.array([dice_3d(pred, gt, k) for k in range(1, 5)])
    print(pid, "  ".join(f"{n}={d:.3f}" for n, d in zip(NAMES, results[pid])))

scores = np.stack(list(results.values()))
print("\nMean ± SD over patients")
for n, col in zip(NAMES, scores.T):
    print(f"  {n:10s} {np.nanmean(col):.3f} ± {np.nanstd(col):.3f}")
print(f"  {'foreground':10s} {np.nanmean(scores):.3f}")

np.savez(pred_dir / "dice3d.npz", **results)