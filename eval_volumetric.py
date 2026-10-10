"""3D evaluation per patient and per organ, from stitched NIfTI predictions.

All distances are in mm, using the voxel spacing of the ground truth.
  dice       volumetric overlap
  prec, rec  voxel precision and recall
  hd95       95th percentile of the pooled symmetric surface distances
  assd       average symmetric surface distance
  nsd        fraction of surface within TAU_MM of the other surface
  cldice     centerline Dice (topology; meaningful for tubular organs only)
"""

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize

NAMES = ["esophagus", "heart", "trachea", "aorta"]
METRICS = ["dice", "prec", "rec", "hd95", "assd", "nsd", "cldice"]
TAU_MM = 3.0
MARGIN = 10  # voxels kept around the organs when cropping, for speed


def crop(p: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argwhere(p | g)
    lo = np.maximum(idx.min(0) - MARGIN, 0)
    hi = idx.max(0) + MARGIN + 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    return p[sl], g[sl]


def surface(m: np.ndarray) -> np.ndarray:
    return m & ~binary_erosion(m)


def surface_distances(p, g, spacing) -> tuple[np.ndarray, np.ndarray]:
    """Distances (mm) from each surface voxel of p to the surface of g, and vice versa."""
    sp, sg = surface(p), surface(g)
    dist_to_g = distance_transform_edt(~sg, sampling=spacing)
    dist_to_p = distance_transform_edt(~sp, sampling=spacing)
    return dist_to_g[sp], dist_to_p[sg]


def cldice(p: np.ndarray, g: np.ndarray) -> float:
    sk_p, sk_g = skeletonize(p).astype(bool), skeletonize(g).astype(bool)
    if sk_p.sum() == 0 or sk_g.sum() == 0:
        return 0.0
    tprec = (sk_p & g).sum() / sk_p.sum()
    tsens = (sk_g & p).sum() / sk_g.sum()
    return 0.0 if tprec + tsens == 0 else 2 * tprec * tsens / (tprec + tsens)


def organ_metrics(p: np.ndarray, g: np.ndarray, spacing) -> list[float]:
    if g.sum() == 0:
        return [np.nan] * len(METRICS)  # organ absent from GT: undefined
    if p.sum() == 0:
        # Missed entirely: overlap-type metrics are 0, distances are undefined.
        return [0.0, np.nan, 0.0, np.nan, np.nan, 0.0, 0.0]
    p, g = crop(p, g)
    tp = (p & g).sum()
    d_pg, d_gp = surface_distances(p, g, spacing)
    d_all = np.concatenate([d_pg, d_gp])
    return [2 * tp / (p.sum() + g.sum()),
            tp / p.sum(),
            tp / g.sum(),
            np.percentile(d_all, 95),
            d_all.mean(),
            (d_all <= TAU_MM).mean(),
            cldice(p, g)]


pred_dir, gt_pattern = Path(sys.argv[1]), sys.argv[2]
results: dict[str, dict[str, np.ndarray]] = {m: {} for m in METRICS}

for pred_path in sorted(pred_dir.glob("*.nii.gz")):
    pid = pred_path.name.split(".")[0]
    gt_nib = nib.load(gt_pattern.format(id_=pid))
    gt = np.asarray(gt_nib.dataobj)
    pred = np.asarray(nib.load(pred_path).dataobj)
    assert pred.shape == gt.shape, (pid, pred.shape, gt.shape)
    spacing = gt_nib.header.get_zooms()[:3]

    per_organ = np.array([organ_metrics(pred == k, gt == k, spacing) for k in range(1, 5)])
    for j, m in enumerate(METRICS):
        results[m][pid] = per_organ[:, j]

    print(pid)
    for n, row in zip(NAMES, per_organ):
        print(f"   {n:10s} " + "  ".join(f"{m}={v:.3f}" for m, v in zip(METRICS, row)))

print("\nMean ± SD over patients (hd95 and assd in mm)")
for m in METRICS:
    arr = np.stack(list(results[m].values()))
    cols = "  ".join(f"{n}={np.nanmean(c):.3f}±{np.nanstd(c):.3f}" for n, c in zip(NAMES, arr.T))
    print(f"  {m:7s} {cols}")
    np.savez(pred_dir / f"{m}.npz", **results[m])