"""3D evaluation of a run's best-epoch validation predictions.

Predictions are reconstructed onto the **native GT grid** before scoring, so 3D
metrics are always computed against the untouched native-spacing GT. This works
for both slicing modes:

* plain (``target_spacing: null``): each 2D slice was resized to ``shape``; the
  inverse is a per-slice resize back to the native in-plane size (z unchanged).
* resampled (``target_spacing: [sx, sy, sz]``): each volume was resampled to a
  common spacing and center crop/pad'd to ``shape`` (see preprocessing.py). The
  inverse undoes the crop/pad, then resamples the volume back to native spacing.

Everything is nearest-neighbour (``order=0``) so labels stay integer. The single
reconstruction entry point is ``stitch_to_native`` -- extend it (not evaluate_3d)
when adding spacing-aware behaviour.

Metrics come in two families (see METRICS_PLAN.md): overlap (dice) and boundary
(hd / hd95 / assd / nsd). Because reconstruction always lands on the native grid,
the boundary metrics use the true GT voxel spacing (mm) for every run, resampled
or not -- so a resampled run is scored on the same physical footing as a plain one.
"""

from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from skimage.io import imread
from skimage.transform import resize

from segpipe.data import CLASS_NAMES, K
from preprocessing import center_crop_pad
from stitch import get_z

# Predictions are saved as class*63 PNGs (see train.py); {0,63,126,189,252} for K=5.
_LABEL_STEP = 63
_LABEL_VALUES = {k * _LABEL_STEP for k in range(K)}

# Tolerance (mm) for the Normalised Surface Dice.
NSD_TAU_MM: float = 1.0
# Looser tolerance for nsd3: the GT slice spacing is 2.0 or 2.5 mm, so at 1 mm a
# surface one slice off in z already fails; 3 mm forgives a one-slice error.
NSD3_TAU_MM: float = 3.0


def dice(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    # Volumetric overlap: weight voxel counts by the physical voxel volume (mm^3).
    # For Dice this cancels out (every voxel shares the same volume), so the number
    # is identical to the unweighted form, but the metric is now genuinely computed
    # in physical units and serves as the template for spacing-dependent metrics.
    voxel_volume = float(np.prod(spacing))
    intersection = float((pred & gt).sum()) * voxel_volume
    total = float(pred.sum() + gt.sum()) * voxel_volume
    return 1.0 if total == 0 else 2 * intersection / total


# ---------------------------------------------------------------------------
# Boundary metrics (Metrics Reloaded recommendations for organ segmentation).
# Distances are computed in millimetres, using the voxel spacing, on the surface
# voxels of each class -- so anisotropic slice thickness (e.g. SegTHOR ~0.98mm
# in-plane vs 2.5mm through-plane) is handled correctly. HD (max) is kept next to
# HD95 so the two can be compared: max-HD is dominated by a single outlier voxel,
# HD95 is its robust cousin.
#   hd   : max (classic) Hausdorff  -> worst-case boundary error (outlier-sensitive)
#   hd95 : 95th-percentile Hausdorff -> robust worst-case
#   assd : average symmetric surface distance -> mean boundary error
#   nsd  : Normalised Surface Dice at NSD_TAU_MM -> fraction of surface within tau
# When the class is absent from both pred and gt the boundary is undefined -> NaN,
# so the organ is left out of the nanmean. When it is absent from only one of them
# (a missed organ, or an organ predicted that is not there) the score is the worst
# case: the volume's diagonal in mm for the distances, 0 for NSD. Returning NaN
# there would drop the failure from the mean and make a missed organ look better
# than a badly segmented one.
# ---------------------------------------------------------------------------

def _empty_score(pred: np.ndarray, gt: np.ndarray, spacing, worst_distance: bool) -> float:
    """Score for a class with an empty surface in pred and/or gt (see the note above)."""
    if not pred.any() and not gt.any():
        return float("nan")
    if not worst_distance:
        return 0.0
    return float(np.linalg.norm(np.asarray(gt.shape) * np.asarray(spacing, dtype=np.float64)))


def _surface_distances(pred: np.ndarray, gt: np.ndarray, spacing):
    """Symmetric nearest-surface distances (mm) between two binary masks.
    Returns (d_pred_to_gt, d_gt_to_pred), or (None, None) if either mask is empty."""
    def surface(mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.empty((0, mask.ndim))
        return np.argwhere(mask & ~binary_erosion(mask))

    pred_surf, gt_surf = surface(pred), surface(gt)
    if len(pred_surf) == 0 or len(gt_surf) == 0:
        return None, None

    sp = np.asarray(spacing, dtype=np.float64)
    d_pred_to_gt, _ = cKDTree(gt_surf * sp).query(pred_surf * sp)
    d_gt_to_pred, _ = cKDTree(pred_surf * sp).query(gt_surf * sp)
    return d_pred_to_gt, d_gt_to_pred


def hd(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    d_pg, d_gp = _surface_distances(pred, gt, spacing)
    if d_pg is None:
        return _empty_score(pred, gt, spacing, worst_distance=True)
    return float(max(d_pg.max(), d_gp.max()))


def hd95(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    d_pg, d_gp = _surface_distances(pred, gt, spacing)
    if d_pg is None:
        return _empty_score(pred, gt, spacing, worst_distance=True)
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


def assd(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    d_pg, d_gp = _surface_distances(pred, gt, spacing)
    if d_pg is None:
        return _empty_score(pred, gt, spacing, worst_distance=True)
    return float((d_pg.sum() + d_gp.sum()) / (len(d_pg) + len(d_gp)))


def nsd(pred: np.ndarray, gt: np.ndarray, spacing, tau: float = NSD_TAU_MM) -> float:
    d_pg, d_gp = _surface_distances(pred, gt, spacing)
    if d_pg is None:
        return _empty_score(pred, gt, spacing, worst_distance=False)
    within = (d_pg <= tau).sum() + (d_gp <= tau).sum()
    return float(within / (len(d_pg) + len(d_gp)))


def nsd3(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    return nsd(pred, gt, spacing, tau=NSD3_TAU_MM)


# name -> fn(pred_mask, gt_mask, spacing) -> float
METRICS: dict = {
    "dice": dice,
    "hd": hd,
    "hd95": hd95,
    "assd": assd,
    "nsd": nsd,
    "nsd3": nsd3,
}


def _read_pred_stack(images: list[Path], idxes: list[int]) -> np.ndarray:
    """Stack a patient's predicted PNG slices into (H, W, Zpred) integer class labels."""
    h, w = imread(images[idxes[0]]).shape
    zmax = max(get_z(images[i]) for i in idxes)
    stack = np.zeros((h, w, zmax + 1), dtype=np.int16)
    for i in idxes:
        sl = imread(images[i])
        assert set(np.unique(sl)) <= _LABEL_VALUES, np.unique(sl)
        stack[:, :, get_z(images[i])] = sl // _LABEL_STEP
    return stack


def stitch_to_native(images: list[Path], idxes: list[int],
                     native_shape: tuple[int, int, int],
                     native_spacing, cfg) -> np.ndarray:
    """Reconstruct a patient's prediction onto the native GT grid ``native_shape``.

    ``native_spacing`` is the GT voxel spacing (mm) and is only used for resampled
    runs, to work out the resampled in-plane size that the crop/pad started from.
    Returns an int16 label volume with exactly ``native_shape``.
    """
    X, Y, Z = native_shape
    pred = _read_pred_stack(images, idxes)  # (H, W, Zpred)

    target_spacing = cfg.data.get("target_spacing")
    if target_spacing is not None:
        # Undo the per-slice center crop/pad back to the resampled in-plane grid
        # (Xp, Yp) = round(native_size * native_spacing / target_spacing), i.e. the
        # size slice_patient's resample produced before crop/pad'ing to `shape`.
        Xp = max(1, round(X * native_spacing[0] / target_spacing[0]))
        Yp = max(1, round(Y * native_spacing[1] / target_spacing[1]))
        pred = np.stack([center_crop_pad(pred[:, :, z], (Xp, Yp)) for z in range(pred.shape[2])], axis=-1)

    # Resample (in-plane and, when resampled, along z too) to the exact native grid.
    out = resize(pred, (X, Y, Z), order=0, mode="constant", preserve_range=True, anti_aliasing=False)
    return np.rint(out).astype(np.int16)


def evaluate_3d(run_dir: Path, cfg, patient_ids: list[str], metric_names, postprocess=None,
                raw: bool = True) -> dict:
    """Reconstruct best_epoch/val onto the native grid and score against the GT volumes.

    Returns {"metrics_3d": ...}. With a ``postprocess`` callable (segpipe.postprocess.build_postprocess)
    the post-processed volumes are scored too, as {"metrics_3d_post": ...}; the raw scores are unchanged.
    raw=False (post-processing experiments, postprocess_run.py) scores only the post-processed volumes,
    and reports them as "metrics_3d".
    """
    for name in metric_names:
        if name not in METRICS:
            raise KeyError(f"unknown metric '{name}'. Known: {sorted(METRICS)}")
    assert raw or postprocess is not None, "raw=False needs a postprocess"

    images = sorted((run_dir / "best_epoch" / "val").glob("*.png"))
    # suffix -> (volume folder, post-processing or None)
    variants = {"": (run_dir / "volumes" / "val", None if raw else postprocess)}
    if raw and postprocess is not None:
        variants["_post"] = (run_dir / "volumes" / "val_post", postprocess)
    for volume_dir, _ in variants.values():
        volume_dir.mkdir(parents=True, exist_ok=True)
    source_pattern = str(Path(cfg.data.gt) / "train" / "{id_}" / "GT.nii.gz")

    scores = {suffix: {name: {} for name in metric_names} for suffix in variants}  # -> metric -> patient -> K values
    for pid in patient_ids:
        idxes = [i for i, p in enumerate(images) if p.stem.rsplit("_", 1)[0] == pid]
        gt_nib = nib.load(source_pattern.format(id_=pid))
        gt = np.asarray(gt_nib.dataobj)
        spacing = gt_nib.header.get_zooms()[:3]

        pred = stitch_to_native(images, idxes, gt.shape, spacing, cfg)
        assert pred.shape == gt.shape, (pred.shape, gt.shape)
        for suffix, (volume_dir, step) in variants.items():
            volume = pred if step is None else step(pred, spacing)
            nib.save(nib.nifti1.Nifti1Image(volume, affine=gt_nib.affine, header=gt_nib.header),
                     str(volume_dir / f"{pid}.nii.gz"))
            for name in metric_names:
                scores[suffix][name][pid] = np.array([METRICS[name](volume == k, gt == k, spacing)
                                                      for k in range(K)])

    return {f"metrics_3d{suffix}": _summarise(scores[suffix], run_dir / f"metrics_3d{suffix}")
            for suffix in variants}


def _summarise(scores: dict, out_dir: Path) -> dict:
    """metric -> patient -> K values: save them as npz and reduce to mean / per class / per patient."""
    out_dir.mkdir(exist_ok=True)
    summary = {}
    for name, per_patient in scores.items():
        np.savez(out_dir / f"{name}.npz", **per_patient)  # patient -> K values
        table = np.stack(list(per_patient.values()))  # patients x K; averages leave out the background
        # nanmean: boundary metrics are NaN only for organs absent from both the
        # prediction and the GT of a patient, which must not poison the mean
        # (no-op for dice, which never returns NaN). A missed organ is scored as
        # the worst case, not NaN, so it does count.
        summary[name] = {
            "mean": round(float(np.nanmean(table[:, 1:])), 4),
            "per_class": {CLASS_NAMES[k]: round(float(np.nanmean(table[:, k])), 4) for k in range(1, K)},
            "per_patient": {pid: {CLASS_NAMES[k]: round(float(v[k]), 4) for k in range(1, K)}
                            for pid, v in per_patient.items()},
        }
    return summary
