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
"""

from pathlib import Path

import numpy as np
import nibabel as nib
from skimage.io import imread
from skimage.transform import resize

from segpipe.data import CLASS_NAMES, K
from preprocessing import center_crop_pad
from stitch import get_z

# Predictions are saved as class*63 PNGs (see train.py); {0,63,126,189,252} for K=5.
_LABEL_STEP = 63
_LABEL_VALUES = {k * _LABEL_STEP for k in range(K)}


def dice(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    # Volumetric overlap: weight voxel counts by the physical voxel volume (mm^3).
    # For Dice this cancels out (every voxel shares the same volume), so the number
    # is identical to the unweighted form, but the metric is now genuinely computed
    # in physical units and serves as the template for spacing-dependent metrics.
    voxel_volume = float(np.prod(spacing))
    intersection = float((pred & gt).sum()) * voxel_volume
    total = float(pred.sum() + gt.sum()) * voxel_volume
    return 1.0 if total == 0 else 2 * intersection / total


# name -> fn(pred_mask, gt_mask, spacing) -> float
METRICS: dict = {
    "dice": dice,
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


def evaluate_3d(run_dir: Path, cfg, patient_ids: list[str], metric_names) -> dict:
    """Reconstruct best_epoch/val onto the native grid and score against the GT volumes."""
    for name in metric_names:
        if name not in METRICS:
            raise KeyError(f"unknown metric '{name}'. Known: {sorted(METRICS)}")

    images = sorted((run_dir / "best_epoch" / "val").glob("*.png"))
    volume_dir = run_dir / "volumes" / "val"
    volume_dir.mkdir(parents=True, exist_ok=True)
    source_pattern = str(Path(cfg.data.gt) / "train" / "{id_}" / "GT.nii.gz")

    scores = {name: {} for name in metric_names}  # metric -> patient -> K values
    for pid in patient_ids:
        idxes = [i for i, p in enumerate(images) if p.stem.rsplit("_", 1)[0] == pid]
        gt_nib = nib.load(source_pattern.format(id_=pid))
        gt = np.asarray(gt_nib.dataobj)
        spacing = gt_nib.header.get_zooms()[:3]

        pred = stitch_to_native(images, idxes, gt.shape, spacing, cfg)
        assert pred.shape == gt.shape, (pred.shape, gt.shape)
        nib.save(nib.nifti1.Nifti1Image(pred, affine=gt_nib.affine, header=gt_nib.header),
                 str(volume_dir / f"{pid}.nii.gz"))

        for name in metric_names:
            scores[name][pid] = np.array([METRICS[name](pred == k, gt == k, spacing) for k in range(K)])

    out_dir = run_dir / "metrics_3d"
    out_dir.mkdir(exist_ok=True)
    summary = {}
    for name, per_patient in scores.items():
        np.savez(out_dir / f"{name}.npz", **per_patient)  # patient -> K values
        table = np.stack(list(per_patient.values()))  # patients x K; averages leave out the background
        summary[name] = {
            "mean": round(float(table[:, 1:].mean()), 4),
            "per_class": {CLASS_NAMES[k]: round(float(table[:, k].mean()), 4) for k in range(1, K)},
            "per_patient": {pid: {CLASS_NAMES[k]: round(float(v[k]), 4) for k in range(1, K)}
                            for pid, v in per_patient.items()},
        }
    return summary
