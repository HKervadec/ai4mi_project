"""3D evaluation of a run's best-epoch validation predictions."""

from pathlib import Path

import numpy as np
import nibabel as nib

from segpipe.data import CLASS_NAMES, K
from stitch import merge_patient


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


def evaluate_3d(run_dir: Path, cfg, patient_ids: list[str], metric_names) -> dict:
    """Stitch best_epoch/val into volumes (stitch.py) and score them against the GT volumes."""
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
        merge_patient(pid, str(volume_dir), images, idxes, 256, source_pattern)

        pred = np.asarray(nib.load(str(volume_dir / f"{pid}.nii.gz")).dataobj)
        gt_nib = nib.load(source_pattern.format(id_=pid))
        gt = np.asarray(gt_nib.dataobj)
        spacing = gt_nib.header.get_zooms()[:3]
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
