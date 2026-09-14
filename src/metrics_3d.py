"""3D segmentation metrics on binary volumes, with physical voxel spacing (mm).

Conventions (same as dataset_analysis, stated so tables are unambiguous):
  dice  : 2|G∩P| / (|G|+|P|); both empty -> NaN; one empty -> 0
  hd95  : max of the two directed 95th-percentile surface distances (mm); NaN if either is empty
  assd  : mean of all symmetric surface distances (mm); NaN if either is empty
"""
import numpy as np
from scipy import ndimage

METRICS = ("dice", "hd95", "assd")
_STRUCTURE = ndimage.generate_binary_structure(3, 1)


def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    total = np.count_nonzero(gt) + np.count_nonzero(pred)
    return 2 * np.count_nonzero(gt & pred) / total if total else np.nan


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=_STRUCTURE, border_value=0)


def surface_distances(gt: np.ndarray, pred: np.ndarray, spacing) -> tuple[np.ndarray, np.ndarray]:
    """Directed distances pred-surface -> gt-surface and gt-surface -> pred-surface, in mm."""
    # Crop to the joint bounding box and pad by one background voxel: exact (all voxels outside the
    # box are background for both masks) and much faster on 512x512xZ CT volumes.
    coords = np.argwhere(gt | pred)
    box = tuple(slice(a, b + 1) for a, b in zip(coords.min(0), coords.max(0)))
    sg, sp = _surface(np.pad(gt[box], 1)), _surface(np.pad(pred[box], 1))
    to_gt = ndimage.distance_transform_edt(~sg, sampling=spacing)
    to_pred = ndimage.distance_transform_edt(~sp, sampling=spacing)
    return to_gt[sp], to_pred[sg]


def volume_metrics(gt: np.ndarray, pred: np.ndarray, spacing) -> dict:
    out = {"dice": dice(gt, pred), "hd95": np.nan, "assd": np.nan}
    if gt.any() and pred.any():
        a, b = surface_distances(gt, pred, spacing)
        out["hd95"] = float(max(np.percentile(a, 95), np.percentile(b, 95)))
        out["assd"] = float(np.concatenate([a, b]).mean())
    return out
