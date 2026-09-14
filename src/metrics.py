"""Training-time 2D metrics built from per-slice counts (intersection, gt area, pred area).

Conventions match dataset_analysis: hard Dice, no smoothing, joint-empty = NaN (not 1).
`dice` is computed per patient (counts summed over the patient's slices, i.e. a 3D Dice at
256x256 resolution) and averaged over patients. The course code's slice-mean Dice with
smooth=1e-8 is kept as `dice_legacy_fg` only to compare against the original baseline.
"""
import warnings

import numpy as np
import torch
from torch import Tensor

from src.data import patient_of


def slice_counts(pred_class: Tensor, gt_onehot: Tensor) -> np.ndarray:
    """(B,H,W) class map + (B,K,H,W) one-hot -> (B,K,3) int64 [intersection, gt, pred]."""
    num_classes = gt_onehot.shape[1]
    pred = torch.nn.functional.one_hot(pred_class, num_classes).permute(0, 3, 1, 2).bool()
    gt = gt_onehot.bool()
    counts = torch.stack([(gt & pred).sum((2, 3)), gt.sum((2, 3)), pred.sum((2, 3))], dim=-1)
    return counts.cpu().numpy().astype(np.int64)


def dice_from_counts(counts: np.ndarray) -> np.ndarray:
    inter, gt, pred = counts[..., 0], counts[..., 1], counts[..., 2]
    denom = gt + pred
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * inter / denom, np.nan)


def legacy_dice_fg(counts: np.ndarray) -> float:
    """main.py's `log_dice_val[e, :, 1:].mean()`: smoothed slice Dice over all foreground classes."""
    t = torch.from_numpy(counts).float()
    dice = (2 * t[..., 0] + 1e-8) / (t[..., 1] + t[..., 2] + 1e-8)
    return dice[:, 1:].mean().item()


def epoch_metrics(prefix: str, counts: np.ndarray, stems: list[str], cfg: dict) -> dict:
    patients = [patient_of(s, cfg["data"]["patient_regex"]) for s in stems]
    ids = sorted(set(patients))
    per_patient = np.zeros((len(ids), counts.shape[1], 3), dtype=np.int64)
    index = {p: i for i, p in enumerate(ids)}
    for p, c in zip(patients, counts):
        per_patient[index[p]] += c
    dice = dice_from_counts(per_patient)  # (P, K)

    out = {}
    names = cfg["data"]["class_names"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN class -> NaN, on purpose
        per_class = np.nanmean(dice, axis=0)
        for k in cfg["eval"]["classes"]:
            out[f"{prefix}_dice_{names[k]}"] = float(per_class[k])
        out[f"{prefix}_dice_fg"] = float(np.nanmean(per_class[cfg["eval"]["classes"]]))
    out[f"{prefix}_dice_legacy_fg"] = legacy_dice_fg(counts)
    return out
