#!/usr/bin/env python3

"""
Intensity preprocessing for CT (SegTHOR).

This module isolates the CT intensity-normalization step so the change is easy to
review and to A/B against the original pipeline. It replaces the baseline
per-volume min-max normalization in ``slice_segthor.norm_arr`` with **fixed HU
windowing**.

Why HU windowing
----------------
CT voxels are calibrated in Hounsfield Units (an absolute physical scale), so a
fixed intensity window is physically meaningful across patients. The baseline
min-max normalization is not: it rescales each volume by its own global min/max,
so a single high-attenuation outlier (metal, contrast bolus, reconstruction
artifact) blows up the range and crushes soft tissue into a few grey levels.

In this dataset that is not hypothetical: several volumes reach 25000-26000 HU
while their 99th percentile is only ~300-400 HU. Under min-max, the mediastinal
soft tissue (~-50..+100 HU) that separates the esophagus, aorta and heart maps
into a handful of grey levels and becomes nearly flat -- worst for the esophagus,
the hardest SegTHOR class. A fixed soft-tissue window restores that contrast and
makes the same HU map to the same input value in every patient.

The default is the clinical **mediastinal / soft-tissue window** (level 40 HU,
width 400 HU -> clip to [-160, 240] HU), the standard window for reading
mediastinal structures and a common preprocessing choice for thoracic soft-organ
segmentation (cf. windowing as a standard CT step; nnU-Net likewise clips CT
intensities before normalization, Isensee et al., Nature Methods 2021).

Trade-off to keep in mind: a soft-tissue window maps the air-filled trachea lumen
to 0 (like external air), so the trachea is learned from its wall/shape rather
than lumen intensity. This is expected for a single mediastinal window; a lung
window or a multi-window stack could be explored separately if trachea recall
suffers.
"""

import numpy as np

# Named clinical CT windows as (level, width) in HU. `level` is the window
# center, `width` the total window span; the clip range is level +/- width/2.
CT_WINDOWS: dict[str, tuple[float, float]] = {
    "mediastinal": (40.0, 400.0),   # soft tissue: esophagus, heart, aorta -> [-160, 240]
    "soft_tissue": (50.0, 400.0),   # slight variant sometimes used         -> [-150, 250]
    "lung": (-600.0, 1500.0),       # airways / lung parenchyma              -> [-1350, 150]
}

# Default window used by the slicing pipeline.
MEDIASTINAL: tuple[float, float] = CT_WINDOWS["mediastinal"]


def window_bounds(level: float, width: float) -> tuple[float, float]:
    """Return the (low, high) HU clip bounds for a window given as (level, width)."""
    assert width > 0, f"window width must be positive, got {width}"
    return level - width / 2.0, level + width / 2.0


def apply_hu_window(volume: np.ndarray, level: float, width: float) -> np.ndarray:
    """Clip a CT volume to a fixed HU window and scale it to uint8 [0, 255].

    Unlike per-volume min-max, the mapping is anchored to the *fixed* window
    bounds, so a given HU value maps to the same output value in every patient
    (that consistency is the whole point of windowing).

    Parameters
    ----------
    volume : np.ndarray
        CT data in Hounsfield Units (any integer/float dtype).
    level, width : float
        Window center and total width in HU (see ``CT_WINDOWS``).

    Returns
    -------
    np.ndarray
        uint8 array, same shape as ``volume``, values in [0, 255].
    """
    lo, hi = window_bounds(level, width)

    casted = volume.astype(np.float32)
    clipped = np.clip(casted, lo, hi)
    norm = (clipped - lo) / (hi - lo)          # -> [0, 1]
    res = (norm * 255.0).round()

    assert res.min() >= 0, res.min()
    assert res.max() <= 255, res.max()

    return res.astype(np.uint8)
