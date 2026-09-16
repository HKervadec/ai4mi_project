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
from scipy.ndimage import zoom

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


# --------------------------------------------------------------------------- #
# Spatial preprocessing: voxel-spacing resampling                              #
# --------------------------------------------------------------------------- #
#
# Why resampling
# --------------
# CT voxel spacing (mm/pixel) is set by the acquisition protocol -- field of
# view, reconstruction matrix, slice thickness -- not by anatomy. In SegTHOR the
# in-plane spacing varies 0.896..1.367 mm (1.53x) and slice thickness 2.0..2.5 mm
# across patients, so the *same* physical organ lands on a different pixel grid in
# different scans. That is a non-anatomical confound the network should not have
# to untangle (true anatomical size variation is preserved -- only the sampling
# grid is harmonized). Resampling every volume to one fixed spacing removes it;
# this is the first step of the nnU-Net preprocessing pipeline (Isensee et al.,
# Nature Methods 2021).
#
# Note the interaction with the downstream fixed-size (256x256) output: resizing
# every slice to a fixed *pixel* count re-normalizes the grid to each patient's
# field of view, which would cancel the in-plane harmonization. So resampling is
# paired with center crop/pad (constant mm/pixel) instead of resize -- see
# ``center_crop_pad`` and ``slice_segthor.slice_patient``.


def resample_volume(volume: np.ndarray,
                    src_spacing: tuple[float, float, float],
                    dst_spacing: tuple[float, float, float],
                    order: int) -> np.ndarray:
    """Resample a 3D volume from ``src_spacing`` to ``dst_spacing`` (mm/voxel).

    The zoom factor per axis is ``src / dst``: a source voxel that is coarser than
    the target (src > dst) is upsampled (factor > 1), a finer one downsampled.

    Parameters
    ----------
    volume : np.ndarray
        3D array indexed ``[x, y, z]`` (matching ``nib.dataobj`` order here).
    src_spacing, dst_spacing : (float, float, float)
        Physical spacing (mm) along each axis, same axis order as ``volume``.
    order : int
        Spline interpolation order. Use ``1`` (linear) for CT intensities and
        ``0`` (nearest) for integer label maps so no fractional labels appear.
    """
    assert volume.ndim == 3, volume.shape
    assert all(s > 0 for s in src_spacing), src_spacing
    assert all(s > 0 for s in dst_spacing), dst_spacing

    factors = tuple(src / dst for src, dst in zip(src_spacing, dst_spacing))
    # grid_mode=False keeps zoom's default sampling; matches the modest factors
    # here and the baseline (non-anti-aliased) resize it is A/B'd against.
    return zoom(volume, factors, order=order)


def center_crop_pad(slice2d: np.ndarray, shape: tuple[int, int],
                    pad_value: int = 0) -> np.ndarray:
    """Center-crop or symmetrically pad a 2D slice to exactly ``shape``.

    Used instead of ``resize`` after resampling: crop/pad changes the pixel count
    without touching mm/pixel, so the spacing set by ``resample_volume`` is
    preserved (a resize would re-scale it back to the per-patient field of view).
    Target organs are central mediastinal structures, so center cropping trims
    only outer body on wide field-of-view scans. ``pad_value=0`` is background/air
    for both the windowed CT and the label map.
    """
    out = np.full(shape, pad_value, dtype=slice2d.dtype)
    for axis, target in enumerate(shape):
        src = slice2d.shape[axis]
        if src > target:  # crop centered
            start = (src - target) // 2
            slice2d = np.take(slice2d, range(start, start + target), axis=axis)
    # slice2d is now <= target on every axis; place it centered into `out`.
    offsets = [(t - s) // 2 for t, s in zip(shape, slice2d.shape)]
    out[offsets[0]:offsets[0] + slice2d.shape[0],
        offsets[1]:offsets[1] + slice2d.shape[1]] = slice2d
    return out



# --------------------------------------------------------------------------- #
# Ideas left to implement

# Multi-window channels (mediastinal + lung). Directly fixes the trade-off you documented: the mediastinal window maps the trachea lumen to 0. Stacking the two windows as input channels is well-established for CT organ seg and is cheap. Requires a 1→2 channel change on the net's first conv.
# Foreground/body crop + drop empty slices. Attacks the core difficulty (class imbalance — esophagus is tiny, most axial slices have no organ). Pure SliceConfig work.
# Z-score normalization after clipping (the nnU-Net normalization, vs your current linear→[0,255]). One A/B; windowing already does most of the work, so expect small.