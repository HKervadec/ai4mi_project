"""In-plane field-of-view normalisation.

Crops or pads every slice so it covers the same physical width (target FoV),
so that after resizing to a fixed shape one pixel spans the same number of
millimetres for every patient.
"""

import numpy as np


def fov_target_size(dx: float, target_fov_mm: float) -> int:
    """Number of pixels that span target_fov_mm at in-plane spacing dx."""
    return round(target_fov_mm / dx)


def center_crop_or_pad(arr: np.ndarray, size: tuple[int, int], pad_value) -> np.ndarray:
    """Center-crop or pad the first two axes of arr to size; other axes untouched."""
    out = np.full((*size, *arr.shape[2:]), pad_value, dtype=arr.dtype)
    src, dst = [], []
    for s, t in zip(arr.shape[:2], size):
        if s >= t:  # crop
            s0, d0, n = (s - t) // 2, 0, t
        else:       # pad
            s0, d0, n = 0, (t - s) // 2, s
        src.append(slice(s0, s0 + n))
        dst.append(slice(d0, d0 + n))
    out[dst[0], dst[1]] = arr[src[0], src[1]]
    return out