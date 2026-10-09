"""Post-processing of the stitched 3D predictions (eval.postprocess).

Applied in segpipe.evaluate.evaluate_3d on the native-grid label volume, after
stitching and before scoring. The raw metrics are always kept; the post-processed
ones are stored next to them (summary.json: metrics_3d_post), so one run gives both.
"""

import numpy as np
from scipy.ndimage import (binary_closing, binary_fill_holes, distance_transform_edt,
                           generate_binary_structure, label, minimum)

from segpipe.data import CLASS_NAMES


def _organs(volume: np.ndarray, skip) -> list[int]:
    """Labels of the organs present in `volume`, minus the organ names in `skip`."""
    unknown = set(skip) - set(CLASS_NAMES[1:])
    assert not unknown, f"unknown organ(s) in skip: {sorted(unknown)}. Known: {CLASS_NAMES[1:]}"
    return [int(k) for k in np.unique(volume) if k != 0 and CLASS_NAMES[k] not in skip]


def _bbox(mask: np.ndarray, margin=(0, 0, 0)) -> tuple[slice, ...]:
    """Bounding box of `mask` grown by `margin` voxels per axis, clipped to the volume."""
    box = []
    for axis, m in enumerate(margin):
        nonzero = np.flatnonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != axis)))
        box.append(slice(max(0, nonzero[0] - m), min(mask.shape[axis], nonzero[-1] + 1 + m)))
    return tuple(box)


def largest_cc(volume: np.ndarray, spacing, connectivity: int = 3, min_fraction: float = 0.0,
               max_distance_mm: float | None = None, skip: list[str] = ()) -> np.ndarray:
    """Keep only the largest 3D connected component of every organ; the rest becomes background.

    Each SegTHOR organ is one connected structure, so smaller components are false positives
    (e.g. stray blobs far from the heart, which barely move Dice but blow up HD95).
    connectivity: 1 = 6-, 2 = 18-, 3 = 26-neighbourhood.
    min_fraction: also keep every component at least this fraction of the largest one's size, so a
        prediction broken into big pieces (the thin esophagus often is, along z) keeps its pieces.
    max_distance_mm: also keep every component that comes within this distance (mm) of the largest
        one. Real fragments of a broken organ lie next to it, stray blobs far away, whatever their size.
        A component is kept if it passes min_fraction or max_distance_mm.
    skip: organ names (segpipe.data.CLASS_NAMES) left untouched, e.g. [esophagus].
    """
    structure = generate_binary_structure(3, connectivity)
    out = volume.copy()
    for k in _organs(volume, skip):
        components, n = label(volume == k, structure=structure)
        if n <= 1:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0  # label 0 is everything outside this organ
        keep = sizes >= min_fraction * sizes.max() if min_fraction > 0 else np.zeros(len(sizes), dtype=bool)
        largest = sizes.argmax()
        keep[largest] = True  # the largest is always kept
        if max_distance_mm is not None:
            # every component lies inside the organ's bounding box, so distances computed there are exact
            box = _bbox(components > 0)
            crop = components[box]
            distance = distance_transform_edt(crop != largest, sampling=spacing)
            closest = minimum(distance, labels=crop, index=np.arange(1, n + 1))  # mm, per component
            keep[1:] |= np.asarray(closest) <= max_distance_mm
        out[(components > 0) & ~keep[components]] = 0
    return out


def closing(volume: np.ndarray, spacing, radius_mm: float = 5.0, axis: str = "z",
            skip: list[str] = ()) -> np.ndarray:
    """Morphological closing per organ, to bridge small gaps between fragments of the same organ.

    The 2D model predicts slice by slice, so the thin esophagus often misses a few slices and falls
    apart into fragments along z (which largest_cc then removes). Closing joins pieces at most about
    2 * radius_mm apart. Only background voxels are filled: an organ never grows into another one.
    radius_mm: structuring element radius in mm, converted to voxels per axis with `spacing`.
    axis: "z" closes along z only (a line, bridges missing slices), "3d" with an ellipsoid.
    skip: organ names (segpipe.data.CLASS_NAMES) left untouched.
    """
    assert axis in ("z", "3d"), f"axis must be 'z' or '3d', got {axis!r}"
    radius = [max(0, round(radius_mm / s)) for s in spacing]
    if axis == "z":
        radius[0] = radius[1] = 0
    if not any(radius):
        return volume
    grid = np.ogrid[tuple(slice(-r, r + 1) for r in radius)]
    structure = sum((g / max(r, 1)) ** 2 for g, r in zip(grid, radius)) <= 1
    out = volume.copy()
    for k in _organs(volume, skip):
        mask = volume == k
        # crop with a margin, and pad so the erosion never meets the volume border
        box = _bbox(mask, radius)
        padded = np.pad(mask[box], [(r, r) for r in radius])
        closed = binary_closing(padded, structure=structure)[tuple(slice(r, -r or None) for r in radius)]
        region = out[box]  # a view: writing into it fills `out`
        region[closed & (volume[box] == 0)] = k
    return out


def fill_holes(volume: np.ndarray, spacing, per_slice: bool = True, skip: list[str] = ()) -> np.ndarray:
    """Fill holes inside every organ. Only background voxels are filled, never another organ.

    per_slice: fill holes in each axial slice (holes open along z are filled too); False fills
        only cavities fully enclosed in 3D, which predictions rarely have.
    skip: organ names (segpipe.data.CLASS_NAMES) left untouched.
    """
    out = volume.copy()
    for k in _organs(volume, skip):
        mask = volume == k
        if per_slice:
            filled = np.stack([binary_fill_holes(mask[:, :, z]) for z in range(mask.shape[2])], axis=-1)
        else:
            filled = binary_fill_holes(mask)
        out[filled & (volume == 0)] = k
    return out


# name -> fn(volume, spacing, **params) -> volume
POSTPROCESS: dict = {
    "largest_cc": largest_cc,
    "closing": closing,
    "fill_holes": fill_holes,
}


class Compose:
    def __init__(self, steps):
        self.steps = steps  # [(fn, params)]

    def __call__(self, volume, spacing):
        for fn, params in self.steps:
            volume = fn(volume, spacing, **params)
        return volume


def build_postprocess(items):
    """eval.postprocess: ["largest_cc"] or [{name: largest_cc, connectivity: 1}] -> callable or None."""
    steps = []
    for item in items or []:
        params = {"name": item} if isinstance(item, str) else dict(item)
        name = params.pop("name")
        if name not in POSTPROCESS:
            raise KeyError(f"unknown post-processing '{name}'. Known: {sorted(POSTPROCESS)}")
        steps.append((POSTPROCESS[name], params))
    return Compose(steps) if steps else None
