"""Post-processing of the stitched 3D predictions (eval.postprocess).

Applied in segpipe.evaluate.evaluate_3d on the native-grid label volume, after
stitching and before scoring. The raw metrics are always kept; the post-processed
ones are stored next to them (summary.json: metrics_3d_post), so one run gives both.
"""

import numpy as np
from scipy.ndimage import generate_binary_structure, label

from segpipe.data import CLASS_NAMES


def largest_cc(volume: np.ndarray, spacing, connectivity: int = 3, min_fraction: float = 0.0,
               skip: list[str] = ()) -> np.ndarray:
    """Keep only the largest 3D connected component of every organ; the rest becomes background.

    Each SegTHOR organ is one connected structure, so smaller components are false positives
    (e.g. stray blobs far from the heart, which barely move Dice but blow up HD95).
    connectivity: 1 = 6-, 2 = 18-, 3 = 26-neighbourhood.
    min_fraction: also keep every component at least this fraction of the largest one's size, so a
        prediction broken into big pieces (the thin esophagus often is, along z) keeps its pieces.
    skip: organ names (segpipe.data.CLASS_NAMES) left untouched, e.g. [esophagus].
    """
    unknown = set(skip) - set(CLASS_NAMES[1:])
    assert not unknown, f"unknown organ(s) in skip: {sorted(unknown)}. Known: {CLASS_NAMES[1:]}"
    structure = generate_binary_structure(3, connectivity)
    out = volume.copy()
    for k in np.unique(volume):
        if k == 0 or CLASS_NAMES[k] in skip:
            continue
        components, n = label(volume == k, structure=structure)
        if n <= 1:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0  # label 0 is everything outside this organ
        keep = sizes >= min_fraction * sizes.max() if min_fraction > 0 else np.zeros(len(sizes), dtype=bool)
        keep[sizes.argmax()] = True  # the largest is always kept
        out[(components > 0) & ~keep[components]] = 0
    return out


# name -> fn(volume, spacing, **params) -> volume
POSTPROCESS: dict = {
    "largest_cc": largest_cc,
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
