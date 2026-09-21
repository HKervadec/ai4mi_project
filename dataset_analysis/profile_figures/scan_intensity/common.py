"""
Shared data helpers and drawing helpers for the scan intensity figures.

Every figure script in this folder reads the exact 1 HU histograms written by
tools/dataset_profile.py and/or the raw CT volumes, and writes one PNG to
figures/profile/scan_intensity/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from dataset_profile import binned_pcts, hist_stats  # noqa: E402
from profile_figures.common import out_subdir  # noqa: E402

DATA_DIR = Path("data/segthor_part1/train")
PROFILE_DIR = Path("figures/profile")
OUT_NAME = "scan_intensity"
SCAN_MIN_HU = -1000  # lowest value stored in every scan
REGION_NAMES = ["outside", "outer third", "middle third", "central third"]
SEE_THROUGH_HU = (-1000, -900)  # display only: opacity rises from nearly 0 to full across this range


def output_path(profile_dir: Path, filename: str) -> Path:
    """Path of a figure file in this concept's output folder, created if needed."""
    return out_subdir(profile_dir, OUT_NAME) / filename


def patient_name(patient: str) -> str:
    """Turn "Patient_01" into "patient 01"."""
    return patient.replace("Patient_", "patient ")


def patients_in(hists: dict) -> list[str]:
    """Sorted patient ids present in a load_tables() histogram dict."""
    return sorted({patient for patient, _ in hists})


def pool_histograms(parts: list[tuple[np.ndarray, int]]) -> tuple[np.ndarray, int]:
    """Add 1 HU histograms that may start at different values.

    Args:
        parts: (counts, start) pairs, counts[i] = number of values equal to start + i.

    Returns:
        (counts, start) of the summed histogram.
    """
    start = min(s for _, s in parts)
    stop = max(s + len(c) for c, s in parts)
    total = np.zeros(stop - start, dtype=np.int64)
    for counts, s in parts:
        total[s - start : s - start + len(counts)] += counts
    return total, start


def nnunet_ct_normalization(hists: dict) -> dict:
    """nnU-Net's CT normalization values, recomputed from its own foreground sample.

    nnU-Net pools a random sample of labeled voxels from every training case,
    clips to that pool's 0.5th/99.5th percentiles and then subtracts its mean and
    divides by its standard deviation.

    Args:
        hists: {(patient, group): (counts, start)} from load_tables(), with an "nnunet sample" group.

    Returns:
        Dict with "clip_low", "clip_high", "mean" and "std" in HU.
    """
    pooled = pool_histograms([h for (_, group), h in hists.items() if group == "nnunet sample"])
    stats = hist_stats(*pooled)
    return {"clip_low": stats["p0.5"], "clip_high": stats["p99.5"], "mean": stats["mean"], "std": stats["std"]}


def normalize(hu: np.ndarray, norm: dict) -> np.ndarray:
    """Apply nnU-Net's CT normalization (clip, subtract mean, divide by std)."""
    return (np.clip(hu, norm["clip_low"], norm["clip_high"]) - norm["mean"]) / norm["std"]


def range_percent(counts: np.ndarray, start: int, edges: np.ndarray) -> np.ndarray:
    """Percentage of values in each HU range [edges[i], edges[i+1]).

    Values below edges[0] are counted in the first range and values at or above
    edges[-1] in the last, so the result always sums to 100.

    Args:
        counts: 1 HU histogram counts.
        start: HU value of counts[0].
        edges: Increasing range edges.

    Returns:
        Array of len(edges) - 1 percentages.
    """
    pct = binned_pcts(counts, start, edges)
    pct[1] += pct[0]
    pct[-2] += pct[-1]
    return pct[1:-1]


def load_ct(data_dir: Path, patient: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one CT volume.

    Args:
        data_dir: Folder with one Patient_XX folder per patient.
        patient: Patient id, e.g. "Patient_01".

    Returns:
        HU volume (x, y, z; slice 0 is the lowest slice) and voxel spacing in mm.
    """
    img = nib.load(str(data_dir / patient / f"{patient}.nii.gz"))
    return np.asanyarray(img.dataobj), np.array(img.header.get_zooms()[:3], dtype=float)


def slice_regions(ct_slice: np.ndarray, spacing: tuple[float, float]) -> np.ndarray:
    """Split one axial slice into outside and outer/middle/central thirds of its scanned area.

    The scanned area is the largest connected patch of values above -1000, with
    holes filled. Inside it, each pixel's distance to the area's edge is divided by
    the largest such distance in the slice and cut into thirds.

    Args:
        ct_slice: 2D HU slice.
        spacing: Pixel spacing (x, y) in mm.

    Returns:
        int8 array: 0 outside, 1 outer third, 2 middle third, 3 central third.
    """
    pieces, n = label(ct_slice > SCAN_MIN_HU)
    regions = np.zeros(ct_slice.shape, dtype=np.int8)
    if n == 0:
        return regions
    area = binary_fill_holes(pieces == np.bincount(pieces.ravel())[1:].argmax() + 1)
    depth = distance_transform_edt(area, sampling=spacing)
    relative = depth / depth.max()
    regions[area] = np.minimum(relative[area] * 3, 2).astype(np.int8) + 1
    return regions


def see_through_alpha(hu: np.ndarray) -> np.ndarray:
    """Display opacity for 3D slice pixels: nearly transparent at -1000, fully opaque from -900 HU up."""
    return np.interp(hu, SEE_THROUGH_HU, (0.04, 0.95))


def clip_lines(ax, norm: dict, label: bool = False) -> None:
    """Dashed vertical lines at nnU-Net's clip values, optionally labeled above the axes."""
    for value in (norm["clip_low"], norm["clip_high"]):
        ax.axvline(value, color="#555555", ls="--", lw=1, zorder=1)
        if label:
            text = f" nnU-Net clip {value:.0f}".replace("-", "−")
            ax.text(value, 1.01, text, transform=ax.get_xaxis_transform(), fontsize=11, color="#555555", va="bottom")


def stack_indices(n_slices: int, n: int = 8) -> np.ndarray:
    """n evenly spaced slice indices from the lowest to the highest slice."""
    return np.linspace(0, n_slices - 1, n).round().astype(int)


def draw_slice_stack(ax, colors: list[np.ndarray], spacing: np.ndarray, indices: np.ndarray, step: int = 4) -> None:
    """Draw axial slices as flat colored planes stacked at their height in a 3D axis.

    Args:
        ax: Axis created with projection="3d".
        colors: One RGBA image (x, y, 4) per slice, at full resolution.
        spacing: Voxel spacing (x, y, z) in mm.
        indices: Slice index of each image, used for its height.
        step: Draw every step-th pixel in x and y (display only).
    """
    nx, ny = colors[0].shape[:2]
    xx, yy = np.meshgrid(np.arange(0, nx, step) * spacing[0], np.arange(0, ny, step) * spacing[1], indexing="ij")
    for rgba, index in zip(colors, indices):
        zz = np.full_like(xx, index * spacing[2])
        ax.plot_surface(
            xx, yy, zz, facecolors=rgba[::step, ::step], rstride=1, cstride=1, shade=False, lw=0, antialiased=False
        )
    height = (indices[-1] - indices[0]) * spacing[2]
    ax.set_box_aspect((nx * spacing[0], ny * spacing[1], height * 1.4))
    ax.view_init(elev=22, azim=-55)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("white")
        axis.set_major_locator(plt.MaxNLocator(3))
    ax.grid(False)
    ax.set_xlabel("x (mm)", labelpad=8)
    ax.set_ylabel("y (mm)", labelpad=8)
    ax.set_zlabel("height above lowest slice (mm)", labelpad=8)
