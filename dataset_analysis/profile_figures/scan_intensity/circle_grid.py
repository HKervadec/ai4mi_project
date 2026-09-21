#!/usr/bin/env python3
"""
Circle grid of each scan's intensity distribution across HU bands.

Each row is one patient's scan, each column an HU band. Circle size and color
(log scale) both show the percentage of that scan's voxels in the band, from the
exact 1 HU histograms (every voxel, no sampling). Voxels of exactly -1000 get
their own first column, then -999..-901, then 100 HU steps up to 1500..1599, and
everything from 1600 up is one final column. nnU-Net's clip values are placed
inside the column that contains them.

Usage:
    python dataset_analysis/profile_figures/scan_intensity/circle_grid.py --profile-dir figures/profile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from dataset_profile import load_tables  # noqa: E402
from profile_figures.common import apply_ticks_style, title_block  # noqa: E402
from profile_figures.scan_intensity.common import (
    output_path,  # noqa: E402
    PROFILE_DIR,
    nnunet_ct_normalization,
    patient_name,
    patients_in,
    range_percent,
)

MIN_PCT = 1e-3  # log color/size floor, in percent of a scan's voxels
MAX_PCT = 60  # log color/size ceiling, in percent of a scan's voxels
MIN_MARKER, MAX_MARKER = 6, 520


def band_edges() -> tuple[np.ndarray, list[str]]:
    """HU band edges and labels for this figure's columns.

    Bands: exactly -1000, then -999..-901, then 100 HU steps from -900..-801
    up to 1500..1599, then everything at or above 1600. The last edge is a
    sentinel above 1600 so range_percent folds all higher values into the
    final band.

    Returns:
        Edges (length n_bands + 1) and one label per band.
    """
    steps = list(range(-900, 1600, 100))  # -900, -800, ..., 1500
    edges = np.array([-1000, -999, *range(-900, 1601, 100), 1601])
    labels = ["= −1000", "−999…−901"] + [f"{a}…{a + 99}".replace("-", "−") for a in steps] + ["≥ 1600"]
    return edges, labels


def band_index(edges: np.ndarray, value: float) -> int:
    """Index of the band containing an HU value, clamped to the valid range."""
    return int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, len(edges) - 2))


def column_position(edges: np.ndarray, value: float) -> float:
    """x-axis position of an HU value, interpolated within its band's column.

    Args:
        edges: Band edges from band_edges().
        value: HU value to place.

    Returns:
        x coordinate, where each band occupies [i - 0.5, i + 0.5) around its
        column index i.
    """
    i = band_index(edges, value)
    lo, hi = edges[i], edges[i + 1]
    return (i - 0.5) + (value - lo) / (hi - lo)


def marker_sizes(pct: np.ndarray) -> np.ndarray:
    """Marker sizes on a log scale between MIN_MARKER and MAX_MARKER.

    Args:
        pct: Percentages (may contain zeros or NaN).

    Returns:
        Marker sizes, same shape as pct.
    """
    filled = np.nan_to_num(pct, nan=MIN_PCT)
    scaled = (np.log10(np.clip(filled, MIN_PCT, None)) - np.log10(MIN_PCT)) / (np.log10(MAX_PCT) - np.log10(MIN_PCT))
    return MIN_MARKER + np.clip(scaled, 0, 1) * (MAX_MARKER - MIN_MARKER)


def main():
    """Draw the circle grid figure from the profiled intensity histograms."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-dir", type=Path, default=PROFILE_DIR)
    args = ap.parse_args()

    hists = load_tables(args.profile_dir)["histograms"]
    patients = patients_in(hists)
    names = [patient_name(p) for p in patients]
    norm = nnunet_ct_normalization(hists)

    edges, labels = band_edges()
    pct = np.array([range_percent(*hists[(p, "scan")], edges) for p in patients])
    n_patients, n_bands = pct.shape

    apply_ticks_style()
    xs, ys = np.meshgrid(np.arange(n_bands), np.arange(n_patients)[::-1])
    values = np.where(pct > 0, pct, np.nan)

    fig, ax = plt.subplots(figsize=(17, 11))
    sc = ax.scatter(
        xs.ravel(),
        ys.ravel(),
        s=marker_sizes(values).ravel(),
        c=values.ravel(),
        cmap="rocket_r",
        norm=LogNorm(MIN_PCT, MAX_PCT),
        edgecolor="0.6",
        lw=0.5,
    )

    for value in (norm["clip_low"], norm["clip_high"]):
        x = column_position(edges, value)
        ax.axvline(x, color="#555555", ls="--", lw=1)
        text = f" nnU-Net clip {value:.0f}".replace("-", "−")
        ax.text(x, n_patients - 0.2, text, fontsize=11, color="#555555")

    ax.set_xticks(range(n_bands))
    ax.set_xticklabels(labels, rotation=90, fontsize=11)
    ax.set_yticks(range(n_patients))
    ax.set_yticklabels(names[::-1])
    ax.set_xlim(-0.7, n_bands - 0.3)
    ax.set_ylim(-0.7, n_patients + 0.3)
    ax.set_xlabel("intensity band (HU)", fontsize=13)
    ax.grid(color="0.93")
    sns.despine(ax=ax, left=True, bottom=True)
    ax.tick_params(length=0)

    cb = fig.colorbar(sc, ax=ax, shrink=0.35, pad=0.01)
    cb.set_label("% of the scan's voxels (size and color)")

    title_block(
        fig,
        "Scan Intensity: Share of Voxels per HU Band",
        "Each row is one scan, each column an intensity band; bigger and darker circles hold more of the scan's voxels.",
    )
    fig.subplots_adjust(left=0.08, right=1.0, top=0.88, bottom=0.14)

    out_path = output_path(args.profile_dir, "circle_grid.png")
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
