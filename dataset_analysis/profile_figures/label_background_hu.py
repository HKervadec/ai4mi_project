#!/usr/bin/env python3
"""
Small figure of the HU of the background against the HU of all labeled voxels (the four organs
together), every patient pooled. Each curve is scaled to its own peak, in 10 HU bins, so the
shapes can be compared although the background has far more voxels. The median HU of each group is written in the legend. Reads intensity_histograms.npz written by tools/dataset_profile.py.

Usage:
    python dataset_analysis/profile_figures/label_background_hu.py --profile-dir figures/profile
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from dataset_profile import hist_stats, load_tables  # noqa: E402
from plot_style import BACKGROUND_COLOR  # noqa: E402
from profile_figures.common import apply_ticks_style, build_arg_parser, out_subdir  # noqa: E402
from profile_figures.label_intensity import binned_counts  # noqa: E402
from profile_figures.label_makeup_rings import LABELED_COLOR  # noqa: E402
from profile_figures.scan_intensity.common import pool_histograms  # noqa: E402

HU_RANGE = (-1050, 550)
BIN_HU = 10
GROUPS = (
    ("background", "background", BACKGROUND_COLOR),
    ("all labels", "labeled voxels", LABELED_COLOR),
)
OUTLINE_DARKEN = 0.6  # outline color = fill color times this


def pooled_histogram(hists: dict, group: str) -> tuple[np.ndarray, int]:
    """Add one group's 1 HU histograms over all patients.

    Args:
        hists: {(patient, group): (counts, start)} from load_tables().
        group: Group name, e.g. "background".

    Returns:
        (counts, start) of the summed histogram.
    """
    return pool_histograms([hist for (_, g), hist in hists.items() if g == group])


def main():
    """Draw the background against labeled HU figure from intensity_histograms.npz."""
    args = build_arg_parser(__doc__).parse_args()
    hists = load_tables(args.profile_dir)["histograms"]

    apply_ticks_style()
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    for group, name, color in GROUPS:
        counts, start = pooled_histogram(hists, group)
        x, binned = binned_counts(counts, start, *HU_RANGE, BIN_HU)
        median = hist_stats(counts, start)["median"]
        share = binned / binned.max()
        outline = tuple(OUTLINE_DARKEN * c for c in to_rgb(color))
        ax.fill_between(x, share, step="mid", color=color, alpha=0.6, lw=0)
        ax.step(
            x,
            share,
            where="mid",
            color=outline,
            lw=1,
            label=f"{name} (median {median:.0f} HU)",
        )
    ax.set_xlim(*HU_RANGE)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.set_xlabel("HU")
    ax.legend(
        loc="upper left", bbox_to_anchor=(0.12, 1), fontsize=8.5, handlelength=1.2
    )
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    out = (
        out_subdir(args.profile_dir, "label_background_hu") / "label_background_hu.png"
    )
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
