#!/usr/bin/env python3
"""
Contact between label pairs as a strip plot. One row per pair of organs (sorted by median shared
border area); every patient is a dot at the area of border the two organs share, split in the two
organs' colors, and a dark bar marks the median. A small open dot means the pair does not touch in
that patient. Rows are named by the first letters of the two organs, each in its own color.
Only pairs of labels that have voxels are drawn. Reads label_pairs.csv and labels.csv written by tools/dataset_profile.py.

Usage:
    python dataset_analysis/profile_figures/label_pairs_strip.py --profile-dir figures/profile
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from plot_style import EARTH_LABEL_COLORS  # noqa: E402
from profile_figures.common import build_arg_parser, out_subdir, title_block  # noqa: E402
from profile_figures.label_correction_example import INK  # noqa: E402
from utils import CLASSES  # noqa: E402


def ordered_pairs(pairs: pd.DataFrame) -> list[tuple[int, int]]:
    """Label pairs from the largest to the smallest median shared border area.

    Args:
        pairs: label_pairs.csv table, one row per patient and pair.

    Returns:
        (label_a, label_b) for every pair; ties keep the order of the labels.
    """
    median = pairs.groupby(["label_a", "label_b"]).shared_face_area_mm2.median().reset_index()
    ordered = median.sort_values("shared_face_area_mm2", ascending=False, kind="stable")
    return [(int(a), int(b)) for a, b in ordered[["label_a", "label_b"]].to_numpy()]


def draw_pair_name(ax, y: float, first: int, second: int) -> None:
    """Write the first letters of two organs, each in its own color, right-aligned left of the axes.

    Args:
        ax: Axes whose figure has been drawn once, so text widths are known.
        y: Row position in data coordinates.
        first: Label of the organ written first.
        second: Label of the organ written second.
    """
    renderer = ax.figure.canvas.get_renderer()
    x = -0.03
    row = ax.transAxes.inverted().transform(ax.transData.transform((0, y)))[1]
    for text, color in ((CLASSES[second][0], EARTH_LABEL_COLORS[second]), (" – ", INK), (CLASSES[first][0], EARTH_LABEL_COLORS[first])):
        label = ax.text(x, row, text, ha="right", va="center", fontsize=11, fontweight="bold", color=color, transform=ax.transAxes)
        x -= label.get_window_extent(renderer).width / ax.get_window_extent(renderer).width


def main():
    """Draw the strip plot of shared border area per label pair."""
    ap = build_arg_parser(__doc__)
    args = ap.parse_args()
    labels = pd.read_csv(args.profile_dir / "labels.csv")
    present = set(labels[labels.voxels > 0].label)
    pairs = pd.read_csv(args.profile_dir / "label_pairs.csv")
    pairs = pairs[pairs.label_a.isin(present) & pairs.label_b.isin(present)]
    order = ordered_pairs(pairs)
    rng = np.random.default_rng(5)

    fig, ax = plt.subplots(figsize=(8.4, 2.6))
    for row, (a, b) in enumerate(order):
        area = pairs[(pairs.label_a == a) & (pairs.label_b == b)].shared_face_area_mm2.to_numpy()
        y = row + rng.uniform(-0.16, 0.16, len(area))
        touching = area > 0
        for x_i, y_i in zip(area[touching], y[touching]):
            ax.plot([x_i], [y_i], ls="none", marker="o", markersize=6.5, fillstyle="left", markerfacecolor=EARTH_LABEL_COLORS[a],
                    markerfacecoloralt=EARTH_LABEL_COLORS[b], markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        ax.scatter(area[~touching], y[~touching], s=18, facecolor="white", edgecolor="#9A9A9A", linewidth=0.9, zorder=3)
        ax.plot([np.median(area)] * 2, [row - 0.3, row + 0.3], color=INK, linewidth=2, zorder=4)
    ax.set(ylim=(len(order) - 0.45, -0.55), xlim=(-60, 1800), yticks=range(len(order)), yticklabels=[""] * len(order))
    ax.set_xlabel("shared border area (mm²)", fontsize=10, fontweight="bold")
    ax.set_xticks(np.arange(0, 1800, 250))
    ax.set_xticks(np.arange(0, 1800, 50), minor=True)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", which="major", color="#DADADA", linewidth=0.9)
    ax.grid(axis="x", which="minor", color="#EEEEEE", linewidth=0.6)
    ax.grid(axis="y", color="#EFEFEF", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    title_block(fig, "Contact Between Label Pairs", "One dot per patient, bar = median.", title_size=15, subtitle_size=10)
    fig.subplots_adjust(top=0.72, bottom=0.22, left=0.09, right=0.99)
    fig.canvas.draw()
    for row, (a, b) in enumerate(order):
        draw_pair_name(ax, row, a, b)

    out = out_subdir(args.profile_dir, "label_pairs") / "label_pairs_strip.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
