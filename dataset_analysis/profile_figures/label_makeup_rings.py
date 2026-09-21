#!/usr/bin/env python3
"""
Label make-up as rings. The big ring is every voxel of every patient, split into background and
labeled voxels; a gray panel zooms into the thin labeled sliver and holds one small ring per
release, each split by organ with the organ's share of the labeled voxels written on it. Reads
labels.csv and patients.csv written by tools/dataset_profile.py for each release.

Usage:
    python dataset_analysis/profile_figures/label_makeup_rings.py \
        --profile-dir figures/before/profile figures/profile \
        --names "3 labels (aorta merged)" "4 labels (aorta separate)" --out-dir figures/comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle, Wedge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from plot_style import BACKGROUND_COLOR, EARTH_LABEL_COLORS  # noqa: E402
from profile_figures.common import title_block  # noqa: E402
from profile_figures.label_correction_example import INK  # noqa: E402
from profile_figures.label_correction_per_patient import background_share  # noqa: E402
from utils import CLASSES  # noqa: E402

# Esophagus, aorta, heart, trachea: the old merged label sits where esophagus and aorta end up.
ORDER = (1, 4, 2, 3)
LABELED_COLOR = "#8E8E8E"
INSIDE_MIN_DEGREES = 50


def organ_voxels(labels: pd.DataFrame) -> pd.Series:
    """Voxels of each organ, all patients pooled.

    Args:
        labels: labels.csv of one release.

    Returns:
        Voxel counts indexed by label id, in ORDER, for the organs that have voxels.
    """
    voxels = labels.groupby("label").voxels.sum()
    return voxels.reindex([k for k in ORDER if voxels.get(k, 0) > 0])


def wedge_angles(fractions, start: float = 90.0) -> list[tuple[float, float]]:
    """Angles of consecutive wedges laid out clockwise.

    Args:
        fractions: Share of the ring of each wedge, summing to 1.
        start: Angle in degrees where the first wedge begins (90 is twelve o'clock).

    Returns:
        (theta1, theta2) in degrees for every wedge, as matplotlib's Wedge expects them.
    """
    edges = start - 360 * np.concatenate([[0], np.cumsum(fractions)])
    return [(edges[i + 1], edges[i]) for i in range(len(fractions))]


def draw_organ_ring(ax, center: tuple[float, float], radius: float, width: float, voxels: pd.Series) -> None:
    """Draw one ring split by organ, each wedge labeled with its share of the ring.

    Args:
        ax: Axes whose data units are inches.
        center: Ring center.
        radius: Outer radius.
        width: Ring thickness.
        voxels: Voxels per organ from organ_voxels().
    """
    fractions = voxels / voxels.sum()
    for k, fraction, (theta1, theta2) in zip(voxels.index, fractions, wedge_angles(fractions)):
        ax.add_patch(Wedge(center, radius, theta1, theta2, width=width, facecolor=EARTH_LABEL_COLORS[k], edgecolor="white", linewidth=1.3))
        mid = np.radians((theta1 + theta2) / 2)
        cos, sin = np.cos(mid), np.sin(mid)
        text = f"{100 * fraction:.1f}%"
        if 360 * fraction >= INSIDE_MIN_DEGREES:
            r = radius - width / 2
            ax.text(center[0] + r * cos, center[1] + r * sin, text, ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        else:
            r = radius + 0.06
            ax.text(center[0] + r * cos, center[1] + r * sin, text, ha="left" if cos >= 0 else "right",
                    va="bottom" if sin > 0.3 else ("top" if sin < -0.3 else "center"), fontsize=9.5, fontweight="bold", color=EARTH_LABEL_COLORS[k])


def main():
    """Draw the big all-voxels ring and one organ ring per release."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-dir", type=Path, nargs=2, default=[Path("figures/before/profile"), Path("figures/profile")],
                    help="Profile folders of the release before and the release after the correction.")
    ap.add_argument("--names", nargs=2, default=["3 labels (aorta merged)", "4 labels (aorta separate)"])
    ap.add_argument("--out-dir", type=Path, default=Path("figures/comparison"))
    args = ap.parse_args()

    labels = [pd.read_csv(d / "labels.csv") for d in args.profile_dir]
    patients = [pd.read_csv(d / "patients.csv") for d in args.profile_dir]
    voxels = [organ_voxels(t) for t in labels]
    background = background_share(patients[0], labels[0])
    labeled = 100 - background
    total = patients[0].voxels.sum()

    width, height = 9.0, 4.7
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    title_block(fig, "Label Make-Up", f"All voxels of the {len(patients[0])} patients ({total / 1e6:,.0f} million), and the labeled voxels by organ.")

    big = (1.5, 2.1)
    half = 180 * labeled / 100
    ax.add_patch(Wedge(big, 1.3, half, 360 - half, width=0.44, facecolor=BACKGROUND_COLOR, edgecolor="none"))
    ax.add_patch(Wedge(big, 1.3, -half, half, width=0.44, facecolor=LABELED_COLOR, edgecolor="none"))
    ax.text(big[0], big[1] + 0.1, f"{background:.2f}%", ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
    ax.text(big[0], big[1] - 0.13, "background", ha="center", fontsize=10, color=INK)
    ax.text(big[0], big[1] - 0.36, f"{labeled:.2f}% labeled", ha="center", fontsize=8.8, color=INK)

    left, right, bottom, top = 3.35, 8.85, 0.5, 3.65
    ax.add_patch(FancyBboxPatch((left, bottom), right - left, top - bottom, boxstyle="round,pad=0,rounding_size=.12", facecolor="#F4F4F4", edgecolor="#DDDDDD", linewidth=1, zorder=0))
    tip = (big[0] + 1.3 * np.cos(np.radians(half)), big[1] + 1.3 * np.sin(np.radians(half)))
    for y, corner in ((tip[1], top), (2 * big[1] - tip[1], bottom)):
        ax.plot([tip[0], left], [y, corner], color="#D0D0D0", linewidth=1, zorder=0)
    for name, v, x in zip(args.names, voxels, (4.65, 7.55)):
        center = (x, 2.1)
        draw_organ_ring(ax, center, 0.95, 0.42, v)
        ax.text(*center, f"{v.sum() / 1e6:.1f} M", ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)
        ax.text(center[0], center[1] - 0.05, "labeled voxels", ha="center", va="top", fontsize=8, color=INK)
        ax.text(x, bottom + 0.22, name, ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)

    x = 3.6
    for k in [k for k in ORDER if any(k in v.index for v in voxels)]:
        ax.add_patch(Rectangle((x, 0.13), 0.15, 0.15, facecolor=EARTH_LABEL_COLORS[k], edgecolor="none"))
        ax.text(x + 0.22, 0.205, CLASSES[k], fontsize=10, fontweight="bold", color=EARTH_LABEL_COLORS[k], va="center")
        x += 1.3

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "label_makeup_rings.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
