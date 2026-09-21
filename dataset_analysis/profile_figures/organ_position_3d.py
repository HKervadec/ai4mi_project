#!/usr/bin/env python3
"""
Organ position and extent in 3D, one panel per organ and one with all organs together. Every panel
uses the same axes, positions relative to each patient's heart center. In an organ panel every
patient has a thin box (the bounding box of the label) and a dot at the organ's center; the bold box
is the median box. The last panel has the median boxes of all organs, lightly filled, and the median
center of each organ. Reads labels.csv written by tools/dataset_profile.py.

Usage:
    python dataset_analysis/profile_figures/organ_position_3d.py --profile-dir figures/profile
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from plot_style import EARTH_LABEL_COLORS  # noqa: E402
from profile_figures.common import build_arg_parser, out_subdir, title_block  # noqa: E402
from profile_figures.label_bounding_box import REFERENCE_LABEL, _box_faces, box_edges, label_boxes  # noqa: E402
from profile_figures.label_makeup_rings import ORDER  # noqa: E402
from utils import CLASSES  # noqa: E402

LIMITS = [(-130, 130), (-110, 130), (-160, 240)]
BOX_ASPECT = (1, 1.2, 1.7)
AXIS_NAMES = ("left–right (mm)", "anterior–posterior (mm)", "superior–inferior (mm)")
INK = "#333333"


def organ_centers(labels: pd.DataFrame, label: int) -> np.ndarray:
    """Center of an organ for every patient, relative to that patient's heart center.

    Args:
        labels: labels.csv table, one row per patient and label.
        label: Label number of the organ.

    Returns:
        Centers in mm, shape (n_patients, 3), rows sorted by patient.
    """
    columns = [f"{a}_center_mm" for a in "xyz"]
    rows = labels[labels.label == label].sort_values("patient")
    reference = labels[labels.label == REFERENCE_LABEL].sort_values("patient")
    return rows[columns].to_numpy() - reference[columns].to_numpy()


def axis_angle(ax, start: tuple[float, float, float], end: tuple[float, float, float]) -> float:
    """On-screen angle of a line in a 3D axes, folded so text along it is never upside down.

    Args:
        ax: 3D axes with its view set and drawn once.
        start: First point in data coordinates.
        end: Second point in data coordinates.

    Returns:
        Angle in degrees within [-90, 90].
    """
    points = [ax.transData.transform(proj3d.proj_transform(*p, ax.get_proj())[:2]) for p in (start, end)]
    angle = np.degrees(np.arctan2(points[1][1] - points[0][1], points[1][0] - points[0][0]))
    return angle - 180 if angle > 90 else (angle + 180 if angle < -90 else angle)


def style_axes(ax) -> None:
    """Shared limits and proportions, a grid line every 50 mm, numbers every 100 mm, axis names."""
    ax.computed_zorder = False
    for set_lim, lim in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), LIMITS):
        set_lim(*lim)
    ax.set_box_aspect(BOX_ASPECT)
    for axis, name in zip((ax.xaxis, ax.yaxis, ax.zaxis), AXIS_NAMES):
        axis.set_major_locator(MultipleLocator(50))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}" if v % 100 == 0 else ""))
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("#DDDDDD")
        axis.set_label_text(name, fontsize=8.5, fontweight="bold")
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.xaxis.labelpad = ax.yaxis.labelpad = -6
    ax.zaxis.labelpad = -2
    ax.grid(True, color="#D8D8D8", linewidth=0.7)
    ax.view_init(elev=14, azim=-52)
    ax.tick_params(labelsize=8, pad=-2)


def align_axis_labels(ax) -> None:
    """Rotate the left-right and anterior-posterior labels to run along their axes."""
    (x0, x1), (y0, y1), (z0, _) = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    ax.xaxis.label.set_rotation(axis_angle(ax, (x0, y0, z0), (x1, y0, z0)))
    ax.yaxis.label.set_rotation(axis_angle(ax, (x1, y0, z0), (x1, y1, z0)))


def draw_organ(ax, labels: pd.DataFrame, label: int) -> None:
    """Every patient's box and center of one organ, and the median box in bold.

    Args:
        ax: 3D axes.
        labels: labels.csv table.
        label: Label number of the organ.
    """
    color = EARTH_LABEL_COLORS[label]
    starts, sizes = label_boxes(labels, label, reference=REFERENCE_LABEL)
    edges = [edge for start, size in zip(starts, sizes) for edge in box_edges(start, size)]
    ax.add_collection3d(Line3DCollection(edges, colors=color, linewidths=0.8, alpha=0.4, zorder=1))
    ax.add_collection3d(Line3DCollection(box_edges(np.median(starts, axis=0), np.median(sizes, axis=0)), colors=color, linewidths=2.1, zorder=2))
    x, y, z = organ_centers(labels, label).T
    ax.scatter(x, y, z, s=20, color=color, edgecolor="white", linewidth=0.8, depthshade=False, zorder=5)


def draw_all_organs(ax, labels: pd.DataFrame, present: list[int]) -> None:
    """The median box of every organ, lightly filled, with the median center of each.

    Args:
        ax: 3D axes.
        labels: labels.csv table.
        present: Labels to draw.
    """
    for label in present:
        color = EARTH_LABEL_COLORS[label]
        starts, sizes = label_boxes(labels, label, reference=REFERENCE_LABEL)
        start, size = np.median(starts, axis=0), np.median(sizes, axis=0)
        ax.add_collection3d(Poly3DCollection(_box_faces(start, size), facecolor=color, alpha=0.07, edgecolor="none", zorder=1))
        ax.add_collection3d(Line3DCollection(box_edges(start, size), colors=color, linewidths=1.3, zorder=2))
        x, y, z = np.median(organ_centers(labels, label), axis=0)
        ax.scatter([x], [y], [z], s=30, color=color, edgecolor="white", linewidth=1.0, depthshade=False, zorder=5)


def main():
    """Draw one 3D panel per organ and a last one with all organs."""
    ap = build_arg_parser(__doc__)
    args = ap.parse_args()
    labels = pd.read_csv(args.profile_dir / "labels.csv")
    labels = labels[labels.voxels > 0]
    present = [k for k in ORDER if k in set(labels.label)]

    fig = plt.figure(figsize=(3.44 * (len(present) + 1), 5.6))
    title_block(fig, "Organ Position and Extent in 3D", "One box per patient, bold box = median, dot = organ center; positions relative to the heart center.")
    axes = []
    for i in range(len(present) + 1):
        ax = fig.add_subplot(1, len(present) + 1, i + 1, projection="3d")
        if i < len(present):
            draw_organ(ax, labels, present[i])
            ax.set_title(CLASSES[present[i]], fontsize=15, fontweight="bold", color=EARTH_LABEL_COLORS[present[i]], pad=-4)
        else:
            draw_all_organs(ax, labels, present)
            ax.set_title("all organs", fontsize=15, fontweight="bold", color=INK, pad=-4)
        style_axes(ax)
        axes.append(ax)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.86, bottom=0.02, wspace=-0.02)
    fig.canvas.draw()
    for ax in axes:
        align_axis_labels(ax)

    out = out_subdir(args.profile_dir, "organ_position_3d") / "organ_position_3d.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
