#!/usr/bin/env python3
"""
Label bounding boxes in 3D, for whichever labels have voxels in the profiled
dataset (1-3 for the original course release, 1-4 once the aorta annotation
is present).

Writes two figures:
  label_bbox_size_3d.png      one panel per label; every patient's bounding box
                              drawn around one shared center, so only size and
                              shape differ, with the median box drawn bold.
  label_bbox_position_3d.png  all labels in one space; each patient's boxes are
                              shifted so that patient's label 2 center is at 0.

Box size along an axis = (last - first voxel index + 1) x voxel spacing.
Axes follow the image (LPS): +x patient left, +y posterior, +z superior.

Reads labels.csv written by tools/dataset_profile.py.

Usage:
    python dataset_analysis/profile_figures/label_bounding_box.py --profile-dir figures/profile
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profile_figures.common import LABEL_COLORS, apply_ticks_style, build_arg_parser, header, out_subdir  # noqa: E402

COLORS = LABEL_COLORS
AXIS_NAMES = ("left–right, x (mm)", "anterior–posterior, y (mm)", "superior–inferior, z (mm)")
REFERENCE_LABEL = 2
SIZE_LIMITS = [(-100, 100), (-85, 85), (-165, 165)]
POSITION_LIMITS = [(-110, 110), (-90, 140), (-150, 240)]


def label_boxes(labels: pd.DataFrame, label: int, reference: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Bounding boxes of one label for every patient.

    Box edges sit on the outer voxel faces, so each box is `size_mm` long on every axis.

    Args:
        labels: labels.csv table, one row per patient and label.
        label: Label number to take the boxes of.
        reference: If given, shift each patient's box so that patient's center of this label is at 0.

    Returns:
        Box start (n_patients, 3) and box size (n_patients, 3) in mm, rows sorted by patient.
    """
    rows = labels[labels.label == label].sort_values("patient")
    size = rows[[f"{a}_size_mm" for a in "xyz"]].to_numpy()
    first = rows[[f"{a}_min_mm" for a in "xyz"]].to_numpy()
    last = rows[[f"{a}_max_mm" for a in "xyz"]].to_numpy()
    start = first - (size - (last - first)) / 2
    if reference is not None:
        ref = labels[labels.label == reference].sort_values("patient")
        start = start - ref[[f"{a}_center_mm" for a in "xyz"]].to_numpy()
    return start, size


def box_edges(start: np.ndarray, size: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The 12 edges of an axis-aligned box.

    Args:
        start: Corner with the smallest coordinates, shape (3,).
        size: Side lengths, shape (3,).

    Returns:
        12 (point, point) pairs.
    """
    corners = [start + size * np.array(c) for c in itertools.product((0, 1), repeat=3)]
    return [(corners[a], corners[b]) for a, b in itertools.combinations(range(8), 2) if bin(a ^ b).count("1") == 1]


def _box_faces(start: np.ndarray, size: np.ndarray) -> list[np.ndarray]:
    faces = []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for side in (0, 1):
            face = np.tile(start + size * (np.arange(3) == axis) * side, (4, 1))
            face[:, u] += size[u] * np.array([0, 1, 1, 0])
            face[:, v] += size[v] * np.array([0, 0, 1, 1])
            faces.append(face)
    return faces


def draw_box(ax, start, size, color, bold: bool = False) -> None:
    """Draw one box: thin see-through wireframe, or bold edges with a light fill."""
    if bold:
        ax.add_collection3d(Poly3DCollection(_box_faces(start, size), facecolor=color, alpha=0.18, edgecolor="none"))
    ax.add_collection3d(
        Line3DCollection(
            box_edges(start, size), colors=color, alpha=1.0 if bold else 0.33, linewidths=2.4 if bold else 0.9
        )
    )


def style_axes(ax, limits: list[tuple[float, float]], elev: float, azim: float) -> None:
    """Shared limits, true mm proportions, 50 mm grid and axis names."""
    for set_lim, lim in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), limits):
        set_lim(*lim)
    ax.set_box_aspect([b - a for a, b in limits])
    for axis, name in zip((ax.xaxis, ax.yaxis, ax.zaxis), AXIS_NAMES):
        axis.set_major_locator(plt.MultipleLocator(50))
        axis.pane.set_facecolor((0.97, 0.97, 0.97, 1))
        axis.pane.set_edgecolor("#DDDDDD")
        axis._axinfo["grid"]["color"] = "#E6E6E6"
        axis.set_label_text(name)
        axis.labelpad = 10
    ax.view_init(elev=elev, azim=azim)


def line_handles(extra: list[tuple[str, dict]]) -> list:
    return [plt.Line2D([], [], label=name, **style) for name, style in extra]


def size_figure(labels: pd.DataFrame, out: Path, present_labels: list[int]) -> None:
    """One 3D panel per label, all boxes around one center.

    Args:
        labels: labels.csv.
        out: Path to write label_bbox_size_3d.png to.
        present_labels: Labels with at least one voxel in this dataset.
    """
    fig = plt.figure(figsize=(6 * len(present_labels), 8))
    for i, label in enumerate(present_labels):
        ax = fig.add_subplot(1, len(present_labels), i + 1, projection="3d")
        _, sizes = label_boxes(labels, label)
        for size in sizes:
            draw_box(ax, -size / 2, size, COLORS[label])
        median = np.median(sizes, axis=0)
        draw_box(ax, -median / 2, median, COLORS[label], bold=True)
        style_axes(ax, SIZE_LIMITS, elev=18, azim=-60)
        ax.set_title(f"label {label}", fontsize=16, fontweight="bold", color=COLORS[label], loc="left")
    handles = line_handles(
        [
            ("one patient's bounding box", {"color": "#555555", "lw": 0.9, "alpha": 0.5}),
            ("median size on each axis", {"color": "#555555", "lw": 2.4}),
        ]
    )
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    header(
        fig,
        "Bounding Box Size per Label",
        f"One wireframe per patient (n={labels.patient.nunique()}); boxes share a center so size and shape differ; "
        "bold = median box.",
        subtitle_y=0.945,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1, wspace=0.05)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def position_figure(labels: pd.DataFrame, out: Path, present_labels: list[int]) -> None:
    """All labels in one 3D space, placed around each patient's label 2 center.

    Args:
        labels: labels.csv.
        out: Path to write label_bbox_position_3d.png to.
        present_labels: Labels with at least one voxel in this dataset.
    """
    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(projection="3d")
    for label in present_labels:
        starts, sizes = label_boxes(labels, label, reference=REFERENCE_LABEL)
        for start, size in zip(starts, sizes):
            draw_box(ax, start, size, COLORS[label])
        draw_box(ax, np.median(starts, axis=0), np.median(sizes, axis=0), COLORS[label], bold=True)
    style_axes(ax, POSITION_LIMITS, elev=14, azim=-55)
    handles = line_handles(
        [(f"label {label}", {"color": COLORS[label], "lw": 6}) for label in present_labels]
        + [
            ("one patient", {"color": "#555555", "lw": 0.9, "alpha": 0.5}),
            ("median start and size", {"color": "#555555", "lw": 2.4}),
        ]
    )
    fig.legend(handles=handles, loc="lower center", ncol=len(present_labels) + 2, frameon=False)
    header(
        fig,
        "Bounding Box Position Relative to Label 2",
        "One wireframe per patient; color = label; boxes shifted so each patient's label 2 center sits at (0,0,0).",
        subtitle_y=0.945,
    )
    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.08)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    """Draw both 3D bounding-box figures from labels.csv."""
    ap = build_arg_parser(__doc__)
    args = ap.parse_args()

    labels = pd.read_csv(args.profile_dir / "labels.csv")
    labels = labels[labels.voxels > 0]
    # Labels with at least one voxel somewhere in this dataset -- not hardcoded,
    # so the same figure works whether the aorta annotation (label 4) is present.
    present_labels = sorted(int(v) for v in labels.label.unique())
    apply_ticks_style()
    out_dir = out_subdir(args.profile_dir, "label_bounding_box")
    size_figure(labels, out_dir / "label_bbox_size_3d.png", present_labels)
    position_figure(labels, out_dir / "label_bbox_position_3d.png", present_labels)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
