#!/usr/bin/env python3
"""
One patient's four organs (esophagus, heart, trachea, aorta) on an axial CT slice
(label_example_2D.png) and as 3D surfaces (label_example_3D.png), both drawn from the corrected
ground truth of that patient. The slice is the one where the smallest of the four organs is
largest, so all four are on it.

Usage:
    python dataset_analysis/profile_figures/label_example.py --data-dir data/segthor_part1_corrected/train \
        --patient Patient_18 --out-dir figures/comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from plot_style import EARTH_LABEL_COLORS  # noqa: E402
from profile_figures.label_correction_example import INK, chip, draw_slice, load_patient, shaded, surface  # noqa: E402
from utils import CLASSES  # noqa: E402

ORGANS = (1, 2, 3, 4)
CROP = {"above": 75, "below": 120, "half_width": 135}  # slice crop around the labeled pixels, in pixels
MESH_FIG = (2.6, 4.15)  # size of the 3D figure in inches
MESH_AXES_W = 5.6  # width of its 3D axes in inches; wider than the figure, which draws the surfaces larger
CAPTION_H = 0.5  # room for the caption under the surfaces, in inches
SLICE_TEXT = "#F2F2EE"  # patient and slice, written on the slice
CHIP_OFFSETS = {1: (-85, 38), 2: (-100, -30), 3: (105, -55), 4: (95, 38)}  # name box relative to the organ center, in pixels


def organ_slice(gt: np.ndarray) -> int:
    """Axial slice where the smallest of the four organs is largest.

    Args:
        gt: Corrected label volume (x, y, z).

    Returns:
        Slice index.
    """
    return int(np.argmax(np.min([(gt == k).sum(axis=(0, 1)) for k in ORGANS], axis=0)))


def main():
    """Draw the slice and the surfaces of one patient as two figures."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_part1_corrected/train"))
    ap.add_argument("--patient", default="Patient_18")
    ap.add_argument("--azim", type=float, default=0, help="Horizontal viewing angle of the 3D figure in degrees.")
    ap.add_argument("--elev", type=float, default=6, help="Vertical viewing angle of the 3D figure in degrees.")
    ap.add_argument("--out-dir", type=Path, default=Path("figures/comparison"))
    args = ap.parse_args()

    gt, ct, spacing = load_patient(args.data_dir, args.patient)
    z = organ_slice(gt)
    rows, cols = np.nonzero((gt[:, :, z] > 0).T)
    cy, cx = int(rows.mean()), int(cols.mean())
    crop = (slice(cy - CROP["above"], cy + CROP["below"]), slice(cx - CROP["half_width"], cx + CROP["half_width"]))
    labels = gt[:, :, z].T[crop]

    fig = plt.figure(figsize=(7, 7 * (CROP["above"] + CROP["below"]) * spacing[0] / (2 * CROP["half_width"] * spacing[1])))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    draw_slice(ax, ct[:, :, z].T[crop], [(labels == k, EARTH_LABEL_COLORS[k]) for k in ORGANS], spacing[0] / spacing[1])
    for k in ORGANS:
        rr, cc = np.nonzero(labels == k)
        dx, dy = CHIP_OFFSETS[k]
        chip(ax, CLASSES[k], (cc.mean(), rr.mean()), (cc.mean() + dx, rr.mean() + dy), EARTH_LABEL_COLORS[k])
    ax.text(
        0.985, 0.975, f"{args.patient}  ·  axial slice {z} of {gt.shape[2]}", transform=ax.transAxes, color=SLICE_TEXT, fontsize=13,
        ha="right", va="top", path_effects=[pe.withStroke(linewidth=2.5, foreground="#00000099")],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "label_example_2D.png"
    fig.savefig(out, dpi=150, transparent=True)
    print(f"wrote {out}")

    box = tuple(slice(max(i.min() - 4, 0), i.max() + 5) for i in np.nonzero(gt > 0))
    origin = tuple(b.start * s for b, s in zip(box, spacing))
    end = tuple(o + n * s for o, n, s in zip(origin, gt[box].shape, spacing))
    fig = plt.figure(figsize=MESH_FIG)
    ax = fig.add_axes(
        [(1 - MESH_AXES_W / MESH_FIG[0]) / 2, CAPTION_H / MESH_FIG[1], MESH_AXES_W / MESH_FIG[0], 1 - CAPTION_H / MESH_FIG[1]],
        projection="3d",
    )
    for k in ORGANS:
        triangles = surface(gt[box] == k, spacing, origin)
        ax.add_collection3d(Poly3DCollection(triangles, facecolors=shaded(triangles, EARTH_LABEL_COLORS[k]), edgecolors="none"))
    ax.set_xlim(origin[0], end[0])
    ax.set_ylim(origin[1], end[1])
    ax.set_zlim(origin[2], end[2])
    ax.set_box_aspect(tuple(np.subtract(end, origin)))
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_axis_off()
    fig.text(0.02, 0.02, f"{args.patient}\n3D surfaces of all slices", fontsize=12, color=INK, va="bottom", linespacing=1.3)
    out = args.out_dir / "label_example_3D.png"
    fig.savefig(out, dpi=150, transparent=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
