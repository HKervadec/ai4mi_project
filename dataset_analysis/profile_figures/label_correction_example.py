#!/usr/bin/env python3
"""
One patient's esophagus label before and after the aorta correction: the same axial CT
slice and the same 3D surfaces under both label sets, with the connected pieces of the
esophagus label told apart (face connectivity, 3D).

Before, the aorta is part of the esophagus label, so the label is two pieces (the larger
one is the aorta); after, the esophagus and the aorta are one piece each. Writes the pair side by
side (label_correction_example.png) and each half on its own (..._before.png, ..._after.png).

Usage:
    python dataset_analysis/profile_figures/label_correction_example.py \
        --before-data-dir data/segthor_part1/train --data-dir data/segthor_part1_corrected/train \
        --patient Patient_02 --out-dir figures/comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter, generate_binary_structure, label
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from plot_style import EARTH_LABEL_COLORS, tint  # noqa: E402
from utils import CLASSES  # noqa: E402

ESOPHAGUS, AORTA = 1, 4
FACE = generate_binary_structure(3, 1)
PIECE_COLORS = [EARTH_LABEL_COLORS[ESOPHAGUS], tint(EARTH_LABEL_COLORS[ESOPHAGUS], 0.5)]  # largest piece first
INK = "#333333"
WINDOW = (-160, 240)  # CT window in HU
CROP_HALF = (75, 55)  # half width and height of the slice crop, in pixels
MESH_STEP = 2  # in-plane voxel step of the surface meshes
MESH_W = 3.6  # width of the 3D axes in inches; it is wider than its column, which draws the surfaces larger
VIEW = {"elev": 8, "azim": 155}
GROUP_W, FIG_H, ARROW_GAP = 5.05, 2.8, 0.8  # inches: one before or after group, figure height, space for the arrow
PANEL_H, PANEL_BOTTOM = 2.7, 0.05  # height of the slice panel and its distance from the figure bottom, in inches
LIGHT = np.array([-0.35, -0.6, 0.7]) / np.linalg.norm([-0.35, -0.6, 0.7])


def example_slice(gt: np.ndarray) -> int:
    """Axial slice where the esophagus and the aorta are both largest.

    Args:
        gt: Corrected label volume (x, y, z).

    Returns:
        Index of the slice maximizing the smaller of the two label areas.
    """
    both = np.minimum((gt == ESOPHAGUS).sum(axis=(0, 1)), (gt == AORTA).sum(axis=(0, 1)))
    return int(np.argmax(both))


def ranked_pieces(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Face-connected pieces of a mask, numbered from the largest.

    Args:
        mask: Boolean mask (x, y, z).

    Returns:
        Integer volume in which piece 1 is the largest, 2 the next, and so on (0 outside), and
        the number of pieces.
    """
    pieces, n = label(mask, FACE)
    order = np.argsort(-np.bincount(pieces.ravel())[1:], kind="stable")
    rank = np.zeros(n + 1, dtype=int)
    rank[order + 1] = np.arange(1, n + 1)
    return rank[pieces], n


def surface(mask: np.ndarray, spacing: tuple[float, float, float], origin: tuple[float, float, float]) -> np.ndarray:
    """Triangles of a mask's surface in millimeters.

    Args:
        mask: Boolean mask (x, y, z).
        spacing: Voxel size in mm.
        origin: Position of the mask's first voxel in mm.

    Returns:
        Array (triangles, 3 corners, 3 coordinates).
    """
    smooth = gaussian_filter(mask.astype(np.float32)[::MESH_STEP, ::MESH_STEP], 0.8)
    verts, faces, _, _ = marching_cubes(smooth, 0.5, spacing=(spacing[0] * MESH_STEP, spacing[1] * MESH_STEP, spacing[2]))
    return verts[faces] + np.asarray(origin)


def shaded(triangles: np.ndarray, color) -> np.ndarray:
    """Face colors that darken triangles turned away from the light.

    Args:
        triangles: Array (triangles, 3 corners, 3 coordinates).
        color: Base color.

    Returns:
        RGBA array, one row per triangle.
    """
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    light = 0.5 + 0.5 * np.abs(normals @ LIGHT)
    rgb = np.array(to_rgb(color))[None] * light[:, None] + (1 - light[:, None]) * 0.12
    return np.c_[np.clip(rgb, 0, 1), np.ones(len(rgb))]


def chip(ax, text: str, xy, xytext, color) -> None:
    """Organ name in a colored box with a line to a point of the slice.

    Args:
        ax: Slice axes.
        text: Text of the box.
        xy: Point the line ends at.
        xytext: Center of the box.
        color: Box and line color.
    """
    dark = sum(to_rgb(color)) / 3 < 0.55
    ax.annotate(
        text, xy, xytext, color="white" if dark else INK, fontsize=13, fontweight="bold", ha="center", va="center",
        arrowprops={"arrowstyle": "-", "color": color, "lw": 1.6}, bbox={"boxstyle": "round,pad=0.28", "fc": color, "ec": "none"},
    )


def draw_slice(ax, ct: np.ndarray, layers: list[tuple[np.ndarray, object]], aspect: float) -> None:
    """CT slice with filled, outlined label masks on top.

    Args:
        ax: Axes to draw on.
        ct: CT slice in HU, rows anterior to posterior.
        layers: (mask, color) pairs, drawn in order.
        aspect: Pixel height over width.
    """
    ax.imshow(np.clip((ct - WINDOW[0]) / (WINDOW[1] - WINDOW[0]), 0, 1), cmap="gray", aspect=aspect)
    for mask, color in layers:
        fill = np.zeros(mask.shape + (4,))
        fill[mask] = (*to_rgb(color), 0.55)
        ax.imshow(fill, aspect=aspect, interpolation="nearest")
        ax.contour(gaussian_filter(mask.astype(float), 0.7), levels=[0.5], colors=[color], linewidths=1.8)
    ax.set_axis_off()


def load_patient(data_dir: Path, patient: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Labels, CT and voxel size of one patient.

    Args:
        data_dir: Folder with one Patient_* folder per patient.
        patient: Patient folder name.

    Returns:
        Label volume, CT volume and voxel size in mm.
    """
    gt = nib.load(str(data_dir / patient / "GT.nii.gz"))
    ct = nib.load(str(data_dir / patient / f"{patient}.nii.gz"))
    return np.asarray(gt.dataobj), np.asarray(ct.dataobj).astype(np.float32), tuple(float(v) for v in gt.header.get_zooms()[:3])


class Scene(NamedTuple):
    """What both groups share: the CT slice and the frame of the 3D view."""

    ct: np.ndarray
    aspect: float
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    end: tuple[float, float, float]


class Group(NamedTuple):
    """One side of the comparison: what is drawn on the slice and as surfaces, and its text."""

    info: str  # patient and slice, written in the corner of the slice
    layers: list[tuple[np.ndarray, object]]  # slice masks with their color, in drawing order
    chips: list[tuple[str, tuple[float, float], tuple[float, float], object]]  # text, point, box center, color
    meshes: list[tuple[np.ndarray, object]]  # 3D masks with their color


def draw_group(fig, left: float, group: Group, scene: Scene) -> None:
    """Draw a slice with its labels and patient info, and the 3D surfaces, into a column of the figure.

    Args:
        fig: Figure to draw into.
        left: Left edge of the column in inches.
        group: What to draw.
        scene: The slice and the 3D frame.
    """
    width, height = fig.get_size_inches()
    slice_w = PANEL_H * 2 * CROP_HALF[0] / (2 * CROP_HALF[1] * scene.aspect)
    slice_ax = fig.add_axes([left / width, PANEL_BOTTOM / height, slice_w / width, PANEL_H / height])
    draw_slice(slice_ax, scene.ct, group.layers, scene.aspect)
    for text, xy, xytext, color in group.chips:
        chip(slice_ax, text, xy, xytext, color)
    slice_ax.text(
        0.02, 0.03, group.info, transform=slice_ax.transAxes, color="white", fontsize=11, ha="left", va="bottom", linespacing=1.3,
        bbox={"boxstyle": "round,pad=0.3", "fc": "black", "ec": "none", "alpha": 0.55},
    )
    center = left + slice_w + (GROUP_W - slice_w) / 2
    mesh_ax = fig.add_axes([(center - MESH_W / 2) / width, 0, MESH_W / width, 1], projection="3d")
    for mask, color in group.meshes:
        triangles = surface(mask, scene.spacing, scene.origin)
        mesh_ax.add_collection3d(Poly3DCollection(triangles, facecolors=shaded(triangles, color), edgecolors="none"))
    mesh_ax.set_xlim(scene.origin[0], scene.end[0])
    mesh_ax.set_ylim(scene.origin[1], scene.end[1])
    mesh_ax.set_zlim(scene.origin[2], scene.end[2])
    mesh_ax.set_box_aspect(tuple(np.subtract(scene.end, scene.origin)))
    mesh_ax.view_init(**VIEW)
    mesh_ax.set_axis_off()


def arrow(fig, x: float, y: float) -> None:
    """Block arrow pointing right, centered on a point of the figure.

    Args:
        fig: Figure to draw on.
        x: Horizontal center in inches.
        y: Vertical center in inches.
    """
    width, height = fig.get_size_inches()
    half = 0.36
    fig.add_artist(
        FancyArrowPatch(
            ((x - half) / width, y / height), ((x + half) / width, y / height), transform=fig.transFigure,
            arrowstyle="simple,head_width=1.5,head_length=1.1,tail_width=0.6", mutation_scale=30, color=INK, linewidth=0,
        )
    )


def main():
    """Draw the slice and the surfaces of one patient under both label sets."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_part1_corrected/train"))
    ap.add_argument("--before-data-dir", type=Path, default=Path("data/segthor_part1/train"))
    ap.add_argument("--patient", default="Patient_02", help="A patient whose esophagus label has two pieces before the correction.")
    ap.add_argument("--out-dir", type=Path, default=Path("figures/comparison"))
    args = ap.parse_args()

    gt_before, ct, spacing = load_patient(args.before_data_dir, args.patient)
    gt_after, _, _ = load_patient(args.data_dir, args.patient)
    pieces, n_pieces = ranked_pieces(gt_before == ESOPHAGUS)
    z = example_slice(gt_after)
    rows, cols = np.nonzero(((gt_after == ESOPHAGUS) | (gt_after == AORTA))[:, :, z].T)
    cy, cx = int(rows.mean()) - 6, int(cols.mean()) + 4
    crop = (slice(cy - CROP_HALF[1], cy + CROP_HALF[1]), slice(cx - CROP_HALF[0], cx + CROP_HALF[0]))
    box = tuple(slice(max(i.min() - 4, 0), i.max() + 5) for i in np.nonzero(gt_before > 0))
    origin = tuple(b.start * s for b, s in zip(box, spacing))
    end = tuple(o + n * s for o, n, s in zip(origin, gt_before[box].shape, spacing))
    scene = Scene(ct[:, :, z].T[crop], spacing[0] / spacing[1], spacing, origin, end)

    piece_slice, labels_slice = pieces[:, :, z].T[crop], gt_after[:, :, z].T[crop]
    where = {r: np.nonzero(piece_slice == r) for r in range(1, n_pieces + 1)}
    info = f"{args.patient}\naxial slice {z}"
    before = Group(
        info,
        [(piece_slice == r, PIECE_COLORS[r - 1]) for r in range(n_pieces, 0, -1)],
        [
            (f"{CLASSES[ESOPHAGUS]} piece {r}", (cc.mean(), rr.mean()), (cc.mean() + (22 if r == 1 else -12), 100 if r == 1 else 12), PIECE_COLORS[r - 1])
            for r, (rr, cc) in where.items()
        ],
        [(pieces[box] == r, PIECE_COLORS[r - 1]) for r in range(1, n_pieces + 1)],
    )
    centers = {k: np.nonzero(labels_slice == k) for k in (ESOPHAGUS, AORTA)}
    after = Group(
        info,
        [(labels_slice == k, EARTH_LABEL_COLORS[k]) for k in (ESOPHAGUS, AORTA)],
        [
            (CLASSES[k], (cc.mean(), rr.mean()), (cc.mean() + dx, dy), EARTH_LABEL_COLORS[k])
            for (k, (rr, cc)), dx, dy in zip(centers.items(), (-12, 22), (12, 100))
        ],
        [(gt_after[box] == k, EARTH_LABEL_COLORS[k]) for k in (ESOPHAGUS, AORTA)],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(2 * GROUP_W + ARROW_GAP, FIG_H))
    draw_group(fig, 0, before, scene)
    draw_group(fig, GROUP_W + ARROW_GAP, after, scene)
    arrow(fig, GROUP_W + ARROW_GAP / 2, PANEL_BOTTOM + PANEL_H / 2)
    outputs = {"label_correction_example.png": fig}
    for suffix, group in (("before", before), ("after", after)):
        single = plt.figure(figsize=(GROUP_W, FIG_H))
        draw_group(single, 0, group, scene)
        outputs[f"label_correction_example_{suffix}.png"] = single
    for filename, figure in outputs.items():
        figure.savefig(args.out_dir / filename, dpi=150, transparent=True)
        print(f"wrote {args.out_dir / filename}")


if __name__ == "__main__":
    main()
