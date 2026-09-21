#!/usr/bin/env python3
"""
Label pairs: where the labeled organs meet, drawn in 3D for every patient, with
a strip under each row of shapes giving each pair's shared border area.
Which pairs appear depends on which labels have voxels in the profiled
dataset (1-3 for the original course release, plus any pair with the aorta,
label 4, once that annotation is present).

Each patient's labels are drawn as faint gray outlines at the same physical
scale; the part of a label's surface that lies within one voxel of another
label is painted in a color for that pair and direction: within a slice where the
surface faces sideways, between slices where it faces up or down. The strips split the exact shared face area the same way:
faces between left-right or anterior–posterior neighbors are within a slice, faces
between voxels on neighboring slices are between slices. Patients are sorted
by total shared border area.

Reads label_pairs.csv written by tools/dataset_profile.py and the label masks
from the data directory.

Usage:
    python dataset_analysis/profile_figures/label_pairs.py --profile-dir figures/profile \
        --data-dir data/segthor_part1/train
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import distance_transform_edt, zoom
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profile_figures.common import apply_ticks_style, build_arg_parser, header, out_subdir  # noqa: E402
from plot_style import EARTH, tint  # noqa: E402

MESH_MM = 2.0
HALF_WIDTH_MM, HALF_HEIGHT_MM = 90, 170  # shared 3D box for every patient
DIRECTIONS = ("within a slice", "between slices")
# One earth-palette color per label pair (same family as the scan field-of-view figure); the
# direction is the shade: full color between slices, a light tint of it within a slice.
PAIR_COLORS = {(1, 2): EARTH[1], (1, 3): EARTH[0], (2, 3): EARTH[2]}
# Used only for pairs PAIR_COLORS doesn't name (e.g. involving the aorta, label 4).
_FALLBACK_PAIR_COLORS = [EARTH[3], "#7C5E86", "#5E6B7A"]
WITHIN_TINT = 0.6
FACING_UP = 0.7  # |normal z| above this counts a surface patch as facing the neighboring slice
SURFACE_RGBA = (0.6, 0.6, 0.6, 0.06)  # faint gray so only the contact patches stand out
MIN_BAR = 0.02  # any non-zero contact gets at least this bar length (fraction of a patient cell)
LIGHT = np.array([-0.4, -0.6, 0.7]) / np.linalg.norm([-0.4, -0.6, 0.7])


def contact_colors(pairs: list[tuple[int, int]]) -> dict[tuple[tuple[int, int], str], str]:
    """Color for every (pair, direction).

    Args:
        pairs: Label pairs that actually occur in this dataset.

    Returns:
        {(pair, direction): hex color}: the pair's color between slices, a tint of it within a slice.
    """
    fallback = iter(_FALLBACK_PAIR_COLORS)
    colors = {}
    for pair in pairs:
        base = PAIR_COLORS.get(pair) or next(fallback)
        colors[(pair, DIRECTIONS[0])] = to_hex(tint(base, WITHIN_TINT))
        colors[(pair, DIRECTIONS[1])] = to_hex(base)
    return colors


def label_grids(seg: np.ndarray, zooms: np.ndarray, labels: list[int], step_mm: float) -> dict[int, np.ndarray]:
    """Resample several labels onto one shared isotropic grid covering all of them.

    Args:
        seg: Integer label volume.
        zooms: Voxel spacing (x, y, z) in mm.
        labels: Label values to resample; at least one must be present.
        step_mm: Output voxel size in mm.

    Returns:
        {label: boolean mask}, all with the same shape and a one-voxel empty border.
    """
    idx = np.argwhere(np.isin(seg, labels))
    lo, hi = idx.min(axis=0), idx.max(axis=0) + 1
    crop = seg[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    factors = np.asarray(zooms) / step_mm
    return {lab: np.pad(zoom((crop == lab).astype(np.float32), factors, order=1) > 0.5, 1) for lab in labels}


def shared_area_by_direction(a: np.ndarray, b: np.ndarray, zooms: np.ndarray) -> dict[str, float]:
    """Shared face area between two label masks, split into within-slice and between-slice faces.

    Args:
        a: Boolean mask of the first label, shape (x, y, z).
        b: Boolean mask of the second label, same shape.
        zooms: Voxel spacing (x, y, z) in mm.

    Returns:
        {"within a slice": area of faces between x or y neighbors (mm²),
         "between slices": area of faces between z neighbors (mm²)}.
    """
    face_area = [zooms[1] * zooms[2], zooms[0] * zooms[2], zooms[0] * zooms[1]]
    per_axis = []
    for axis in range(3):
        head, tail = [slice(None)] * 3, [slice(None)] * 3
        head[axis], tail[axis] = slice(None, -1), slice(1, None)
        head, tail = tuple(head), tuple(tail)
        per_axis.append(face_area[axis] * int((a[head] & b[tail]).sum() + (b[head] & a[tail]).sum()))
    return {DIRECTIONS[0]: per_axis[0] + per_axis[1], DIRECTIONS[1]: per_axis[2]}


def _shade(pair: tuple[int, int], direction: str, colors: dict) -> np.ndarray:
    return np.array(to_rgb(colors[(pair, direction)]))


def _draw_patient(ax, grids: dict[int, np.ndarray], pairs: list[tuple[int, int]], colors: dict) -> None:
    near_other = {lab: distance_transform_edt(~g, sampling=MESH_MM) for lab, g in grids.items()}
    center = np.array(next(iter(grids.values())).shape) * MESH_MM / 2
    for lab, grid in grids.items():
        if not grid.any():
            continue
        verts, faces, _, _ = marching_cubes(grid.astype(np.float32), 0.5, spacing=(MESH_MM,) * 3)
        tri = verts[faces]
        rgba = np.tile(SURFACE_RGBA, (len(faces), 1))
        cell = np.clip((tri.mean(axis=1) / MESH_MM).astype(int), 0, np.array(grid.shape) - 1)
        tri = tri - center
        tri[..., 0] *= -1  # array x runs towards patient left; flip so patient left is on the viewer's right
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
        facing_up = np.abs(normals[:, 2]) > FACING_UP
        for pair in pairs:
            if lab in pair:
                other = pair[1] if lab == pair[0] else pair[0]
                touching = near_other[other][cell[:, 0], cell[:, 1], cell[:, 2]] <= 1.5 * MESH_MM
                for direction, mask in zip(DIRECTIONS, (~facing_up, facing_up)):
                    rgba[touching & mask] = np.r_[_shade(pair, direction, colors), 1.0]
        rgba[:, :3] *= (0.5 + 0.5 * np.abs(normals @ LIGHT))[:, None]
        ax.add_collection3d(Poly3DCollection(tri, facecolors=rgba, linewidths=0))
    ax.set_xlim(-HALF_WIDTH_MM, HALF_WIDTH_MM)
    ax.set_ylim(-HALF_WIDTH_MM, HALF_WIDTH_MM)
    ax.set_zlim(-HALF_HEIGHT_MM, HALF_HEIGHT_MM)
    ax.set_box_aspect((1, 1, HALF_HEIGHT_MM / HALF_WIDTH_MM), zoom=1.5)
    ax.set_axis_off()
    ax.view_init(elev=8, azim=-70)


def main():
    """Draw the label pairs contact figure."""
    ap = build_arg_parser(__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_part1/train"))
    args = ap.parse_args()

    labels_table = pd.read_csv(args.profile_dir / "labels.csv")
    # Labels with at least one voxel somewhere in this dataset -- not hardcoded,
    # so the same figure works whether the aorta annotation (label 4) is present.
    present_labels = sorted(int(v) for v in labels_table.loc[labels_table.voxels > 0, "label"].unique())
    pair_list = list(itertools.combinations(present_labels, 2))
    colors = contact_colors(pair_list)

    pairs = pd.read_csv(args.profile_dir / "label_pairs.csv")
    pairs = pairs[pairs["label_a"].isin(present_labels) & pairs["label_b"].isin(present_labels)]
    order = pairs.groupby("patient")["shared_face_area_mm2"].sum().sort_values().index

    rows = [(pair, direction) for pair in pair_list for direction in DIRECTIONS]
    area = {}

    apply_ticks_style()
    # Layout in inches so it grows with the number of pair rows (6 for 3 labels, 12 for 4).
    row3d_in, overlap_in, strip_in, label_in, band_in, top_in = 2.5, 0.5, 0.2 * len(rows), 0.35, 1.4, 0.95
    block_in = row3d_in - overlap_in + strip_in + label_in
    height = top_in + 2 * block_in + band_in
    fig = plt.figure(figsize=(20, height))
    left, row_h, strip_h = 0.09, row3d_in / height, strip_in / height
    col_w = (0.99 - left) / 10
    row_colors = [_shade(pair, direction, colors) for pair, direction in rows]
    blocks = []
    for r in range(2):
        top = 1 - (top_in + r * block_in) / height
        row_patients = order[r * 10 : (r + 1) * 10]
        for c, patient in enumerate(row_patients):
            img = nib.load(str(args.data_dir / patient / "GT.nii.gz"))
            seg, zooms = np.asarray(img.dataobj), np.array(img.header.get_zooms()[:3], dtype=float)
            ax = fig.add_axes([left + c * col_w, top - row_h, col_w, row_h], projection="3d")
            _draw_patient(ax, label_grids(seg, zooms, present_labels, MESH_MM), pair_list, colors)
            for a, b in pair_list:
                split = shared_area_by_direction(seg == a, seg == b, zooms)
                for direction in DIRECTIONS:
                    area[(patient, (a, b), direction)] = split[direction]
        blocks.append((top, row_patients))
    area_max = max(area.values())
    for top, row_patients in blocks:
        for c, patient in enumerate(row_patients):
            cell = fig.add_axes([left + c * col_w + 0.004, top - row_h - strip_h + overlap_in / height, col_w - 0.008, strip_h])
            vals = np.array([area[(patient, pair, direction)] for pair, direction in rows])
            width = np.where(vals > 0, np.maximum(0.92 * vals / area_max, MIN_BAR), 0)
            cell.barh(np.arange(len(rows)) + 0.5, width, left=0.04, height=0.75, color=row_colors)
            cell.set_xlim(0, 1)
            cell.set_ylim(len(rows), 0)
            cell.set_yticks(np.arange(len(rows)) + 0.5)
            cell.set_yticklabels([f"{a} & {b}, {d}" for (a, b), d in rows] if c == 0 else [], fontsize=10)
            cell.tick_params(axis="y", length=0, pad=9)
            cell.hlines(np.arange(len(rows)) + 0.5, -0.09, 0, colors=row_colors, lw=2.6, clip_on=False)
            cell.set_xticks([0.5], [f"patient {patient[-2:]}"])
            cell.tick_params(axis="x", length=0, labelsize=11)
            for side in ("top", "right", "bottom"):
                cell.spines[side].set_visible(False)
            cell.spines["left"].set_color("#bbbbbb")

    key = fig.add_axes([0.66, 0.6 / height, 0.3, 0.1 / height])
    key.set_xlim(0, area_max)
    key.set_yticks([])
    key.set_xlabel("bar length = shared border area (mm²); any contact is drawn at least a short tick", fontsize=11)
    for sp in ("left", "right", "top"):
        key.spines[sp].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_shade(p, d, colors), label=f"labels {p[0]} & {p[1]}, {d}")
        for p in pair_list
        for d in DIRECTIONS
    ]
    fig.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.55 / height), ncol=len(pair_list), frameon=False,
        fontsize=12,
    )
    header(
        fig,
        "Contact Between Label Pairs",
        f"One 3D render per patient (n={pairs.patient.nunique()}), sorted by total shared border area; "
        "color = surface within 1 voxel of the other label (color per pair and direction); "
        "bars = shared border area per pair and direction (mm²).",
        1 - 0.6 / height,
    )
    out = out_subdir(args.profile_dir, "label_pairs") / "label_pairs_contact_3d.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
