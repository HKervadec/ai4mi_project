#!/usr/bin/env python3
"""
Label size and intensity: every patient's labels drawn in 3D at the same
physical scale and sorted by volume, above one panel overlapping the HU
histograms of those labels and the background. Which labels appear depends
on which have voxels in the profiled dataset. Each patient has the same
shade in both parts (light = smallest volume, dark = largest).

Reads labels.csv, patients.csv and intensity_histograms.npz written by
tools/dataset_profile.py, and the label masks from the data directory.

Usage:
    python dataset_analysis/profile_figures/label_size_and_intensity.py --profile-dir figures/profile \
        --data-dir data/segthor_part1/train
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import zoom
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from dataset_profile import load_tables  # noqa: E402
from profile_figures.common import apply_ticks_style, build_arg_parser, header, out_subdir  # noqa: E402
from plot_style import EARTH_LABEL_COLORS  # noqa: E402
from profile_figures.label_intensity import binned_counts  # noqa: E402

COLORS = {**EARTH_LABEL_COLORS, "background": "#8C8C8C"}
MESH_MM = 2.0  # isotropic grid the masks are resampled to before meshing
HALF_WIDTH_MM, HALF_HEIGHT_MM = 70, 175  # shared 3D box for every shape
HU_RANGE = (-1030, 270)
BIN_HU = 10
LIGHT = np.array([-0.4, -0.6, 0.7]) / np.linalg.norm([-0.4, -0.6, 0.7])


def shades(color: str, n: int, lightest: float = 0.78) -> list[np.ndarray]:
    """Blend a color towards white, from lightest (index 0) to the full color (index n-1).

    Args:
        color: Any matplotlib color.
        n: Number of shades.
        lightest: How far the first shade is blended towards white (0-1).

    Returns:
        n RGB arrays.
    """
    base = np.array(to_rgb(color))
    return [base + (1 - base) * t for t in np.linspace(lightest, 0.0, n)]


def resample_mask(mask: np.ndarray, zooms: np.ndarray, step_mm: float) -> np.ndarray:
    """Crop a boolean mask to its bounding box and resample it to an isotropic grid.

    Args:
        mask: Boolean mask with at least one True voxel.
        zooms: Voxel spacing (x, y, z) in mm.
        step_mm: Output voxel size in mm.

    Returns:
        Boolean mask on a step_mm grid covering the bounding box.
    """
    idx = np.argwhere(mask)
    lo, hi = idx.min(axis=0), idx.max(axis=0) + 1
    sub = mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].astype(np.float32)
    return zoom(sub, np.asarray(zooms) / step_mm, order=1) > 0.5


def draw_shape(ax, mask: np.ndarray, color: np.ndarray) -> None:
    """Draw a resampled mask as a shaded surface centered in a shared 3D box."""
    verts, faces, _, _ = marching_cubes(np.pad(mask, 1).astype(np.float32), 0.5, spacing=(MESH_MM,) * 3)
    tri = (verts - MESH_MM - np.array(mask.shape) * MESH_MM / 2)[faces]
    tri[..., 0] *= -1  # array x runs towards patient left; flip so patient left is on the viewer's right
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    ax.add_collection3d(
        Poly3DCollection(tri, facecolors=np.outer(0.5 + 0.5 * np.abs(normals @ LIGHT), color), linewidths=0)
    )
    ax.set_xlim(-HALF_WIDTH_MM, HALF_WIDTH_MM)
    ax.set_ylim(-HALF_WIDTH_MM, HALF_WIDTH_MM)
    ax.set_zlim(-HALF_HEIGHT_MM, HALF_HEIGHT_MM)
    ax.set_box_aspect((1, 1, HALF_HEIGHT_MM / HALF_WIDTH_MM), zoom=1.85)
    ax.set_axis_off()
    ax.view_init(elev=12, azim=-60)


def main():
    """Draw the label size and intensity figure."""
    ap = build_arg_parser(__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_part1/train"))
    args = ap.parse_args()

    tables = load_tables(args.profile_dir)
    # Labels with at least one voxel somewhere in this dataset -- not hardcoded,
    # so the same figure works whether the aorta annotation (label 4) is present.
    labels_table = tables["labels"]
    LABELS = sorted(int(v) for v in labels_table.loc[labels_table.voxels > 0, "label"].unique())
    hists = tables["histograms"]
    patients = sorted({p for p, _ in hists})
    volume = labels_table.set_index(["patient", "label"]).volume_ml
    order = {lab: volume.xs(lab, level="label").reindex(patients).sort_values().index for lab in LABELS}
    order["background"] = tables["patients"].set_index("patient").voxels.reindex(patients).sort_values().index

    apply_ticks_style()
    # Layout in inches so the shape rows keep their size and spacing when a label is added.
    row_h_in, pitch_in, band_in = 2.27, 1.55, 3.0
    height = band_in + pitch_in * (len(LABELS) - 1) + row_h_in
    fig = plt.figure(figsize=(17, height))
    col_w, row_h, left = 0.0465, row_h_in / height, 0.07  # axes overlap: 3D axes leave empty margin
    pitch = pitch_in / height
    masks = {}
    for patient in patients:
        img = nib.load(str(args.data_dir / patient / "GT.nii.gz"))
        seg, zooms = np.asarray(img.dataobj), np.array(img.header.get_zooms()[:3], dtype=float)
        for lab in LABELS:
            masks[(patient, lab)] = resample_mask(seg == lab, zooms, MESH_MM)
    for r, lab in enumerate(LABELS):
        top = 1.0 - r * pitch
        for c, (patient, shade) in enumerate(zip(order[lab], shades(COLORS[lab], len(patients)))):
            ax = fig.add_axes([left + c * col_w, top - row_h, col_w, row_h], projection="3d")
            draw_shape(ax, masks[(patient, lab)], shade)
            fig.text(
                left + (c + 0.5) * col_w,
                top - row_h / 2 - 0.63 / height,
                patient[-2:],
                ha="center",
                fontsize=10,
                color="#555555",
            )
        fig.text(0.01, top - row_h / 2, f"label {lab}", fontsize=17, fontweight="bold", va="center")

    keys = ["background", *LABELS]
    gap = 0.035
    panel_w = (0.9 - gap * (len(keys) - 1)) / len(keys)
    for i, key in enumerate(keys):
        name = key if key == "background" else f"label {key}"
        ax = fig.add_axes([left + i * (panel_w + gap), 0.63 / height, panel_w, 1.97 / height])
        per_patient = [binned_counts(*hists[(p, name)], *HU_RANGE, BIN_HU) for p in order[key]]
        x = per_patient[0][0]
        for (_, counts), shade in zip(per_patient, shades(COLORS[key], len(patients), lightest=0.6)):
            ax.step(x, 100 * counts / counts.sum(), where="mid", color=shade, lw=1.1)
        pooled = sum(c for _, c in per_patient)
        ax.step(x, 100 * pooled / pooled.sum(), where="mid", color=COLORS[key], lw=2.6, zorder=5)
        ax.set_xlim(*HU_RANGE)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("HU", fontsize=12)
        if i == 0:
            ax.set_ylabel("% of the group's voxels", fontsize=12)
        ax.set_title(name, fontsize=15, fontweight="bold", loc="left", color=COLORS[key])
        sns.despine(ax=ax)
    fig.text(
        left,
        3.15 / height,
        "HU inside each label and the background: one line per patient (same shading as the shapes above), thick line = all patients pooled",
        fontsize=15,
        fontweight="bold",
    )

    label_range = f"{LABELS[0]}–{LABELS[-1]}" if len(LABELS) > 1 else str(LABELS[0])
    header(
        fig,
        "Label Size and Intensity",
        f"Top: each patient's label {label_range} shape at the same physical scale, sorted smallest (light) to "
        f"largest (dark) by volume. Bottom: {BIN_HU} HU-wide bins as % of the group's voxels, one line per patient.",
        subtitle_y=1 - 0.55 / height,
    )
    out = out_subdir(args.profile_dir, "label_size_and_intensity") / "label_size_intensity.png"
    fig.savefig(out, dpi=115)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
