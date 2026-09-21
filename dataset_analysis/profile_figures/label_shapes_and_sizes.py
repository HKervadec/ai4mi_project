#!/usr/bin/env python3
"""
Label shapes and sizes per patient, for whichever labels have voxels in the
profiled dataset (1-3 for the original course release, 1-4 once the aorta
annotation is present).

Writes two figures into figures/profile/label_shapes_and_sizes/:
  label_shapes_3d.png        one 3D panel per patient with its label surfaces, at the
                             same physical scale and camera angle in every panel.
  label_size_per_patient.png one row per patient: voxels, volume (mL) and % of the
                             scan's voxels per label, with the median of all
                             patients as the bottom row.

Reads labels.csv written by tools/dataset_profile.py and the label masks from the
data directory.

Usage:
    python dataset_analysis/profile_figures/label_shapes_and_sizes.py --profile-dir figures/profile \
        --data-dir data/segthor_part1/train
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profile_figures.common import LABEL_COLORS, apply_ticks_style, build_arg_parser, header, out_subdir  # noqa: E402

OUT_NAME = "label_shapes_and_sizes"
MESH_STEP = 2  # marching-cubes step in voxels
PAD_MM = 5  # margin around the largest label box in the shared 3D box
MEDIAN_ROW = "median of all patients"
SIZE_COLUMNS = [
    ("voxels_thousands", "voxels (thousands)"),
    ("volume_ml", "volume (mL)"),
    ("pct_of_scan", "% of the scan's voxels"),
]


def label_meshes(task: tuple[Path, str, list[int]]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Surface mesh of every present label of one patient, in mm in the scan's frame.

    Args:
        task: (data directory, patient id, labels to mesh). Labels travel with
            the task rather than as a module global so this stays correct
            under multiprocessing's "spawn" start method, which re-imports
            the module in each worker instead of inheriting parent globals.

    Returns:
        {label: (vertices, faces)} for labels with at least one voxel.
    """
    data_dir, patient, labels = task
    img = nib.load(str(data_dir / patient / "GT.nii.gz"))
    seg = np.asanyarray(img.dataobj)
    spacing = np.array(img.header.get_zooms()[:3], dtype=float)
    meshes = {}
    for value in labels:
        idx = np.argwhere(seg == value)
        if len(idx) == 0:
            continue
        lo = np.maximum(idx.min(axis=0) - 2, 0)
        hi = np.minimum(idx.max(axis=0) + 3, seg.shape)
        sub = (seg[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] == value).astype(np.float32)
        verts, faces, _, _ = marching_cubes(sub, level=0.5, spacing=tuple(spacing), step_size=MESH_STEP)
        meshes[value] = (verts + lo * spacing, faces)
    return meshes


def label_boxes(labels: pd.DataFrame, present_labels: list[int]) -> pd.DataFrame:
    """Box around the present labels together, per patient, in mm.

    Args:
        labels: labels.csv.
        present_labels: Labels with at least one voxel in this dataset.

    Returns:
        One row per patient with <axis>_min_mm and <axis>_max_mm for x, y and z.
    """
    present = labels[labels.label.isin(present_labels) & (labels.voxels > 0)]
    agg = {f"{a}_min_mm": "min" for a in "xyz"} | {f"{a}_max_mm": "max" for a in "xyz"}
    return present.groupby("patient").agg(agg)


def draw_shapes(labels: pd.DataFrame, data_dir: Path, out_dir: Path, present_labels: list[int]) -> None:
    """Draw every patient's label surfaces in 3D at one shared scale.

    Args:
        labels: labels.csv.
        data_dir: Folder with one Patient_XX folder per patient.
        out_dir: Folder to write label_shapes_3d.png into.
        present_labels: Labels with at least one voxel in this dataset.
    """
    boxes = label_boxes(labels, present_labels)
    patients = list(boxes.index)
    half = {a: (boxes[f"{a}_max_mm"] - boxes[f"{a}_min_mm"]).max() / 2 + PAD_MM for a in "xyz"}
    with Pool() as pool:
        meshes = pool.map(label_meshes, [(data_dir, p, present_labels) for p in patients])

    fig = plt.figure(figsize=(20, 22))
    for i, (patient, patient_meshes) in enumerate(zip(patients, meshes)):
        ax = fig.add_subplot(4, 5, i + 1, projection="3d")
        for value, (verts, faces) in patient_meshes.items():
            ax.plot_trisurf(
                verts[:, 0], verts[:, 1], faces, verts[:, 2], color=LABEL_COLORS[value], lw=0, antialiased=False
            )
        for a, setter in zip("xyz", (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            center = (boxes.at[patient, f"{a}_min_mm"] + boxes.at[patient, f"{a}_max_mm"]) / 2
            setter(center - half[a], center + half[a])
        ax.set_box_aspect((half["x"], half["y"], half["z"]), zoom=1.2)
        ax.view_init(elev=10, azim=-60)
        ax.set_axis_off()
        ax.set_title(patient.replace("Patient_", "patient "), fontsize=14, fontweight="bold", y=1.07)

    label_range = f"{present_labels[0]}–{present_labels[-1]}" if len(present_labels) > 1 else str(present_labels[0])
    handles = [plt.Rectangle((0, 0), 1, 1, color=LABEL_COLORS[v], label=f"label {v}") for v in present_labels]
    fig.legend(handles=handles, loc="lower center", ncol=len(present_labels), frameon=False, fontsize=14)
    fig.subplots_adjust(left=0, right=1, bottom=0.03, top=0.92, wspace=-0.2, hspace=0.06)
    header(
        fig,
        "Label Shapes in 3D, Patient by Patient",
        f"Labels {label_range} of every patient at the same physical scale and camera angle.",
        subtitle_y=0.955,
    )
    out = out_dir / "label_shapes_3d.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def draw_sizes(labels: pd.DataFrame, out_dir: Path, present_labels: list[int]) -> None:
    """Draw voxels, volume and % of scan per patient and label, median of all patients at the bottom.

    Args:
        labels: labels.csv.
        out_dir: Folder to write label_size_per_patient.png into.
        present_labels: Labels with at least one voxel in this dataset.
    """
    table = labels[labels.label.isin(present_labels)].assign(voxels_thousands=lambda d: d.voxels / 1000)
    table["row"] = table.patient.str.replace("Patient_", "patient ")
    cols = [col for col, _ in SIZE_COLUMNS]
    medians = table.groupby("label")[cols].median().reset_index().assign(row=MEDIAN_ROW)
    table = pd.concat([table, medians], ignore_index=True)
    rows = sorted(table.row.unique().tolist(), key=lambda r: (r == MEDIAN_ROW, r))
    palette = {v: LABEL_COLORS[v] for v in present_labels}

    fig, axes = plt.subplots(1, len(SIZE_COLUMNS), figsize=(16, 11), sharey=True)
    for ax, (col, xlabel) in zip(axes, SIZE_COLUMNS):
        sns.stripplot(
            data=table,
            x=col,
            y="row",
            hue="label",
            order=rows,
            hue_order=present_labels,
            palette=palette,
            dodge=True,
            jitter=False,
            size=8,
            linewidth=0.6,
            edgecolor="white",
            legend=False,
            ax=ax,
        )
        ax.axhline(len(rows) - 1.5, color="#999999", lw=0.8, ls="--")
        ax.set(xlabel=xlabel, ylabel="")
        ax.set_xlim(left=0)
        ax.xaxis.grid(True, color="#E6E6E6")
        ax.tick_params(axis="y", left=False)
        sns.despine(ax=ax, left=True)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=LABEL_COLORS[v], markersize=9, label=f"label {v}")
        for v in present_labels
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(present_labels), frameon=False)
    label_range = f"{present_labels[0]}–{present_labels[-1]}" if len(present_labels) > 1 else str(present_labels[0])
    header(
        fig,
        "Label Size per Patient",
        f"One dot per patient for each label ({label_range}), across voxels, volume and % of the scan; row below "
        "the dashed line is the median across patients.",
        subtitle_y=0.935,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    out = out_dir / "label_size_per_patient.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    """Draw the label shape and label size figures."""
    ap = build_arg_parser(__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_part1/train"))
    args = ap.parse_args()

    labels = pd.read_csv(args.profile_dir / "labels.csv")
    # Labels with at least one voxel somewhere in this dataset -- not hardcoded,
    # so the same figure works whether the aorta annotation (label 4) is present.
    present_labels = sorted(int(v) for v in labels.loc[labels.voxels > 0, "label"].unique())
    out_dir = out_subdir(args.profile_dir, OUT_NAME)
    apply_ticks_style()
    draw_sizes(labels, out_dir, present_labels)
    draw_shapes(labels, args.data_dir, out_dir, present_labels)


if __name__ == "__main__":
    main()
