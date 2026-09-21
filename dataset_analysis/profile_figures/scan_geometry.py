#!/usr/bin/env python3
"""
Scan geometry: voxel spacing, slices per scan and physical size of every CT scan.

Writes three figures into figures/profile/scan_geometry/:
  scan_geometry.png     dot histograms, one dot per scan.
  grid_per_patient.png  one row per patient, one column per measurement, with
                        the median of all patients as the bottom row.
  field_of_view.png     every axial field-of-view size drawn to scale (bolder
                        outline = more scans), the smallest filled with one of
                        its own slices, next to histograms of slice count,
                        slice spacing and scan length.

Reads patients.csv written by tools/dataset_profile.py, and one CT volume from
--data-dir for the example slice.

Usage:
    python dataset_analysis/profile_figures/scan_geometry.py --profile-dir figures/profile \
        --data-dir data/segthor_part1/train
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profile_figures.common import (
    apply_ticks_style,
    build_arg_parser,
    header,
    out_subdir,
)  # noqa: E402
from plot_style import EARTH, apply_style, decorate, tint  # noqa: E402

MEDIAN_COLOR = "#D1495B"
# (column, panel title, x-axis label, bin width); bins are centered on multiples of the width
PANELS = [
    ("x_spacing_mm", "Patients by pixel size", "pixel size, x = y (mm)", 0.01),
    ("z_spacing_mm", "Patients by slice spacing", "slice spacing, z (mm)", 0.01),
    ("spacing_ratio", "Patients by spacing ratio", "slice spacing ÷ pixel size", 0.05),
    ("z_voxels", "Patients by number of slices", "number of slices", 15),
    ("x_extent_mm", "Patients by image width", "image width (mm)", 25),
    ("z_extent_mm", "Patients by scan length", "scan length (mm)", 40),
]

# (column, column title) for the per-patient grid
GRID_COLUMNS = [
    ("x_spacing_mm", "x spacing (mm)"),
    ("y_spacing_mm", "y spacing (mm)"),
    ("z_spacing_mm", "z spacing (mm)"),
    ("x_voxels", "x voxels"),
    ("y_voxels", "y voxels"),
    ("z_voxels", "z voxels"),
    ("x_extent_mm", "x size (mm)"),
    ("y_extent_mm", "y size (mm)"),
    ("z_extent_mm", "z size (mm)"),
    ("voxels_millions", "total voxels (millions)"),
]


def dot_stacks(values: pd.Series, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Bin values into stacks of dots, one dot per value.

    Args:
        values: One value per scan.
        width: Bin width; bins are centered on multiples of it.

    Returns:
        x (bin center) and y (stack position, 0.5, 1.5, ...) for every value.
    """
    centers = np.round(values.to_numpy(dtype=float) / width) * width
    x, y = [], []
    for center in np.unique(centers):
        n = int((centers == center).sum())
        x += [center] * n
        y += list(np.arange(n) + 0.5)
    return np.array(x), np.array(y)


def draw_overview(scans: pd.DataFrame, out_dir: Path) -> None:
    """Draw the dot-histogram overview of scan geometry.

    Args:
        scans: patients.csv with a spacing_ratio column added.
        out_dir: Folder to write scan_geometry.png into.
    """
    colors = sns.color_palette("crest", len(PANELS))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (col, title, xlabel, width), color in zip(axes.ravel(), PANELS, colors):
        x, y = dot_stacks(scans[col], width)
        ax.scatter(x, y, s=110, color=color, edgecolor="white", lw=1.2)
        ax.plot(
            scans[col].median(),
            0,
            marker="^",
            markersize=13,
            color=MEDIAN_COLOR,
            clip_on=False,
            zorder=6,
            transform=ax.get_xaxis_transform(),
        )
        ax.set_ylim(0, y.max() + 1.1)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(title, fontsize=15, fontweight="bold", loc="left")
        ax.set_xlabel(xlabel, fontsize=13)
        sns.despine(ax=ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("patients")
    fig.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="^",
                ls="",
                color=MEDIAN_COLOR,
                markersize=12,
                label="median",
            )
        ],
        loc="lower center",
        frameon=False,
    )
    header(
        fig,
        "Scan Geometry: Voxel Spacing, Slices and Size",
        f"Voxel spacing, number of slices and physical size of the CT scan of each of the {len(scans)} patients. "
        "Each dot is one patient.",
        subtitle_y=0.935,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    out = out_dir / "scan_geometry.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def draw_grid_per_patient(scans: pd.DataFrame, out_dir: Path) -> None:
    """Draw every patient's geometry as rows, one column per measurement, median row at the bottom.

    Args:
        scans: patients.csv.
        out_dir: Folder to write grid_per_patient.png into.
    """
    table = scans.assign(voxels_millions=scans.voxels / 1e6).sort_values(
        ["x_spacing_mm", "z_spacing_mm"]
    )
    table["patient"] = table.patient.str.replace("Patient_", "patient ")
    cols = [col for col, _ in GRID_COLUMNS]
    median = pd.DataFrame(
        [{"patient": "median of all patients", **table[cols].median().to_dict()}]
    )
    table = pd.concat([table[["patient", *cols]], median], ignore_index=True)
    n_rows = len(table)
    colors = sns.color_palette("crest", len(GRID_COLUMNS))

    fig, axes = plt.subplots(1, len(GRID_COLUMNS), figsize=(30, 10), sharey=True)
    for ax, (col, label), color in zip(axes, GRID_COLUMNS, colors):
        ax.scatter(
            table[col][:-1],
            range(n_rows - 1),
            s=90,
            color=color,
            edgecolor="white",
            lw=1,
            zorder=3,
        )
        ax.scatter(
            table[col].iloc[-1],
            n_rows - 1,
            marker="^",
            s=150,
            color=MEDIAN_COLOR,
            zorder=4,
        )
        ax.axhline(n_rows - 1.5, color="#999999", lw=0.8)
        ax.yaxis.grid(True, color="#E6E6E6")
        ax.set_title(label, fontsize=14, fontweight="bold", loc="left")
        ax.tick_params(axis="y", left=False)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        sns.despine(ax=ax, left=True)
    axes[0].set_yticks(range(n_rows))
    axes[0].set_yticklabels(table.patient)
    axes[0].set_ylim(n_rows - 0.5, -0.5)
    header(
        fig,
        "Scan Geometry per Patient",
        "One row per patient, sorted by pixel size then slice spacing; the bottom row is the median of all patients, "
        "which nnU-Net uses as its target spacing.",
        subtitle_y=0.915,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = out_dir / "grid_per_patient.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def fov_groups(scans: pd.DataFrame) -> pd.DataFrame:
    """Group scans by axial field of view (512 x pixel spacing), largest first.

    Args:
        scans: patients.csv.

    Returns:
        One row per field-of-view size, indexed by the size rounded to whole mm, with
        columns n (scans), width_mm (exact size) and pixel_mm (pixel spacing).
    """
    fov = scans.x_extent_mm.round().astype(int).rename("fov")
    return (
        scans.groupby(fov)
        .agg(
            n=("patient", "size"),
            width_mm=("x_extent_mm", "mean"),
            pixel_mm=("x_spacing_mm", "mean"),
        )
        .sort_index(ascending=False)
    )


def keyed_line(ax, y: float, pieces: list[tuple[str, dict]]) -> None:
    """Write one line of text above an axes, each piece with its own font style.

    Args:
        ax: Axes to write above; y is in axes coordinates.
        y: Baseline height in axes coordinates.
        pieces: (text, extra text kwargs) pairs, drawn left to right.
    """
    prev = None
    for text, style in pieces:
        kw = dict(fontsize=10.5, color="#555555", va="bottom", ha="left") | style
        if prev is None:
            prev = ax.text(0, y, text, transform=ax.transAxes, **kw)
        else:
            prev = ax.annotate(text, xy=(1, 0), xycoords=prev, **kw)


def draw_field_of_view(scans: pd.DataFrame, data_dir: Path, out_dir: Path) -> None:
    """Draw every field-of-view size to scale with slice-count, slice-spacing and scan-length histograms.

    Args:
        scans: patients.csv.
        data_dir: Folder with one Patient_XX folder per patient, for the example slice.
        out_dir: Folder to write field_of_view.png into.
    """
    groups = fov_groups(scans)
    colors = dict(zip(sorted(groups.index), EARTH))
    fov = scans.x_extent_mm.round().astype(int)
    small = scans.loc[fov.idxmin()]
    vol = nib.load(data_dir / small.patient / f"{small.patient}.nii.gz")
    k = vol.shape[2] // 2
    axial = np.asarray(vol.dataobj[:, :, k], dtype=np.float32).T

    with plt.style.context("default"):
        apply_style()
        fig = plt.figure(figsize=(10.6, 7.2))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 0.8], hspace=0.6, wspace=0.16)

        ax = fig.add_subplot(gs[:, 0])
        for i, (size, row) in enumerate(groups.iterrows()):
            c = colors[size]
            ax.add_patch(
                Rectangle(
                    (0, 0),
                    row.width_mm,
                    row.width_mm,
                    facecolor=tint(c, 0.3),
                    edgecolor="none",
                    zorder=2 + i,
                )
            )
            if size == groups.index.min():
                shadow = np.zeros((*axial.shape, 4))
                shadow[..., :3] = mcolors.to_rgb(c)
                shadow[..., 3] = 0.75 * np.clip((axial + 1000) / 1400, 0, 1)
                ax.imshow(
                    shadow,
                    origin="upper",
                    extent=(0, row.width_mm, 0, row.width_mm),
                    zorder=6,
                )
            ax.plot(
                [0, row.width_mm, row.width_mm],
                [row.width_mm, row.width_mm, 0],
                color=c,
                linewidth=1.6 + 7.0 * row.n / groups.n.max(),
                solid_joinstyle="miter",
                solid_capstyle="butt",
                zorder=10 + i,
            )
            ax.text(
                12,
                row.width_mm - 12,
                f"{size} mm  ·  {row.pixel_mm:.3f} mm pixels",
                ha="left",
                va="top",
                fontsize=10.5,
                fontweight="bold",
                color=c,
                zorder=30,
            )
        lim = groups.width_mm.max() + 30
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(0, lim, 100))
        ax.set_yticks(np.arange(0, lim, 100))
        ax.set_xlabel("x: left–right field of view (mm)")
        ax.set_ylabel("y: anterior–posterior field of view (mm)")
        ax.grid(True, color="#E6E6E6", linewidth=0.7)

        hists = [
            (
                "z_voxels",
                np.arange(140, 300, 10),
                "Number of slices per scan (z)",
                "slices",
                None,
            ),
            (
                "z_spacing_mm",
                [1.875, 2.125, 2.375, 2.625],
                "Distance between slices (z)",
                "slice spacing (mm)",
                [2.0, 2.5],
            ),
            (
                "z_extent_mm",
                np.arange(300, 651, 25),
                "Scan length, superior–inferior (z)",
                "mm = slices × slice spacing",
                None,
            ),
        ]
        order = sorted(groups.index)
        for r, (col, bins, title, xlabel, ticks) in enumerate(hists):
            hx = fig.add_subplot(gs[r, 1])
            _, _, bars = hx.hist(
                [scans.loc[fov == f, col] for f in order],
                bins=bins,
                stacked=True,
                color=[tint(colors[f], 0.35) for f in order],
            )
            # hist() only wraps `bars` in a list when there's more than one
            # dataset -- with a single field-of-view group it returns the one
            # BarContainer directly.
            containers = bars if len(order) > 1 else [bars]
            for f, container in zip(order, containers):
                for bar in container:
                    bar.set_edgecolor(colors[f])
                    bar.set_linewidth(1.2)
                    bar.set_visible(bar.get_height() > 0)
            hx.set_title(title, loc="left", fontsize=12, fontweight="bold")
            hx.set_xlabel(xlabel)
            hx.set_ylabel("number of scans")
            hx.yaxis.get_major_locator().set_params(integer=True)
            edges = np.asarray(bins, dtype=float)
            occupied = np.flatnonzero(np.histogram(scans[col], bins=edges)[0])
            pad = 0.3 * np.diff(edges).mean()
            hx.set_xlim(edges[occupied[0]] - pad, edges[occupied[-1] + 1] + pad)
            if ticks is not None:
                hx.set_xticks(ticks)

        decorate(fig, None)
        fig.subplots_adjust(top=0.9, bottom=0.08, left=0.07, right=0.99)
        ax.text(
            0,
            1.068,
            r"Axial field of view (x $\times$ y): the area one slice covers",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        keyed_line(
            ax,
            1.038,
            [
                ("a ", {}),
                (
                    "bolder outline",
                    {"fontweight": "bold", "color": colors[groups.n.idxmax()]},
                ),
                (" = more scans share that size", {}),
            ],
        )
        patient = small.patient.replace("_", " ").lower()
        keyed_line(
            ax,
            1.008,
            [
                (
                    "red shadow",
                    {"fontweight": "bold", "color": colors[groups.index.min()]},
                ),
                (f" = {patient}, middle axial slice ({k + 1} of {vol.shape[2]})", {}),
            ],
        )
        out = out_dir / "field_of_view.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
    print(f"wrote {out}")


def main():
    """Draw the three scan geometry figures from patients.csv."""
    ap = build_arg_parser(__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_part1/train"))
    args = ap.parse_args()

    scans = pd.read_csv(args.profile_dir / "patients.csv")
    scans["spacing_ratio"] = scans.z_spacing_mm / scans.x_spacing_mm
    out_dir = out_subdir(args.profile_dir, "scan_geometry")
    apply_ticks_style()
    draw_overview(scans, out_dir)
    draw_grid_per_patient(scans, out_dir)
    draw_field_of_view(scans, args.data_dir, out_dir)


if __name__ == "__main__":
    main()
