#!/usr/bin/env python3
"""Scan intensity by distance from the edge of the scanned area.

Every axial slice of every patient is split into four regions with
common.slice_regions(): outside the scanned area, and its outer, middle and
central thirds by distance from the area's edge. For each region, a 1 HU HU
histogram of the region's voxels is accumulated per patient (full
resolution, every slice), then range_percent() turns that into the
percentage of the region's voxels in each of a set of fixed-width HU ranges
from -1000 to 800 (values above 800 fold into the last range). Patients are
processed in parallel with multiprocessing.Pool.

Left: patient 01's scan as eight slices stacked in 3D, each pixel colored
by its region. Right: one panel per region, with every patient's histogram
as a semi-transparent step fill and the pooled 20-patient histogram (summed
counts, not averaged percentages) as a dark outline.

Usage:
    python dataset_analysis/profile_figures/scan_intensity/center_edge_outside.py --profile-dir figures/profile
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from dataset_profile import load_tables  # noqa: E402
from profile_figures.common import apply_ticks_style, title_block  # noqa: E402
from profile_figures.scan_intensity.common import (
    output_path,  # noqa: E402
    DATA_DIR,
    PROFILE_DIR,
    REGION_NAMES,
    clip_lines,
    draw_slice_stack,
    load_ct,
    nnunet_ct_normalization,
    patient_name,
    patients_in,
    pool_histograms,
    range_percent,
    slice_regions,
    stack_indices,
)

PLOT_PATIENT = "Patient_01"
N_PLOT_SLICES = 8
HU_RANGE = (-1000, 3071)  # fixed range every stored value falls in, used only to size the accumulator
BIN_EDGES = np.arange(-1000, 801, 20, dtype=float)
Y_FLOOR = 1e-3
REGION_COLORS = ["#CFCFCF", *sns.color_palette("crest", 3)]


def _region_histograms(patient: str, data_dir: Path) -> dict:
    """Per-region 1 HU histogram for one patient's whole scan, at full resolution.

    Args:
        patient: Patient id, e.g. "Patient_01".
        data_dir: Folder with one Patient_XX folder per patient.

    Returns:
        Dict with "patient", "hists" (one (counts, start) pair per region,
        region order matching REGION_NAMES), and, for PLOT_PATIENT only,
        "plot_regions" (int8 array of shape (N_PLOT_SLICES, nx, ny)),
        "plot_indices" and "spacing" -- otherwise those three are None.
    """
    ct, spacing = load_ct(data_dir, patient)
    lo, hi = HU_RANGE
    offset, length = -lo, hi - lo + 1
    hists = [np.zeros(length, dtype=np.int64) for _ in REGION_NAMES]

    plot_regions = plot_indices = None
    if patient == PLOT_PATIENT:
        plot_indices = stack_indices(ct.shape[2], N_PLOT_SLICES)
        plot_regions = np.zeros((N_PLOT_SLICES, ct.shape[0], ct.shape[1]), dtype=np.int8)

    for z in range(ct.shape[2]):
        sl = ct[:, :, z]
        regions = slice_regions(sl, tuple(spacing[:2]))
        for r in range(len(REGION_NAMES)):
            vals = sl[regions == r]
            if vals.size:
                clipped = np.clip(vals, lo, hi).astype(np.int64) + offset
                hists[r] += np.bincount(clipped, minlength=length)
        if plot_indices is not None:
            match = np.flatnonzero(plot_indices == z)
            if match.size:
                plot_regions[match[0]] = regions

    return {
        "patient": patient,
        "hists": [(h, lo) for h in hists],
        "plot_regions": plot_regions,
        "plot_indices": plot_indices,
        "spacing": spacing,
    }


def draw_patient_stack(fig, ax, plot_regions: np.ndarray, plot_indices: np.ndarray, spacing: np.ndarray) -> None:
    """Draw patient 01's eight slices in 3D, colored by region, with a region legend.

    Args:
        fig: Figure to attach the legend to.
        ax: 3D axis to draw into.
        plot_regions: int8 array (N_PLOT_SLICES, nx, ny) of region ids.
        plot_indices: Slice index of each entry in plot_regions.
        spacing: Voxel spacing (x, y, z) in mm.
    """
    cmap = ListedColormap(REGION_COLORS)
    colors = [cmap(regions) for regions in plot_regions]
    draw_slice_stack(ax, colors, spacing, plot_indices)
    ax.set_title(patient_name(PLOT_PATIENT), loc="left", fontsize=14, fontweight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in REGION_COLORS]
    fig.legend(handles, REGION_NAMES, loc="upper left", bbox_to_anchor=(0.03, 0.8), frameon=False, fontsize=12)


def draw_region_panels(fig, gs, per_patient_pct: dict, pooled_pct: dict, norm: dict) -> None:
    """Draw one histogram panel per region, all patients plus the pooled outline.

    Args:
        fig: Figure to draw into.
        gs: A 1x1 GridSpec slice to place the panel column in.
        per_patient_pct: {region index: list of per-patient percent-per-bin arrays}.
        pooled_pct: {region index: pooled percent-per-bin array}.
        norm: nnU-Net normalization stats from nnunet_ct_normalization(), for the clip lines.
    """
    big_ax = fig.add_subplot(gs)
    big_ax.set_ylabel("% of the region's voxels", labelpad=38, fontsize=13)
    big_ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    for spine in big_ax.spines.values():
        spine.set_visible(False)

    rows = gridspec.GridSpecFromSubplotSpec(len(REGION_NAMES), 1, subplot_spec=gs, hspace=0.18)
    axes = []
    for r, name in enumerate(REGION_NAMES):
        ax = fig.add_subplot(rows[r, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
        color = REGION_COLORS[r]

        for pct in per_patient_pct[r]:
            ax.stairs(np.clip(pct, Y_FLOOR, None), BIN_EDGES, fill=True, color=color, alpha=0.15, baseline=Y_FLOOR)
        ax.stairs(np.clip(pooled_pct[r], Y_FLOOR, None), BIN_EDGES, fill=False, color="#222222", linewidth=1.8)

        ax.set_yscale("log")
        ax.set_ylim(Y_FLOOR, 100)
        ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 100])
        clip_lines(ax, norm, label=(r == 0))
        ax.set_title(name, loc="right", fontsize=13, fontweight="bold", pad=2)
        sns.despine(ax=ax)
        if r < len(REGION_NAMES) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

    axes[-1].set_xlabel("intensity (HU)")
    axes[0].text(
        0.99,
        1.35,
        "dark line: pooled across all patients",
        transform=axes[0].transAxes,
        fontsize=10,
        color="#444444",
        ha="right",
    )


def main():
    """Generate the center/edge/outside scan intensity figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-dir", type=Path, default=PROFILE_DIR)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args()

    hists = load_tables(args.profile_dir)["histograms"]
    norm = nnunet_ct_normalization(hists)
    patients = patients_in(hists)

    with mp.Pool() as pool:
        results = pool.starmap(_region_histograms, [(p, args.data_dir) for p in patients])

    region_hists = {r: [] for r in range(len(REGION_NAMES))}
    plot_regions = plot_indices = plot_spacing = None
    for res in results:
        for r, h in enumerate(res["hists"]):
            region_hists[r].append(h)
        if res["patient"] == PLOT_PATIENT:
            plot_regions, plot_indices, plot_spacing = res["plot_regions"], res["plot_indices"], res["spacing"]

    per_patient_pct = {r: [range_percent(c, s, BIN_EDGES) for c, s in region_hists[r]] for r in region_hists}
    pooled_pct = {r: range_percent(*pool_histograms(region_hists[r]), BIN_EDGES) for r in region_hists}

    apply_ticks_style()
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.4, 0.6], wspace=0.28, figure=fig)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    draw_patient_stack(fig, ax3d, plot_regions, plot_indices, plot_spacing)
    draw_region_panels(fig, gs[0, 1], per_patient_pct, pooled_pct, norm)

    title_block(
        fig,
        "Scan Intensity: Center, Edge and Outside the Scanned Area",
        f"Per slice, split by distance from the scanned area's edge. Left: {patient_name(PLOT_PATIENT)}'s "
        "slices by region; right: one line per patient, black = pooled.",
    )
    fig.subplots_adjust(top=0.88)

    out_path = output_path(args.profile_dir, "center_edge_outside.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
