#!/usr/bin/env python3
"""Scan intensity before and after nnU-Net's CT normalization.

Top row: patient 01's CT as two 3D slice stacks, one colored by raw HU and one
by the value after normalize() -- clip to nnU-Net's foreground-sample
percentiles, then subtract the mean and divide by the standard deviation
(nnunet_ct_normalization()). Opacity comes from see_through_alpha() on the raw HU:
nearly transparent at -1000 HU, fully opaque from -900 HU up.

Bottom row: whole-scan intensity histograms for every patient, as overlapping
semi-transparent step fills (% of that scan's voxels, log scale). The left
panel bins raw HU in fixed-width ranges from -1000 to 800 HU; the right panel
bins the same voxels in fixed-width HU ranges between the clip values, mapped
through normalize(). range_percent() folds voxels outside each panel's range
into its first/last bin, so the right panel's end bins are exactly the mass
normalize() itself clips.

Usage:
    python dataset_analysis/profile_figures/scan_intensity/nnunet_normalization.py --profile-dir figures/profile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from dataset_profile import load_tables  # noqa: E402
from profile_figures.common import apply_ticks_style, title_block  # noqa: E402
from profile_figures.scan_intensity.common import (
    output_path,  # noqa: E402
    DATA_DIR,
    PROFILE_DIR,
    draw_slice_stack,
    load_ct,
    see_through_alpha,
    nnunet_ct_normalization,
    normalize,
    patient_name,
    patients_in,
    range_percent,
    stack_indices,
)

N_SLICES = 8
RAW_DISPLAY_RANGE = (-1000, 600)
RAW_HU_RANGE = (-1000, 800)
RAW_BIN_WIDTH = 20
NORM_BIN_WIDTH_HU = 20
Y_LIMITS = (1e-3, 300)
CMAP = "mako"
EXAMPLE_PATIENT = "Patient_01"


def stack_rgba(ct: np.ndarray, values: np.ndarray, indices: np.ndarray, cmap, norm: Normalize) -> list[np.ndarray]:
    """RGBA image per slice, with opacity from the raw HU (see_through_alpha).

    Args:
        ct: Raw HU volume (x, y, z), used for opacity.
        values: Quantity to color by, same shape as ct -- raw HU or normalized.
        indices: Slice indices to draw.
        cmap: A matplotlib colormap.
        norm: A matplotlib Normalize instance for values.

    Returns:
        List of (nx, ny, 4) RGBA arrays, one per slice in indices.
    """
    images = []
    for zi in indices:
        rgba = cmap(norm(values[:, :, zi]))
        rgba[..., 3] = see_through_alpha(ct[:, :, zi])
        images.append(rgba)
    return images


def add_stack_colorbar(fig, ax, cmap, norm: Normalize, label: str) -> None:
    """Attach a colorbar for a 3D slice-stack axis.

    Args:
        fig: Figure the axis belongs to.
        ax: The 3D axis to attach the colorbar next to.
        cmap: The colormap used to draw the stack.
        norm: The Normalize instance used to draw the stack.
        label: Colorbar label.
    """
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.12)
    cbar.set_label(label)


def top_row(fig, gs, data_dir: Path, norm: dict) -> None:
    """Draw the example patient's raw and normalized CT as two 3D slice stacks.

    Args:
        fig: Figure to draw into.
        gs: A 1x2 GridSpec slice to place the two 3D axes in.
        data_dir: Folder with one Patient_XX folder per patient.
        norm: nnU-Net normalization stats from nnunet_ct_normalization().
    """
    ct, spacing = load_ct(data_dir, EXAMPLE_PATIENT)
    indices = stack_indices(ct.shape[2], N_SLICES)
    cmap = colormaps[CMAP]

    ax_raw = fig.add_subplot(gs[0], projection="3d")
    norm_raw = Normalize(vmin=RAW_DISPLAY_RANGE[0], vmax=RAW_DISPLAY_RANGE[1], clip=True)
    draw_slice_stack(ax_raw, stack_rgba(ct, ct, indices, cmap, norm_raw), spacing, indices)
    ax_raw.set_title(f"{patient_name(EXAMPLE_PATIENT)}, raw HU", fontsize=13, loc="left")
    add_stack_colorbar(fig, ax_raw, cmap, norm_raw, "intensity (HU)")

    normalized = normalize(ct.astype(np.float32), norm)
    display_lo, display_hi = normalize(np.array([norm["clip_low"], norm["clip_high"]]), norm)
    ax_norm = fig.add_subplot(gs[1], projection="3d")
    norm_norm = Normalize(vmin=display_lo, vmax=display_hi, clip=True)
    draw_slice_stack(ax_norm, stack_rgba(ct, normalized, indices, cmap, norm_norm), spacing, indices)
    ax_norm.set_title(f"{patient_name(EXAMPLE_PATIENT)}, after nnU-Net normalization", fontsize=13, loc="left")
    add_stack_colorbar(fig, ax_norm, cmap, norm_norm, "value after nnU-Net normalization")


def raw_bin_percents(hists: dict, patients: list[str]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Percentage of each patient's scan voxels in fixed-width raw HU ranges.

    Voxels below/above RAW_HU_RANGE pile into the first/last range.

    Args:
        hists: {(patient, group): (counts, start)} from load_tables.
        patients: Patient ids to include.

    Returns:
        Tuple of (bin edges in HU, list of per-patient bin heights in percent).
    """
    edges = np.arange(RAW_HU_RANGE[0], RAW_HU_RANGE[1] + RAW_BIN_WIDTH, RAW_BIN_WIDTH, dtype=float)
    heights = [range_percent(*hists[(p, "scan")], edges) for p in patients]
    return edges, heights


def norm_bin_percents(hists: dict, patients: list[str], norm: dict) -> tuple[np.ndarray, list[np.ndarray]]:
    """Percentage of each patient's scan voxels in fixed-width HU ranges after normalization.

    Bin edges run from clip_low to clip_high in HU and are then mapped through
    normalize(). Voxels outside the clip range pile into the first/last range,
    exactly matching the clipping normalize() itself applies.

    Args:
        hists: {(patient, group): (counts, start)} from load_tables.
        patients: Patient ids to include.
        norm: nnU-Net normalization stats from nnunet_ct_normalization().

    Returns:
        Tuple of (bin edges in normalized units, list of per-patient bin heights in percent).
    """
    n_bins = round((norm["clip_high"] - norm["clip_low"]) / NORM_BIN_WIDTH_HU)
    edges_hu = np.linspace(norm["clip_low"], norm["clip_high"], n_bins + 1)
    heights = [range_percent(*hists[(p, "scan")], edges_hu) for p in patients]
    return normalize(edges_hu, norm), heights


def step_fill(ax, edges: np.ndarray, heights: list[np.ndarray], colors: list) -> None:
    """Draw one semi-transparent filled step histogram per patient, log y-scale.

    Args:
        ax: Axis to draw into.
        edges: Bin edges (length n_bins + 1).
        heights: Per-patient bin heights in percent (length n_bins each).
        colors: One color per patient, same order as heights.
    """
    for height, color in zip(heights, colors):
        step_y = np.append(height, height[-1])
        ax.step(edges, np.maximum(step_y, Y_LIMITS[0]), where="post", color=color, linewidth=1.1, alpha=0.85)
        ax.fill_between(edges, np.maximum(step_y, Y_LIMITS[0]), step="post", color=color, alpha=0.10)
    ax.set_yscale("log")
    ax.set_ylim(*Y_LIMITS)


def bottom_row(fig, gs, hists: dict, patients: list[str], norm: dict) -> None:
    """Draw the two whole-scan histogram panels (raw and normalized), all patients.

    Args:
        fig: Figure to draw into.
        gs: A 1x2 GridSpec slice to place the two axes in.
        hists: {(patient, group): (counts, start)} from load_tables.
        patients: Patient ids to include, in the color order used for the colorbar.
        norm: nnU-Net normalization stats from nnunet_ct_normalization().
    """
    colors = sns.color_palette("crest", len(patients))

    ax_raw = fig.add_subplot(gs[0])
    edges_raw, heights_raw = raw_bin_percents(hists, patients)
    step_fill(ax_raw, edges_raw, heights_raw, colors)
    ax_raw.set_xlabel("intensity (HU)")
    ax_raw.set_ylabel("% of the scan's voxels")
    ax_raw.set_xlim(*RAW_HU_RANGE)
    ax_raw.set_title("raw HU, whole scan", fontsize=13, loc="left")
    sns.despine(ax=ax_raw)

    ax_norm = fig.add_subplot(gs[1])
    edges_norm, heights_norm = norm_bin_percents(hists, patients, norm)
    step_fill(ax_norm, edges_norm, heights_norm, colors)
    ax_norm.set_xlabel("value after nnU-Net normalization")
    ax_norm.set_ylabel("% of the scan's voxels")
    ax_norm.set_xlim(edges_norm[0] - 0.15, edges_norm[-1] + 0.15)
    ax_norm.set_title("after nnU-Net normalization, whole scan", fontsize=13, loc="left")
    sns.despine(ax=ax_norm)

    below_text = f"← every voxel below {norm['clip_low']:.0f} HU".replace("-", "−")
    above_text = f"every voxel above {norm['clip_high']:.0f} HU →".replace("-", "−")
    ax_norm.text(edges_norm[0] + 0.2, 60, below_text, fontsize=11, color="#333333", ha="left", va="center")
    ax_norm.text(edges_norm[-1] - 0.2, 60, above_text, fontsize=11, color="#333333", ha="right", va="center")

    sm = ScalarMappable(cmap=sns.color_palette("crest", as_cmap=True), norm=Normalize(vmin=1, vmax=len(patients)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_raw, ax_norm], orientation="horizontal", fraction=0.035, pad=0.16, aspect=40)
    cbar.set_label("patient")
    cbar.set_ticks([1, len(patients)])
    cbar.set_ticklabels([f"patient {1:02d}", f"patient {len(patients):02d}"])


def main():
    """Generate the before/after nnU-Net normalization figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-dir", type=Path, default=PROFILE_DIR)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args()

    hists = load_tables(args.profile_dir)["histograms"]
    patients = patients_in(hists)
    norm = nnunet_ct_normalization(hists)

    apply_ticks_style()
    fig = plt.figure(figsize=(18, 11))
    outer = fig.add_gridspec(2, 1, height_ratios=(1.05, 1), hspace=0.30, top=0.86, bottom=0.14)
    top_gs = outer[0].subgridspec(1, 2, wspace=0.02)
    bottom_gs = outer[1].subgridspec(1, 2, wspace=0.28)

    top_row(fig, top_gs, args.data_dir, norm)
    bottom_row(fig, bottom_gs, hists, patients, norm)

    title_block(
        fig,
        "Scan Intensity: Before and After nnU-Net Normalization",
        f"Top: {patient_name(EXAMPLE_PATIENT)} as slices. Bottom: whole-scan histograms of all {len(patients)} "
        f"patients. nnU-Net clips to {norm['clip_low']:.0f}…{norm['clip_high']:.0f} HU, then rescales.".replace(
            " -", " −"
        ).replace("…-", "…−"),
    )

    out_path = output_path(args.profile_dir, "nnunet_normalization.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
