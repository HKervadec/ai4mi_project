#!/usr/bin/env python3
"""
3D view of eight axial slices through one patient's CT scan, colored by
intensity.

Eight evenly spaced axial slices (from the lowest slice of the scan to the
highest) are drawn as flat planes stacked at their true height in mm. Each
pixel is colored by its HU value on the "mako" colormap, normalized over
[-1000, 600] HU. Opacity comes from see_through_alpha(): pixels at -1000 HU
are nearly transparent and opacity rises to full at -900 HU, so the stack
behind each slice stays visible.

Usage:
    python dataset_analysis/profile_figures/scan_intensity/slices_3d.py --patient Patient_01
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
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from profile_figures.common import apply_ticks_style, title_block
from profile_figures.scan_intensity.common import (
    output_path,
    DATA_DIR,
    PROFILE_DIR,
    draw_slice_stack,
    load_ct,
    see_through_alpha,
    patient_name,
    stack_indices,
)

VMIN, VMAX = -1000, 600
N_SLICES = 8


def slice_colors(volume: np.ndarray, indices: np.ndarray, cmap, norm: Normalize) -> list[np.ndarray]:
    """Color the chosen axial slices by HU, with see-through low values.

    Args:
        volume: HU volume (x, y, z).
        indices: Slice indices to color.
        cmap: Colormap mapping [0, 1] to RGBA.
        norm: Normalize instance mapping HU to [0, 1].

    Returns:
        One (x, y, 4) RGBA array per index, at full resolution.
    """
    colors = []
    for idx in indices:
        hu = volume[:, :, idx]
        rgba = cmap(norm(hu))
        rgba[..., 3] = see_through_alpha(hu)
        colors.append(rgba)
    return colors


def main():
    """Draw the stacked-slices figure for one patient and save it."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient", default="Patient_01")
    ap.add_argument("--profile-dir", type=Path, default=PROFILE_DIR)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args()

    volume, spacing = load_ct(args.data_dir, args.patient)
    indices = stack_indices(volume.shape[2], N_SLICES)

    apply_ticks_style()
    cmap = sns.color_palette("mako", as_cmap=True)
    norm = Normalize(vmin=VMIN, vmax=VMAX, clip=True)
    colors = slice_colors(volume, indices, cmap, norm)

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_slice_stack(ax, colors, spacing, indices)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("intensity (HU)")

    title_block(
        fig,
        f"Scan Intensity: Axial Slices of {patient_name(args.patient).capitalize()}",
        "Eight evenly spaced slices from the lowest to the highest, at their height in mm; "
        "values near −1000 are see-through, fully shown from −900 HU up.",
    )
    fig.subplots_adjust(top=0.88)

    out_path = output_path(args.profile_dir, "slices_3d.png")
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
