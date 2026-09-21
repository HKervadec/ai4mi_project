#!/usr/bin/env python3
"""
The -1000 HU spike and everything else, per patient.

Left panel: a lollipop per patient showing the percentage of that scan's
voxels that are exactly SCAN_MIN_HU (-1000), the padding value stored outside
the scanned field of view. Right panel: an overlapping ridge plot, one row
per patient, of the remaining voxels rebinned into 10 HU ranges from -999 to
800 HU (values above 800 fall into the last range) via
range_percent. Ridge heights are each patient's
percentage of its own non -1000 voxels in that range, on one shared scale
(RIDGE_HEIGHT_SCALE), and are capped at a fixed height (RIDGE_HEIGHT_CAP) so
a handful of sharply peaked patients don't dwarf the rest. Both panels read
the exact 1 HU "scan" histograms written by tools/dataset_profile.py; the
nnU-Net clip lines come from its "nnunet sample" histograms via
nnunet_ct_normalization().

Usage:
    python dataset_analysis/profile_figures/scan_intensity/spike_and_ridges.py --profile-dir figures/profile
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from dataset_profile import load_tables  # noqa: E402
from profile_figures.common import apply_ticks_style, title_block  # noqa: E402
from profile_figures.scan_intensity.common import (
    output_path,  # noqa: E402
    PROFILE_DIR,
    SCAN_MIN_HU,
    clip_lines,
    nnunet_ct_normalization,
    patient_name,
    patients_in,
    range_percent,
)

EDGES = np.append(np.arange(-999, 800, 10), 800)  # 10 HU ranges, -999 to 800; values above 800 fold into the last one
RIDGE_HEIGHT_SCALE = 1.0  # row units per percentage point of a patient's non -1000 voxels
RIDGE_HEIGHT_CAP = 1.9  # row units; taller peaks are cut off here (named in the subtitle)
LOLLIPOP_XLIM = 60  # % of voxels exactly -1000


def spike_and_rest(counts: np.ndarray, start: int) -> tuple[float, np.ndarray]:
    """Split one scan's histogram into its -1000 HU spike and the binned remainder.

    Args:
        counts: 1 HU histogram of the whole scan.
        start: HU value of counts[0].

    Returns:
        Percentage of voxels exactly at SCAN_MIN_HU, and the percentage of the
        remaining voxels in each of EDGES' 10 HU ranges (summing to 100).
    """
    spike_index = SCAN_MIN_HU - start
    spike_pct = 100 * counts[spike_index] / counts.sum()
    rest = counts.copy()
    rest[spike_index] = 0
    return spike_pct, range_percent(rest, start, EDGES)


def step_outline(edges: np.ndarray, heights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates tracing `heights` as a step histogram, for fill_between/plot.

    Args:
        edges: Bin edges, length len(heights) + 1.
        heights: One value per bin.

    Returns:
        x and y arrays of the step outline (length 2 * len(heights)).
    """
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(heights, 2)
    return x, y


def main():
    """Draw the spike-and-ridges figure from the scan intensity histograms."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-dir", type=Path, default=PROFILE_DIR)
    args = ap.parse_args()

    hists = load_tables(args.profile_dir)["histograms"]
    patients = patients_in(hists)
    norm = nnunet_ct_normalization(hists)

    apply_ticks_style()
    colors = sns.color_palette("crest", len(patients))

    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 7], wspace=0.04)
    axl = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharey=axl)

    for i, patient in enumerate(patients):
        y0 = len(patients) - 1 - i
        spike_pct, rest_pct = spike_and_rest(*hists[(patient, "scan")])
        heights = np.minimum(rest_pct * RIDGE_HEIGHT_SCALE, RIDGE_HEIGHT_CAP)
        x, y = step_outline(EDGES, heights)
        ax.fill_between(x, y0, y0 + y, color=colors[i], lw=0, zorder=i)
        ax.plot(x, y0 + y, color="white", lw=1.3, zorder=i + 0.5)
        axl.plot([0, spike_pct], [y0 + 0.3] * 2, color=colors[i], lw=1.2)
        axl.plot(spike_pct, y0 + 0.3, "o", color=colors[i], markersize=10, mec="white")

    clip_lines(ax, norm, label=True)
    ax.set_xlim(EDGES[0], EDGES[-1] + 1)
    ax.set_xticks([-999, -750, -500, -250, 0, 250, 500, 800])
    ax.set_xticklabels(["−999", "−750", "−500", "−250", "0", "250", "500", "800+"])
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlabel("intensity (HU), voxels not at −1000")
    sns.despine(ax=ax, left=True)

    axl.set_yticks(np.arange(len(patients)) + 0.3)
    axl.set_yticklabels([patient_name(p) for p in patients[::-1]])
    axl.tick_params(axis="y", left=False)
    axl.set_xlim(0, LOLLIPOP_XLIM)
    axl.set_xticks([0, 25, 50])
    axl.set_xlabel("% of voxels\nexactly −1000")
    axl.set_ylim(-0.2, len(patients) + 1.2)
    sns.despine(ax=axl, left=True)

    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.08)
    title_block(
        fig,
        "Scan Intensity: The −1000 Spike and Everything Else",
        "Left: share of each scan's voxels that are exactly −1000 HU. Right: the remaining voxels per patient, "
        f"in 10 HU ranges (values above 800 HU in the last range); peaks are cut at a height of {RIDGE_HEIGHT_CAP:g} "
        "rows.",
    )

    out_path = output_path(args.profile_dir, "spike_and_ridges.png")
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
