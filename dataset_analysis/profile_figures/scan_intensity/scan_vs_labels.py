#!/usr/bin/env python3
"""
Whole-scan intensity distribution against the distribution inside labels 1-3,
one small panel per patient.

For each patient, the "scan" histogram (every voxel) is rebinned into 20 HU
ranges from -1000 to 1600 HU (values above 1600 fall into the last range) and
drawn as bars above zero; the "all labels" histogram (voxels inside labels
1-3; label 4 has no voxels in this release) is rebinned the same
way and drawn as bars below zero, mirrored. Both use a log scale on percent
of voxels so rare but non-zero ranges stay visible. A gray band repeated
behind every panel's top half is the "scan" percentages averaged across all
20 patients, for comparison. Percentages come from dataset_profile.py's exact
1 HU histograms via range_percent. Dashed lines mark
nnU-Net's CT-normalization clip range, recomputed from its own foreground
sample.

Usage:
    python dataset_analysis/profile_figures/scan_intensity/scan_vs_labels.py --profile-dir figures/profile
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
from plot_style import PALETTE  # noqa: E402
from profile_figures.common import apply_ticks_style, title_block  # noqa: E402
from profile_figures.scan_intensity.common import (
    output_path,  # noqa: E402
    PROFILE_DIR,
    clip_lines,
    nnunet_ct_normalization,
    patient_name,
    patients_in,
    range_percent,
)

EDGES = np.arange(-1000, 1601, 20)  # 20 HU ranges, -1000 to 1600; range_percent folds values >= 1600 into the last one
FLOOR_PCT = 1e-3  # smallest percentage drawn; smaller (including zero) is drawn at height 0
POOLED_COLOR = "0.85"
SCAN_COLOR = PALETTE[0]
LABEL_COLOR = PALETTE[1]


def log_height(pct: np.ndarray, floor: float = FLOOR_PCT) -> np.ndarray:
    """Map a percentage to a log-scaled bar height, floored so 0% doesn't reach -inf.

    Args:
        pct: Percentage values (0-100).
        floor: Smallest percentage represented; values at or below it map to 0.

    Returns:
        log10(max(pct, floor)) - log10(floor): 0 at `floor`, growing with pct.
    """
    return np.log10(np.clip(pct, floor, None)) - np.log10(floor)


def set_mirrored_log_ticks(ax, values: tuple[float, ...] = (0.001, 1, 30), floor: float = FLOOR_PCT) -> None:
    """Set y ticks labeled with percentages, mirrored above and below zero.

    Args:
        ax: Axes whose y data is log_height(pct) above zero and -log_height(pct) below.
        values: Percentage values to tick, smallest (the floor) first.
        floor: Same floor passed to log_height.
    """
    heights = log_height(np.array(values), floor)
    positions = list(-heights[::-1]) + list(heights[1:])
    labels = [f"{v:g}" for v in values[::-1]] + [f"{v:g}" for v in values[1:]]
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)


def main():
    """Draw the whole-scan-vs-labeled-voxels small multiples from the intensity histograms."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-dir", type=Path, default=PROFILE_DIR)
    args = ap.parse_args()

    hists = load_tables(args.profile_dir)["histograms"]
    norm = nnunet_ct_normalization(hists)
    pids = patients_in(hists)

    scan_pct = {p: range_percent(*hists[(p, "scan")], EDGES) for p in pids}
    label_pct = {p: range_percent(*hists[(p, "all labels")], EDGES) for p in pids}
    pooled = log_height(np.mean(list(scan_pct.values()), axis=0))

    apply_ticks_style()
    fig, axes = plt.subplots(4, 5, figsize=(18, 13), sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), pids):
        ax.stairs(pooled, EDGES, baseline=0, color=POOLED_COLOR, fill=True, zorder=1)
        ax.stairs(log_height(scan_pct[p]), EDGES, baseline=0, color=SCAN_COLOR, alpha=0.9, fill=True, zorder=2)
        ax.stairs(-log_height(label_pct[p]), EDGES, baseline=0, color=LABEL_COLOR, alpha=0.95, fill=True, zorder=2)
        ax.axhline(0, color="white", lw=1.5, zorder=3)
        clip_lines(ax, norm)
        ax.set_title(patient_name(p), fontsize=13, fontweight="bold", loc="left")
        sns.despine(ax=ax)

    axes[0, 0].set_ylim(-5, 5)
    axes[0, 0].set_xlim(EDGES[0] - 10, EDGES[-1] + 10)
    axes[0, 0].set_xticks([-1000, -500, 0, 500, 1000, 1600])
    axes[0, 0].set_xticklabels(["−1000", "−500", "0", "500", "1000", "1600+"])
    set_mirrored_log_ticks(axes[0, 0])
    for ax in axes[-1]:
        ax.set_xlabel("intensity (HU)")
    for ax in axes[:, 0]:
        ax.set_ylabel("% of voxels (log)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SCAN_COLOR, label="whole scan (up)"),
        plt.Rectangle((0, 0), 1, 1, color=LABEL_COLOR, label="voxels in labels 1-3 (down)"),
        plt.Rectangle((0, 0), 1, 1, color=POOLED_COLOR, label="whole scan, mean of all 20 patients"),
        plt.Line2D(
            [],
            [],
            color="#555555",
            ls="--",
            label=f"nnU-Net clip {norm['clip_low']:.0f}/{norm['clip_high']:.0f}".replace("-", "−"),
        ),
    ]
    fig.tight_layout(rect=(0, 0.06, 1, 0.88))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=12)
    title_block(
        fig,
        "Scan Intensity: Whole Scan vs Labeled Voxels, per Patient",
        "Up: every voxel of the whole scan. Down: only voxels inside labels 1–3. "
        "Each bar = voxels whose value falls in one 20 HU range, e.g. −1000 to −981.",
    )

    out_path = output_path(args.profile_dir, "scan_vs_labels.png")
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
