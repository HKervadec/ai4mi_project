#!/usr/bin/env python3
"""
Label intensity: HU histograms of the image values inside each labeled organ
with voxels in this dataset (1-3 for the original course release, 1-4 once
the aorta annotation is present), one ridge row per patient plus a row
pooling all patients, with nnU-Net's foreground intensity sample and its
clip values overlaid.

Reads intensity_histograms.npz written by tools/dataset_profile.py.

Usage:
    python dataset_analysis/profile_figures/label_intensity.py --profile-dir figures/profile
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from dataset_profile import HU_OFFSET, hist_stats, load_tables  # noqa: E402
from profile_figures.common import LABEL_COLORS, apply_ticks_style, build_arg_parser, header, out_subdir  # noqa: E402

ALL_GROUPS = [f"label {n}" for n in LABEL_COLORS]
COLORS = {f"label {n}": color for n, color in LABEL_COLORS.items()}
SAMPLE = "nnunet sample"
CLIP_COLOR = "#D1495B"
HU_RANGE = (-1050, 550)
BIN_HU = 10
ROW_HEIGHT = 1.6  # peak height of a histogram, in row spacings


def binned_counts(counts: np.ndarray, start: int, lo: int, hi: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum a 1 HU histogram into fixed-width bins covering [lo, hi).

    Values outside [lo, hi) are dropped.

    Args:
        counts: Number of voxels at each HU value.
        start: HU value of counts[0].
        lo: Lower edge of the first bin.
        hi: Upper edge of the last bin; (hi - lo) must be a multiple of width.
        width: Bin width in HU.

    Returns:
        Bin centers and voxel counts per bin.
    """
    dense = np.zeros(hi - lo)
    a, b = max(start, lo), min(start + len(counts), hi)
    if a < b:
        dense[a - lo : b - lo] = counts[a - start : b - start]
    return np.arange(lo, hi, width) + width / 2, dense.reshape(-1, width).sum(axis=1)


def draw_row(ax, base: float, x: np.ndarray, counts: np.ndarray, **style) -> None:
    """Draw one step histogram scaled to its own peak, sitting on a ridge baseline."""
    y = base + counts / counts.max() * ROW_HEIGHT
    if style.pop("fill", False):
        ax.fill_between(x, base, y, step="mid", color=style["color"], alpha=0.55, lw=0, zorder=style["zorder"])
    ax.step(x, y, where="mid", **style)


def main():
    """Draw the label intensity ridge figure from intensity_histograms.npz."""
    ap = build_arg_parser(__doc__)
    args = ap.parse_args()

    hists = load_tables(args.profile_dir)["histograms"]
    patients = sorted({p for p, _ in hists})
    bins = {key: binned_counts(*hists[key], *HU_RANGE, BIN_HU) for key in hists if key[1] in ALL_GROUPS + [SAMPLE]}
    x = next(iter(bins.values()))[0]
    # Labels with at least one voxel somewhere in this dataset -- not hardcoded,
    # so the same figure works whether the aorta annotation (label 4) is present.
    # A histogram with no voxels sums to all zero bins, which would otherwise
    # divide by zero when draw_row scales it to its own peak.
    GROUPS = [g for g in ALL_GROUPS if sum(bins[(p, g)][1].sum() for p in patients) > 0]
    # nnU-Net clips CT to the 0.5th / 99.5th percentile of its pooled foreground sample
    sample = sum(binned_counts(*hists[(p, SAMPLE)], -HU_OFFSET, HU_OFFSET, 1)[1] for p in patients)
    stats = hist_stats(sample, -HU_OFFSET)
    clip = (stats["p0.5"], stats["p99.5"])
    rows = [("all patients", {g: sum(bins[(p, g)][1] for p in patients) for g in GROUPS + [SAMPLE]})]
    rows += [(p.replace("Patient_", "patient "), {g: bins[(p, g)][1] for g in GROUPS + [SAMPLE]}) for p in patients]

    apply_ticks_style()
    fig, ax = plt.subplots(figsize=(15, 15))
    for i, (name, counts) in enumerate(rows):
        base = -i
        for g in GROUPS:
            draw_row(ax, base, x, counts[g], color=COLORS[g], lw=0.8, zorder=10 - i, fill=True)
        draw_row(ax, base, x, counts[SAMPLE], color="black", lw=0.9, ls=(0, (2, 1.5)), zorder=11 - i)
        ax.axhline(base, color="#999999", lw=0.5, zorder=0)
        ax.text(HU_RANGE[0] - 15, base + 0.15, name, ha="right", fontsize=12, fontweight="bold" if i == 0 else "normal")
    for v in clip:
        ax.axvline(v, color=CLIP_COLOR, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(clip[1] + 8, 1, "nnU-Net clip", transform=ax.get_xaxis_transform(), color=CLIP_COLOR, va="top")
    ax.set_xlim(*HU_RANGE)
    ax.set_ylim(-len(rows) + 0.6, ROW_HEIGHT + 0.3)
    ax.set_yticks([])
    ax.set_xlabel("HU", fontsize=13)
    sns.despine(ax=ax, left=True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[g], alpha=0.6, label=g) for g in GROUPS] + [
        plt.Line2D([], [], color="black", ls=(0, (2, 1.5)), label="nnU-Net sample (5M foreground voxels per patient)"),
        plt.Line2D([], [], color=CLIP_COLOR, ls=(0, (4, 3)), label=f"nnU-Net clip ({clip[0]:.0f}, {clip[1]:.0f} HU)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(GROUPS) + 2, frameon=False, fontsize=12)
    label_nums = [g.split()[1] for g in GROUPS]
    label_range = f"{label_nums[0]}–{label_nums[-1]}" if len(label_nums) > 1 else label_nums[0]
    header(
        fig,
        "Label Intensity per Patient",
        f"One curve per label ({label_range}) per patient, scaled to its own peak, in {BIN_HU} HU bins; dashed "
        "black = nnU-Net's sample, dashed red = its clip; top row pools all patients.",
        subtitle_y=0.955,
    )
    fig.tight_layout(rect=(0.06, 0.03, 1, 0.945))
    out = out_subdir(args.profile_dir, "label_intensity") / "label_intensity.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
