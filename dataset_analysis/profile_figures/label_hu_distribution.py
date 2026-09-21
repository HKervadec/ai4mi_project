#!/usr/bin/env python3
"""
HU distribution of every label with voxels in the profiled dataset, all labels together
as opaque outlined bars (one color per organ) in thousand voxels per 10 HU
bin, summed over all patients of the dataset. The bars are not stacked: each starts at zero and the smaller bar of a bin is
drawn in front, so a taller bar shows only the part that sticks out above it. The
stretch between air and soft tissue is drawn narrower and marked with a break.

Give --profile-dir two folders (e.g. the 3-label data and the 4-label data) and the
first is drawn upwards, the second hangs downwards from the same HU axis, which runs
between the two, so a label only the second has appears only in the lower half. With
one folder only the upward half is drawn.

Reads intensity_histograms.npz and labels.csv written by tools/dataset_profile.py.

Usage:
    python dataset_analysis/profile_figures/label_hu_distribution.py --profile-dir figures/profile

    python dataset_analysis/profile_figures/label_hu_distribution.py \
        --profile-dir figures/segthor_part1/profile figures/full_release/profile \
        --names "3 labels" "4 labels" --out-dir figures/comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from dataset_profile import hist_stats, load_tables  # noqa: E402
from plot_style import EARTH_LABEL_COLORS, tint  # noqa: E402
from profile_figures.common import apply_ticks_style, out_subdir, title_block  # noqa: E402
from profile_figures.label_intensity import binned_counts  # noqa: E402
from profile_figures.scan_intensity.common import pool_histograms  # noqa: E402
from utils import CLASSES  # noqa: E402

LOW_HU = -1050
SQUEEZED_HU = (-900, -200)  # the stretch between air and soft tissue, drawn narrower
SQUEEZE = 6  # how many times narrower
MIN_TISSUE_HIGH = 300
NNUNET_FOREGROUND_SAMPLES = 10e7  # labeled voxels nnU-Net samples over all cases for its statistics
BIN_HU = 10
WIDTH_IN = 12
HALF_IN = 3.3  # height of one half
HEADROOM = 1.05
Y_STEP = 5  # the axis height is rounded up to a multiple of this, in thousand voxels
VOXEL_SCALE = 1e3  # voxels are drawn in thousands
FADED_LABEL = 2  # the heart is drawn lighter so the other labels stand out
FADED_TINT = 0.4
STRONG_TINT = 0.8
THIN_OUTLINE = 0.2
AXIS_WIDTH = 2.2  # the HU axis between the two halves
AXIS_COLOR = "#222222"
JUMP_COLOR = "#7B6FA6"  # marks the squeezed stretch; no bar uses it
GRID_COLOR = "#E3E3E3"
LIGHT_BOLD = [pe.withStroke(linewidth=0.4, foreground="#262626")]  # axis labels: a stroke thickens the letters lightly
FIXED_TICKS = (-1000, -900, -600)  # labeled ticks below the soft-tissue window
NNUNET_COLOR = "#111111"  # nnU-Net's statistics of the labeled voxels
NNUNET_LINES = (  # label, key in foreground_stats(), line style, side of its line (1 right, -1 left), label height in the top axis
    ("0.5th pct.", "p0.5", "--", 1, 0.975),
    ("mean", "mean", "-.", -1, 0.975),
    ("median", "median", ":", 1, 0.975),
    ("99.5th pct.", "p99.5", "--", -1, 0.975),
)


def tissue_high(p99_5: float) -> int:
    """Upper edge of the plotted HU window.

    Args:
        p99_5: Highest 99.5th percentile HU over the labels.

    Returns:
        At least MIN_TISSUE_HIGH, otherwise the percentile plus a margin, rounded up to 50 HU.
    """
    return max(MIN_TISSUE_HIGH, int(np.ceil((p99_5 + 50) / 50) * 50))


def share_outside(counts: np.ndarray, start: int, windows: list[tuple[int, int]]) -> float:
    """Percentage of voxels whose HU lies outside every window.

    Args:
        counts: Number of voxels at each HU value.
        start: HU value of counts[0].
        windows: Half-open (low, high) HU windows.

    Returns:
        Percentage of the histogram's voxels outside all windows.
    """
    hu = np.arange(start, start + len(counts))
    inside = np.zeros(len(counts), dtype=bool)
    for lo, hi in windows:
        inside |= (hu >= lo) & (hu < hi)
    return 100 * counts[~inside].sum() / counts.sum()


def warp(hu) -> np.ndarray:
    """Map HU to plotted x, drawing SQUEEZED_HU SQUEEZE times narrower.

    Args:
        hu: HU value or array of values.

    Returns:
        The plotted x positions; identical to hu below the squeezed stretch.
    """
    hu = np.asarray(hu, dtype=float)
    lo, hi = SQUEEZED_HU
    inside = lo + (hu - lo) / SQUEEZE
    return np.where(hu < lo, hu, np.where(hu <= hi, inside, lo + (hi - lo) / SQUEEZE + hu - hi))


def tick_positions(high: int) -> tuple[list[int], list[int]]:
    """HU values of the labeled (major) and unlabeled (minor) ticks of the x axis.

    Args:
        high: Upper HU edge of the plotted window.

    Returns:
        Major ticks: FIXED_TICKS, then every 100 HU from the soft-tissue window's lower edge.
        Minor ticks: every 50 HU outside the squeezed stretch and every 100 HU inside it,
        without the major ones.
    """
    lo, hi = SQUEEZED_HU
    major = [*FIXED_TICKS, *range(hi, high + 1, 100)]
    minor = [t for t in range(-1050, high + 1, 50) if t not in major and (t <= lo or t >= hi or t % 100 == 0)]
    return major, minor


def bar_style(key, color: str) -> dict:
    """Fill, outline color and outline width of a label's bars.

    Args:
        key: Label number.
        color: The label's color.

    Returns:
        Keyword arguments for a bar or legend patch: the heart is lighter, the other labels
        strong.
    """
    if key == FADED_LABEL:
        return {"facecolor": tint(color, FADED_TINT), "edgecolor": tint(color, 0.85), "linewidth": 0.8}
    return {"facecolor": tint(color, STRONG_TINT), "edgecolor": color, "linewidth": 1.2}


def draw_order(heights: np.ndarray) -> np.ndarray:
    """Drawing rank of every bar so that the smaller bar of a bin is drawn in front.

    Args:
        heights: Bar heights, one row per label and one column per bin.

    Returns:
        Integer array shaped like heights; in each bin 0 is the tallest bar (drawn first,
        at the back) and the highest rank is the smallest (drawn last, in front).
    """
    return np.argsort(np.argsort(-heights, axis=0, kind="stable"), axis=0, kind="stable")


def foreground_stats(per_patient: list[tuple[np.ndarray, int]]) -> dict:
    """Statistics as nnU-Net's fingerprint takes them from the labeled voxels.

    nnU-Net draws the same number of labeled voxels from every case, with replacement, so
    a small case counts as much as a large one. This uses the expected result of that draw.

    Args:
        per_patient: (counts, start) 1 HU histogram of all labeled voxels of each patient.

    Returns:
        hist_stats() of the resampled voxels: p0.5 and p99.5 are the clip bounds, mean and
        std the z-score constants.
    """
    per_case = int(NNUNET_FOREGROUND_SAMPLES // len(per_patient))
    resampled = [(np.rint(counts * per_case / counts.sum()).astype(np.int64), start) for counts, start in per_patient]
    return hist_stats(*pool_histograms(resampled))


def load_dataset(profile_dir: Path) -> dict:
    """Histograms, labels with voxels and pooled statistics of one profiled dataset.

    Args:
        profile_dir: Folder written by tools/dataset_profile.py.

    Returns:
        Dict with patients, labels (those with voxels), pooled ((counts, start) per label),
        stats per label and foreground (all labeled voxels together).
    """
    tables = load_tables(profile_dir)
    hists = tables["histograms"]
    labels_table = tables["labels"]
    labels = sorted(int(v) for v in labels_table.loc[labels_table.voxels > 0, "label"].unique())
    patients = sorted({p for p, _ in hists})
    pooled = {}
    for k in labels:
        parts = [hists[(p, f"label {k}")] for p in patients]
        pooled[k] = pool_histograms([part for part in parts if len(part[0])])
    stats = {k: hist_stats(*pooled[k]) for k in labels}
    per_patient = []
    for p in patients:
        parts = [hists[(p, f"label {k}")] for k in labels]
        per_patient.append(pool_histograms([part for part in parts if len(part[0])]))
    return {
        "patients": patients, "labels": labels, "pooled": pooled, "stats": stats, "foreground": foreground_stats(per_patient),
    }


def bin_values(data: dict, key: int, high: int) -> tuple[np.ndarray, np.ndarray]:
    """Thousand voxels, summed over all patients, in every BIN_HU bin of [LOW_HU, high).

    Args:
        data: Dataset returned by load_dataset().
        key: Label number.
        high: Upper HU edge of the plotted window.

    Returns:
        Bin centers and the label's thousand voxels in each bin.
    """
    x, counts = binned_counts(*data["pooled"][key], LOW_HU, high, BIN_HU)
    return x, counts / VOXEL_SCALE


def draw_half(ax, data: dict, colors: dict, high: int, ymax: float, upward: bool, name: str | None) -> None:
    """Draw every label of a dataset as opaque outlined bars, upwards or hanging downwards.

    Args:
        ax: Axes to draw on.
        data: Dataset returned by load_dataset().
        colors: Color of every label.
        high: Upper HU edge of the plotted window.
        ymax: Height of the axis in thousand voxels.
        upward: True to draw up from the bottom edge, False to hang down from the top edge.
        name: Dataset name written in the corner, or None for no name.
    """
    bins = [bin_values(data, k, high) for k in data["labels"]]
    heights = np.array([y for _, y in bins])
    rank = draw_order(heights)
    for g, k in enumerate(data["labels"]):
        shown = heights[g] > 0
        left, right = warp(bins[g][0][shown] - BIN_HU / 2), warp(bins[g][0][shown] + BIN_HU / 2)
        bars = ax.bar(left, heights[g][shown], width=right - left, align="edge", **bar_style(k, colors[k]))
        for bar, r in zip(bars, rank[g][shown]):
            bar.set_zorder(3 + r)
            if bar.get_width() < BIN_HU / 2:  # squeezed bars are too thin for a full outline
                bar.set_linewidth(THIN_OUTLINE)
    ax.set_ylim((0, ymax) if upward else (ymax, 0))
    ax.set_xlim(warp(LOW_HU), warp(high))
    ax.axvspan(
        *warp(SQUEEZED_HU), facecolor=tint(JUMP_COLOR, 0.12), edgecolor=JUMP_COLOR, linestyle="--", linewidth=1, zorder=0.6
    )
    major, minor = tick_positions(high)
    ax.set_xticks(warp(major), [str(t) for t in major])
    ax.set_xticks(warp(minor), minor=True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", which="major", length=6, pad=4)
    ax.tick_params(axis="x", which="minor", length=3.5)
    ax.tick_params(axis="x", which="both", bottom=upward, top=not upward, labelbottom=False, labeltop=False)
    ax.set_ylabel(f"thousand voxels, all patients\nper {BIN_HU} HU bin", fontsize=12, path_effects=LIGHT_BOLD)
    for side in ("right", "top", "bottom"):
        ax.spines[side].set_visible(False)
    edge = "bottom" if upward else "top"  # the HU axis between the two halves
    ax.spines[edge].set_visible(True)
    ax.spines[edge].set_linewidth(AXIS_WIDTH)
    ax.spines[edge].set_color(AXIS_COLOR)
    if name:
        ax.text(
            warp(high), 1.015 if upward else -0.02, name, transform=ax.get_xaxis_transform(), ha="right",
            va="bottom" if upward else "top", fontsize=13, fontweight="bold", color="#444444",
        )


def mark_jump(ax, upward: bool) -> None:
    """Break marks on the HU axis at both ends of the squeezed stretch, and a label inside it.

    Args:
        ax: Axes drawn in front of the HU axis.
        upward: True if the HU axis is the bottom edge of the axes, False if it is the top edge.
    """
    y0 = 0 if upward else 1
    for x0 in warp(SQUEEZED_HU):
        for offset in (-4, 4):
            ax.plot(
                [x0 + offset - 5, x0 + offset + 5], [y0 - 0.03, y0 + 0.03], transform=ax.get_xaxis_transform(),
                color=JUMP_COLOR, linewidth=2.5, clip_on=False, zorder=30,
            )
    ax.text(
        warp(sum(SQUEEZED_HU) / 2), 0.97 if upward else 0.03, f"shortened axis\nHU {SQUEEZED_HU[0]} to {SQUEEZED_HU[1]}",
        transform=ax.get_xaxis_transform(), color=JUMP_COLOR, ha="center", va="top" if upward else "bottom",
        fontsize=10.5, fontweight="bold", zorder=30,
    )


def main():
    """Draw the labels of one dataset upwards and of a second one hanging below the HU axis."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-dir", type=Path, nargs="+", default=[Path("figures/profile")],
                    help="One or two folders written by tools/dataset_profile.py; the second hangs below the axis.")
    ap.add_argument("--names", nargs="+", help="Name of each --profile-dir; default before/after when two.")
    ap.add_argument("--out-dir", type=Path, help="Where to write the PNG; default <profile-dir>/label_hu_distribution.")
    args = ap.parse_args()
    if len(args.profile_dir) > 2:
        ap.error("--profile-dir takes one or two folders")
    names = args.names or (["before", "after"] if len(args.profile_dir) == 2 else [None])
    if len(names) != len(args.profile_dir):
        ap.error("--names needs one name per --profile-dir")

    datasets = [load_dataset(d) for d in args.profile_dir]
    labels = sorted({k for data in datasets for k in data["labels"]})
    high = tissue_high(max(data["stats"][k]["p99.5"] for data in datasets for k in data["labels"]))
    peak = max(bin_values(data, k, high)[1].max() for data in datasets for k in data["labels"])
    ymax = Y_STEP * np.ceil(HEADROOM * peak / Y_STEP)

    apply_ticks_style()
    height = HALF_IN * len(datasets) + 2.75
    fig, axes = plt.subplots(
        len(datasets), 1, sharex=True, squeeze=False, figsize=(WIDTH_IN, height), gridspec_kw={"hspace": 0.11}
    )
    axes = axes[:, 0]
    for d, (ax, data) in enumerate(zip(axes, datasets)):
        draw_half(ax, data, EARTH_LABEL_COLORS, high, ymax, upward=d == 0, name=names[d])
    axes[0].tick_params(axis="x", labelbottom=True)  # one row of numbers, between the two axes
    axes[0].annotate(
        "intensity (HU)", xy=(0, 0), xycoords="axes fraction", xytext=(-12, -16), textcoords="offset points",
        ha="right", va="center", fontsize=13, path_effects=LIGHT_BOLD,
    )
    for d, ax in enumerate(axes):
        mark_jump(ax, upward=d == 0)
    fg = datasets[-1]["foreground"]  # the last dataset is the one nnU-Net would train on
    for name, key, style, side, height_in_axis in NNUNET_LINES:
        x = warp(fg[key])
        for ax in axes:
            ax.axvline(x, color=NNUNET_COLOR, linestyle=style, linewidth=1.3, zorder=22)
        axes[0].text(
            x + 4 * side, height_in_axis, f"{name}\n{fg[key]:.0f}", transform=axes[0].get_xaxis_transform(),
            ha="left" if side > 0 else "right", va="top", multialignment="left" if side > 0 else "right",
            fontsize=10.5, fontweight="bold", color=NNUNET_COLOR, linespacing=1.2, zorder=30,
        )
    handles = [Patch(label=CLASSES[k], **bar_style(k, EARTH_LABEL_COLORS[k])) for k in labels]
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, fontsize=12, bbox_to_anchor=(0.985, 0.985))
    outside = {}
    for data in datasets:
        for k in data["labels"]:
            outside[k] = max(outside.get(k, 0), share_outside(*data["pooled"][k], [(LOW_HU, high)]))
    hidden = ", ".join(f"{CLASSES[k]} {v:.0f}%" for k, v in outside.items() if v >= 1)
    counts = [len(data["patients"]) for data in datasets]
    n_patients = f"{counts[0]} per dataset" if len(counts) == 2 and counts[0] == counts[1] else " and ".join(map(str, counts))
    caption = [
        f"CT intensity (HU) inside each annotated organ, pooled over all patients (n={n_patients}); "
        f"bars = labeled voxels per {BIN_HU} HU bin.",
    ]
    title_block(fig, "HU Intensity of Each Label", "\n".join(caption))
    if hidden:
        fig.text(0.12, 0.012, f"HU above {high} is not shown (share of the organ's voxels there: {hidden}).", fontsize=10.5, color="#666666")
    fig.subplots_adjust(left=0.12, right=0.985, top=1 - 1.3 / height, bottom=0.4 / height)
    out_dir = args.out_dir or out_subdir(args.profile_dir[0], "label_hu_distribution")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "label_hu_distribution.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
