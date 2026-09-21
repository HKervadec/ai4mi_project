#!/usr/bin/env python3
"""
Slices per label and slice-to-slice area change, for whichever labels have
voxels in the profiled dataset (1-3 for the original course release, 1-4
once the aorta annotation is present).

Writes two figures, both topped by a stacked histogram of how many slices
contain each label:
  slices_and_change_bands.png  label area and area change along the label's
                               slices: median across patients with middle 50%
                               and middle 90% bands, labels overlapped.
  slices_and_change_joint.png  one 2D histogram per label of every neighboring
                               slice pair: position within the label against
                               area change, with the change histogram alongside.

Position within a label runs from 0% (its lowest slice) to 100% (its highest),
so patients with different slice counts line up. Area change is
100 x (area - area of the slice below) / area of the slice below.

Reads labels.csv and label_slices.csv written by tools/dataset_profile.py.

Usage:
    python dataset_analysis/profile_figures/slices_and_change.py --profile-dir figures/profile
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profile_figures.common import LABEL_COLORS, apply_ticks_style, build_arg_parser, header, out_subdir  # noqa: E402

PALETTE = {f"label {k}": v for k, v in LABEL_COLORS.items()}
POSITION_BIN_PCT = 4
CHANGE_LIMIT_PCT = 60  # 2D histogram range; larger changes are placed on its edge
POSITION_LABEL = "position within the label's slices (% from its lowest to highest slice)"
CHANGE_LABEL = "area change from the slice below (%)"


def slice_changes(slices: pd.DataFrame) -> pd.DataFrame:
    """Add position, relative area and slice-to-slice change to per-slice label rows.

    Args:
        slices: label_slices.csv rows with patient, label, slice and area_mm2.

    Returns:
        Copy sorted by patient, label and slice, with pos_pct (0 at the label's lowest
        slice, 100 at its highest), area_pct_max (% of that label's largest slice area)
        and change_pct (% change from the slice below; NaN on the lowest slice).
    """
    df = slices.sort_values(["patient", "label", "slice"]).reset_index(drop=True)
    g = df.groupby(["patient", "label"])
    n = g["slice"].transform("size")
    df["pos_pct"] = np.where(n > 1, 100 * g.cumcount() / (n - 1).clip(lower=1), 0.0)
    df["area_pct_max"] = 100 * df["area_mm2"] / g["area_mm2"].transform("max")
    df["change_pct"] = 100 * g["area_mm2"].diff() / g["area_mm2"].shift()
    return df


def _slices_histogram(ax, labels: pd.DataFrame, present_labels: list[int]) -> None:
    sns.histplot(
        labels,
        x="slices_with_label",
        hue="name",
        hue_order=[f"label {k}" for k in present_labels[::-1]],
        palette=PALETTE,
        multiple="stack",
        binwidth=4,
        edgecolor=".3",
        linewidth=0.5,
        ax=ax,
    )
    ax.set_xlabel("slices containing the label")
    ax.set_ylabel("patients")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    sns.move_legend(ax, "upper right", title=None, frameon=False)
    ax.set_title("Patients by number of slices with the label", loc="left", fontweight="bold")
    sns.despine(ax=ax)


def _bands(ax, df: pd.DataFrame, column: str, ylabel: str, title: str, present_labels: list[int]) -> None:
    position = (df["pos_pct"] / POSITION_BIN_PCT).round() * POSITION_BIN_PCT
    for lab in present_labels:
        q = df[df["label"] == lab].groupby(position)[column].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
        color = LABEL_COLORS[lab]
        ax.fill_between(q.index, q[0.05], q[0.95], color=color, alpha=0.12, lw=0)
        ax.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=0.3, lw=0)
        ax.plot(q.index, q[0.5], color=color, lw=2.4, label=f"label {lab}")
    ax.set_xlim(0, 100)
    ax.set_xlabel(POSITION_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    sns.despine(ax=ax)


def _fig_bands(df: pd.DataFrame, labels: pd.DataFrame, out_dir: Path, present_labels: list[int]) -> None:
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.2], hspace=0.35, wspace=0.18)
    _slices_histogram(fig.add_subplot(gs[0, 0]), labels, present_labels)
    ax = fig.add_subplot(gs[0, 1])
    _bands(
        ax, df, "area_pct_max", "area on the slice (% of that label's largest slice)", "Label area along its slices",
        present_labels,
    )
    ax.legend(
        handles=[plt.Line2D([], [], color=LABEL_COLORS[k], lw=2.4, label=f"label {k}") for k in present_labels]
        + [
            plt.Line2D([], [], color="#555555", lw=2.4, label="median across patients"),
            plt.Rectangle((0, 0), 1, 1, color="#555555", alpha=0.3, label="middle 50% of patients"),
            plt.Rectangle((0, 0), 1, 1, color="#555555", alpha=0.12, label="middle 90% of patients"),
        ],
        frameon=False,
        fontsize=11,
        loc="upper left",
        ncol=2,
    )
    ax = fig.add_subplot(gs[1, :])
    _bands(ax, df, "change_pct", CHANGE_LABEL, "Slice-to-slice area change along the label", present_labels)
    ax.set_ylim(-40, CHANGE_LIMIT_PCT)
    ax.axhline(0, color="#999999", lw=0.8, zorder=0)
    label_range = f"{present_labels[0]}–{present_labels[-1]}" if len(present_labels) > 1 else str(present_labels[0])
    header(
        fig,
        "Slices per Label and Slice-to-Slice Change",
        f"Median line with middle 50%/90% patient bands per label ({label_range}); position runs lowest (0%) to "
        "highest (100%) slice; left: patients by slice count.",
        0.95,
    )
    fig.subplots_adjust(top=0.9, bottom=0.07, left=0.06, right=0.98)
    out = out_dir / "slices_and_change_bands.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def _fig_joint(df: pd.DataFrame, labels: pd.DataFrame, out_dir: Path, present_labels: list[int]) -> None:
    fig = plt.figure(figsize=(19, 11))
    outer = fig.add_gridspec(2, len(present_labels), height_ratios=[1, 1.6], hspace=0.35, wspace=0.25)
    _slices_histogram(fig.add_subplot(outer[0, :]), labels, present_labels)
    pairs = df.dropna(subset=["change_pct"]).assign(
        change=lambda d: d["change_pct"].clip(-CHANGE_LIMIT_PCT, CHANGE_LIMIT_PCT)
    )
    lim = (-CHANGE_LIMIT_PCT, CHANGE_LIMIT_PCT)
    for k, lab in enumerate(present_labels):
        sub = outer[1, k].subgridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax, side = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        d = pairs[pairs["label"] == lab]
        color = LABEL_COLORS[lab]
        sns.histplot(
            d,
            x="pos_pct",
            y="change",
            bins=(25, 30),
            binrange=((0, 100), lim),
            cmap=sns.light_palette(color, as_cmap=True),
            ax=ax,
        )
        sns.histplot(d, y="change", bins=30, binrange=lim, color=color, element="step", ax=side)
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_ylim(*lim)
        side.set_ylim(*lim)
        ax.set_xlabel("position within the label (% of its slices)")
        ax.set_ylabel(CHANGE_LABEL if k == 0 else "")
        ax.set_title(f"label {lab}", loc="left", fontweight="bold", fontsize=15)
        side.set_xlabel("")
        side.set_ylabel("")
        side.tick_params(labelleft=False, labelbottom=False)
        sns.despine(ax=ax)
        sns.despine(ax=side, bottom=True)
    label_range = f"{present_labels[0]}–{present_labels[-1]}" if len(present_labels) > 1 else str(present_labels[0])
    header(
        fig,
        "Slices per Label and Slice-to-Slice Change",
        f"Top: patients by slice count, labels {label_range} stacked. Bottom: every neighboring slice pair per "
        f"label, all patients pooled; darker = more pairs; changes beyond ±{CHANGE_LIMIT_PCT}% sit on the edge.",
        0.95,
    )
    fig.subplots_adjust(top=0.9, bottom=0.07, left=0.06, right=0.98)
    out = out_dir / "slices_and_change_joint.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main():
    """Draw both slices-per-label and slice-to-slice change figures."""
    args = build_arg_parser(__doc__).parse_args()
    slices = pd.read_csv(args.profile_dir / "label_slices.csv")
    labels = pd.read_csv(args.profile_dir / "labels.csv")
    # Labels with at least one voxel somewhere in this dataset -- not hardcoded,
    # so the same figure works whether the aorta annotation (label 4) is present.
    present_labels = sorted(int(v) for v in labels.loc[labels.voxels > 0, "label"].unique())
    slices = slices[slices["label"].isin(present_labels)]
    labels = labels[labels["label"].isin(present_labels)].assign(name=lambda d: "label " + d["label"].astype(str))

    apply_ticks_style()
    out_dir = out_subdir(args.profile_dir, "slices_and_change")
    df = slice_changes(slices)
    _fig_bands(df, labels, out_dir, present_labels)
    _fig_joint(df, labels, out_dir, present_labels)


if __name__ == "__main__":
    main()
