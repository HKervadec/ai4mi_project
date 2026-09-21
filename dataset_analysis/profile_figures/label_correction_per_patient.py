#!/usr/bin/env python3
"""
Label make-up and connected pieces before and after the aorta correction, one line per
patient. Every panel is one organ; the top row is its labeled volume (mL), the bottom row
the number of face-connected pieces of the label in 3D, and each patient's line runs from the
3-label release (open dot) to the 4-label release (filled dot). Above each organ its share of
all labeled voxels, patients pooled, is written for both releases.

Built as a seaborn FacetGrid (one row per measure, one column per organ). Reads labels.csv and
patients.csv written by tools/dataset_profile.py for each release.

Usage:
    python dataset_analysis/profile_figures/label_correction_per_patient.py \
        --profile-dir figures/before/profile figures/profile \
        --names "3 labels (aorta merged)" "4 labels (aorta separate)" --out-dir figures/comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from plot_style import EARTH_LABEL_COLORS, tint  # noqa: E402
from profile_figures.common import title_block  # noqa: E402
from utils import CLASSES  # noqa: E402

MEASURES = {"volume_ml": "volume (mL)", "pieces": "connected pieces\n(3D, faces)"}
INK = "#333333"


def long_table(labels_tables: list[pd.DataFrame], names: list[str]) -> pd.DataFrame:
    """One row per release, patient and organ, for the organs that have voxels in any release.

    Args:
        labels_tables: labels.csv of each release.
        names: Name of each release.

    Returns:
        Columns patient, label, voxels, volume_ml, pieces (face-connected, 0 for an absent
        label), labels (release name) and x (position of the release, 0 for the first).
    """
    parts = [
        table.rename(columns={"pieces_sharing_a_face": "pieces"})[["patient", "label", "voxels", "volume_ml", "pieces"]].assign(
            labels=name, x=x
        )
        for x, (table, name) in enumerate(zip(labels_tables, names))
    ]
    long = pd.concat(parts, ignore_index=True)
    long["pieces"] = long.pieces.fillna(0)
    present = long.groupby("label").voxels.sum() > 0
    return long[long.label.isin(present[present].index)]


def labeled_share(long: pd.DataFrame) -> pd.DataFrame:
    """Share of all labeled voxels of each organ, patients pooled.

    Args:
        long: Table from long_table().

    Returns:
        Percentages indexed by release name with one column per organ.
    """
    voxels = long.pivot_table(index="labels", columns="label", values="voxels", aggfunc="sum")
    return 100 * voxels.div(voxels.sum(axis=1), axis=0)


def background_share(patients: pd.DataFrame, labels: pd.DataFrame) -> float:
    """Percentage of all scan voxels that carry no label.

    Args:
        patients: patients.csv with the voxel count of every scan.
        labels: labels.csv with the voxel count of every label.

    Returns:
        Percentage of the pooled scan voxels outside all labels.
    """
    return 100 * (1 - labels.voxels.sum() / patients.voxels.sum())


def draw_lines(ax, data: pd.DataFrame, color, highlight: str | None) -> None:
    """Draw every patient's line from the first to the second release.

    Args:
        ax: Axes to draw on.
        data: Rows of one organ and one measure, with columns patient, x and value.
        color: Color of the organ.
        highlight: Patient drawn in full color; the others are lighter.
    """
    others = data[data.patient != highlight]
    sns.lineplot(data=others, x="x", y="value", units="patient", estimator=None, color=tint(color, 0.5), linewidth=1.1, ax=ax)
    for frame, width, size in ((others, 1.1, 30), (data[data.patient == highlight], 2.6, 70)):
        if frame.empty:
            continue
        if width > 2:
            sns.lineplot(data=frame, x="x", y="value", units="patient", estimator=None, color=color, linewidth=width, ax=ax)
        first, last = frame[frame.x == 0], frame[frame.x == frame.x.max()]
        ax.scatter(first.x, first.value, s=size, facecolor="white", edgecolor=color, linewidth=1.4, zorder=4)
        ax.scatter(last.x, last.value, s=size, facecolor=color, edgecolor=color, linewidth=1.4, zorder=4)


def main():
    """Draw the per-patient before/after grid of label volume and connected pieces."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-dir", type=Path, nargs=2, default=[Path("figures/before/profile"), Path("figures/profile")],
                    help="Profile folders of the release before and the release after the correction.")
    ap.add_argument("--names", nargs=2, default=["3 labels (aorta merged)", "4 labels (aorta separate)"])
    ap.add_argument("--highlight", default="Patient_02", help="Patient drawn in full color.")
    ap.add_argument("--out-dir", type=Path, default=Path("figures/comparison"))
    args = ap.parse_args()

    labels_tables = [pd.read_csv(d / "labels.csv") for d in args.profile_dir]
    patients_tables = [pd.read_csv(d / "patients.csv") for d in args.profile_dir]
    long = long_table(labels_tables, args.names)
    shares = labeled_share(long)
    organs = sorted(long.label.unique())
    tidy = long.melt(id_vars=["patient", "label", "labels", "x"], value_vars=list(MEASURES), var_name="measure")

    sns.set_theme(style="whitegrid", font_scale=1.05)
    g = sns.FacetGrid(tidy, row="measure", col="label", sharey=False, sharex=True, height=2.3, aspect=1.2, margin_titles=False)
    g.figure.set_size_inches(12.5, 6.3)
    for (row, col), ax in g.axes_dict.items():
        organ = EARTH_LABEL_COLORS[col]
        draw_lines(ax, tidy[(tidy.measure == row) & (tidy.label == col)], organ, args.highlight)
        ax.set(xlim=(-0.35, 1.35), xlabel="", ylabel="")
    for i, measure in enumerate(MEASURES):
        column_max = [tidy[(tidy.measure == measure) & (tidy.label == k)].value.max() for k in organs]
        for j, k in enumerate(organs):
            ax = g.axes[i, j]
            ax.set_title("")
            ax.set_ylim(0, max(column_max[j], 2 if measure == "pieces" else 0) * 1.1)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.set_xticks([0, 1], ["3 labels", "4 labels"] if i == len(MEASURES) - 1 else ["", ""])
            ax.grid(axis="x", visible=False)
        g.axes[i, 0].set_ylabel(MEASURES[measure], fontsize=12, fontweight="bold", labelpad=10)
    for j, k in enumerate(organs):
        g.axes[0, j].set_title(CLASSES[k], loc="left", fontsize=15, fontweight="bold", color=EARTH_LABEL_COLORS[k], pad=26)
        before, after = (shares.loc[name, k] for name in args.names)
        g.axes[0, j].text(0, 1.03, f"{before:.1f}% → {after:.1f}% of labeled voxels", transform=g.axes[0, j].transAxes, fontsize=10.5, color=INK)
    sns.despine(g.figure, left=True, bottom=True)
    handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=INK, label=args.names[0]),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=INK, markeredgecolor=INK, label=args.names[1]),
        Line2D([], [], color=INK, linewidth=2.6, label=f"{args.highlight} (example slide)"),
    ]
    g.figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11.5, bbox_to_anchor=(0.55, 0.0))
    backgrounds = [background_share(p, lab) for p, lab in zip(patients_tables, labels_tables)]
    background = " and ".join(dict.fromkeys(f"{v:.2f}%" for v in backgrounds))
    n = long.patient.nunique()
    title_block(
        g.figure, "Label Make-Up and Connected Pieces per Patient",
        f"One line per patient (n={n}), from the 3-label to the 4-label release; percentages are each organ's share of all labeled voxels, "
        f"patients pooled.\nBackground is {background} of all voxels.",
    )
    height = g.figure.get_size_inches()[1]
    g.figure.subplots_adjust(left=0.1, right=0.985, top=1 - 2.05 / height, bottom=0.95 / height, wspace=0.32, hspace=0.25)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "label_correction_per_patient.png"
    g.figure.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
