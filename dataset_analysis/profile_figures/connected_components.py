#!/usr/bin/env python3
"""
Connected components under five connectivity rules, all patients pooled, for
whichever labels have voxels in the given data-dir (1-3 for the original
course release, 1-4 once the aorta annotation is present). Give --before-data-dir
as well and the figure has two rows, that dataset on top and --data-dir below, with
one column per organ, so the effect of the corrected labels can be read down a column; an organ whose
shares are the same in both datasets is drawn in the top row only, and an organ only in the second
dataset takes a free column.

Rules:
  2D · 4   pixels in an axial slice touch through an edge
  2D · 8   pixels in an axial slice touch through an edge or corner
  3D · 6   voxels of the whole label touch through a face
  3D · 18  voxels touch through a face or edge
  3D · 26  voxels touch through a face, edge or corner

For 2D rules every axial slice containing the label counts once; for 3D rules
every patient counts once. Each heatmap cell is the share of slices or patients
in which the label has that many pieces.

Usage:
    python dataset_analysis/profile_figures/connected_components.py \
        --data-dir data/segthor_part1/train --profile-dir figures/profile

    python dataset_analysis/profile_figures/connected_components.py \
        --data-dir data/segthor_part1_corrected/train --before-data-dir data/segthor_part1/train \
        --names "3 labels (aorta merged)" "4 labels (aorta separate)" --out-dir figures/comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import generate_binary_structure, label

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from plot_style import EARTH_LABEL_COLORS  # noqa: E402
from profile_figures.common import build_arg_parser, out_subdir, title_block  # noqa: E402
from utils import CLASSES  # noqa: E402

# All labels this dataset could ever have; the figure narrows to the ones that
# actually have voxels before drawing.
ALL_LABELS = list(CLASSES)
# (rule name, dimensions, scipy connectivity rank)
RULES = [("2D · 4", 2, 1), ("2D · 8", 2, 2), ("3D · 6", 3, 1), ("3D · 18", 3, 2), ("3D · 26", 3, 3)]
MAX_PIECES = 3
# Organs that the first dataset has merged into another label, as new label -> label it was part of.
SPLIT_FROM = {4: 1}


def piece_counts(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Number of connected pieces of one label mask under every rule.

    Args:
        mask: Boolean label mask (x, y, z), z = axial slice index.

    Returns:
        Rule name -> piece counts: one per axial slice containing the label (2D rules),
        or a single count for the whole mask (3D rules). Empty arrays for an empty mask.
    """
    if not mask.any():
        return {name: np.array([], dtype=int) for name, _, _ in RULES}
    mask = mask[tuple(slice(i.min(), i.max() + 1) for i in np.nonzero(mask))]
    slices = np.flatnonzero(mask.any(axis=(0, 1)))
    counts = {}
    for name, dims, rank in RULES:
        structure = generate_binary_structure(dims, rank)
        if dims == 2:
            counts[name] = np.array([label(mask[:, :, z], structure)[1] for z in slices])
        else:
            counts[name] = np.array([label(mask, structure)[1]])
    return counts


def piece_shares(counts: pd.DataFrame) -> pd.DataFrame:
    """Share of slices (2D) or patients (3D) with each number of pieces.

    Args:
        counts: One row per slice or patient with columns label, rule, pieces.

    Returns:
        Percentages indexed by (label, rule) with one column per piece count 1..MAX_PIECES;
        counts above MAX_PIECES are added to the last column.
    """
    capped = counts.assign(pieces=counts.pieces.clip(upper=MAX_PIECES))
    table = capped.groupby(["label", "rule", "pieces"]).size().unstack("pieces", fill_value=0)
    table = table.reindex(columns=range(1, MAX_PIECES + 1), fill_value=0)
    return 100 * table.div(table.sum(axis=1), axis=0)


def unchanged(before: pd.DataFrame, after: pd.DataFrame, label_id: int) -> bool:
    """Whether a label has the same share table in both datasets.

    Args:
        before: piece_shares() of the first dataset.
        after: piece_shares() of the second dataset.
        label_id: Label to compare.

    Returns:
        True if the label is in both tables and every share is equal.
    """
    labels = before.index.get_level_values("label"), after.index.get_level_values("label")
    if label_id not in labels[0] or label_id not in labels[1]:
        return False
    return np.allclose(before.loc[label_id].sort_index(), after.loc[label_id].sort_index())


def column_slots(top: list[int], row: list[int]) -> dict[int, int]:
    """Grid column of every label in a row of heatmaps.

    Args:
        top: Labels drawn in the top row, left to right.
        row: Labels drawn in this row.

    Returns:
        Label -> column. A label also in the top row stays in the same column, so the two
        can be read down a column; the others take the leftmost free columns.
    """
    slots = {k: top.index(k) for k in row if k in top}
    free = (i for i in range(max(len(top), len(row))) if i not in slots.values())
    return {**slots, **{k: next(free) for k in row if k not in top}}


def draw_split_arrow(fig, source, target, color, text: str) -> None:
    """Arrow from one heatmap down the gap between the columns into another heatmap.

    Args:
        fig: Figure to draw on; the axes positions must be final.
        source: Axes the arrow leaves, from its right edge.
        target: Axes the arrow enters, at its left edge.
        color: Color of the arrow and its text.
        text: Caption written along the vertical part of the arrow.
    """
    a, b = source.get_position(), target.get_position()
    x = (a.x1 + b.x0) / 2
    y0, y1 = a.y0 + a.height / 2, b.y0 + b.height / 2
    fig.add_artist(Line2D([a.x1, x, x], [y0, y0, y1], color=color, linewidth=2, transform=fig.transFigure))
    fig.add_artist(FancyArrowPatch((x, y1), (b.x0, y1), arrowstyle="-|>", mutation_scale=16, color=color, linewidth=2, shrinkA=0, shrinkB=0, transform=fig.transFigure))
    fig.text(x + 0.008, (y0 + y1) / 2, text, rotation=90, ha="left", va="center", fontsize=9, color=color)


def count_pieces(data_dir: Path) -> pd.DataFrame:
    """Piece counts of every label of every patient under every rule.

    Args:
        data_dir: Folder with one Patient_* folder per patient, each holding GT.nii.gz.

    Returns:
        One row per slice or patient with columns label, rule, pieces.
    """
    rows = []
    for patient in sorted(data_dir.glob("Patient_*")):
        gt = np.asarray(nib.load(str(patient / "GT.nii.gz")).dataobj)
        for k in ALL_LABELS:
            for rule, values in piece_counts(gt == k).items():
                rows += [{"label": k, "rule": rule, "pieces": int(v)} for v in values]
    return pd.DataFrame(rows)


def main():
    """Count pieces for every patient and draw the pooled heatmaps, one row per dataset."""
    ap = build_arg_parser(__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_part1/train"))
    ap.add_argument("--before-data-dir", type=Path, help="A second dataset, drawn above --data-dir.")
    ap.add_argument("--names", nargs=2, metavar=("BEFORE", "AFTER"), help="Row names when --before-data-dir is given.")
    ap.add_argument("--out-dir", type=Path, help="Where to write the PNG; default <profile-dir>/connected_components.")
    args = ap.parse_args()
    if args.before_data_dir and not args.names:
        ap.error("--before-data-dir needs --names")

    dirs = [d for d in (args.before_data_dir, args.data_dir) if d]
    tables = [count_pieces(d) for d in dirs]
    # Labels with at least one piece somewhere in these datasets -- not hardcoded,
    # so the same figure works whether the aorta annotation (label 4) is present.
    labels = sorted({int(v) for rows_df in tables for v in rows_df.label.unique()})
    shares = [piece_shares(rows_df) for rows_df in tables]
    counts = [len(list(d.glob("Patient_*"))) for d in dirs]
    n_patients = f"{counts[0]} per dataset" if len(counts) == 2 and counts[0] == counts[1] else " and ".join(map(str, counts))

    sns.set_theme(style="white", font_scale=1.1)
    shown = [
        [k for k in labels if k in table.index.get_level_values("label") and not (r > 0 and unchanged(shares[0], table, k))]
        for r, table in enumerate(shares)
    ]
    slots = [column_slots(shown[0], row) for row in shown]
    n_columns = max(len(row) for row in shown)
    height = 2.3 * len(dirs) + 1.7
    fig, axes = plt.subplots(len(dirs), n_columns, figsize=(3.5 * n_columns + 1.8, height), sharey=True, squeeze=False)
    for r, table_shares in enumerate(shares):
        by_column = {c: k for k, c in slots[r].items()}
        for c, ax in enumerate(axes[r]):
            if c not in by_column:
                ax.axis("off")
                continue
            k = by_column[c]
            cmap = LinearSegmentedColormap.from_list(f"label{k}", ["#FFFFFF", EARTH_LABEL_COLORS[k]])
            sns.heatmap(
                table_shares.loc[k].reindex([name for name, _, _ in RULES]), ax=ax, cmap=cmap, vmin=0, vmax=100,
                linewidths=2, linecolor="white", cbar=False,
            )
            ax.set_title(CLASSES[k], fontsize=16, fontweight="bold", loc="left", color=EARTH_LABEL_COLORS[k])
            ax.set(xlabel="connected components" if r == len(dirs) - 1 else "", ylabel="")
            ax.set_xticklabels([*map(str, range(1, MAX_PIECES)), f"{MAX_PIECES}+"])
            ax.tick_params(axis="y", rotation=0)
    title_block(
        fig,
        "Connected Components per Label",
        f"Patients pooled (n={n_patients}); each cell = % of axial slices (2D rows) or patients (3D rows) with "
        "that many connected components.",
    )
    fig.subplots_adjust(left=0.1 if args.names else 0.07, right=0.9, top=1 - 1.25 / height, bottom=0.8 / height, wspace=0.16, hspace=0.4)
    for r in range(1, len(dirs)):
        for k, column in slots[r].items():
            source = SPLIT_FROM.get(k)
            if k not in slots[0] and source in slots[0]:
                draw_split_arrow(fig, axes[0][slots[0][source]], axes[r][column], EARTH_LABEL_COLORS[k], f"{CLASSES[k]} was part of the {CLASSES[source]} label")
    cax = fig.add_axes([0.925, 0.25, 0.012, 0.45])
    fig.colorbar(
        ScalarMappable(Normalize(0, 100), LinearSegmentedColormap.from_list("share", ["#FFFFFF", "#444444"])), cax=cax,
        label="% of slices (2D) or patients (3D)",
    )
    if args.names:  # dataset name at the left of each row
        for name, row in zip(args.names, axes):
            box = row[0].get_position()
            fig.text(0.012, (box.y0 + box.y1) / 2, name, rotation=90, ha="left", va="center", fontsize=13, fontweight="bold", color="#444444")
    out = (args.out_dir or out_subdir(args.profile_dir, "connected_components")) / "connected_components.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
