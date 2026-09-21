"""
Shared matplotlib style for this project's figures, matched to the look used
in the user's other data-analysis figures: bold declarative titles (state the
finding, not the axis), a light subtitle line, despined axes with horizontal
gridlines only, a below-plot legend with no frame (queued via legend_below(), drawn by
decorate() in figure coordinates), and a muted qualitative color palette.

Usage:
    from plot_style import PALETTE, apply_style, decorate

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    ...
    decorate(fig, "Heart Dominates the Foreground",
              subtitle="Voxel share of the 4 labeled organs (log scale)",
              footnote_text="Class 4 (esophagus) is unlabeled in this partial release.")
    fig.savefig(...)

`decorate()` wraps title/subtitle/footnote to the actual figure width and
reserves title/footnote/legend space in *inches* based on the resulting
line count (not figure-fraction, and not a fixed one-line assumption) so the
layout is correct regardless of figsize or text length. Call it LAST, after
all axes/legends are drawn -- it calls tight_layout/subplots_adjust itself.
"""

from __future__ import annotations

import textwrap

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# Pastel qualitative palette (dusty blue, peach, seafoam, sage, lavender, rose,
# honey, terracotta). Kept mid-tone rather than pale so white text on a
# PALETTE[0] fill (e.g. table headers) and thin lines on white stay legible --
# a true baby-pastel loses too much contrast for those uses.
PALETTE = [
    "#5B84A8",  # dusty blue
    "#E8A87C",  # peach
    "#7FC2BE",  # seafoam
    "#A0CC94",  # sage
    "#A896CC",  # lavender
    "#D48A82",  # rose
    "#E8C27E",  # honey
    "#D99B8C",  # terracotta
]

# Fixed color per segmentation label number, shared by every figure.
LABEL_COLORS = {1: PALETTE[0], 2: PALETTE[1], 3: PALETTE[2], 4: "#8C8C8C"}

# Earthy qualitative palette (brick, deep teal, ochre, olive) for figures that
# need a few categories which read as one family, e.g. scan field-of-view sizes.
EARTH = ["#B5533C", "#2E5E6E", "#C9A15B", "#6E7F5A"]

# Earthy alternative to LABEL_COLORS (teal esophagus, brick heart, ochre trachea,
# olive aorta), used by figures that share the scan field-of-view look.
EARTH_LABEL_COLORS = {1: EARTH[1], 2: EARTH[0], 3: EARTH[2], 4: "#5E9142"}

# Warm sand for the unlabeled background, so it never reads as one of the organs.
BACKGROUND_COLOR = "#CFC7B8"

GRID_COLOR = "#D9D9D9"
FOOTNOTE_COLOR = "#6B6B6B"

# Per-line heights in inches (font size + typical leading), and side margin
# reserved for wrapping estimates.
_TITLE_LINE_H = 0.30
_SUBTITLE_LINE_H = 0.22
_FOOTNOTE_LINE_H = 0.20
_TOP_PAD = 0.08
_GAP = 0.06
_BOTTOM_PAD = 0.05
_LEGEND_H = 0.58
_SIDE_MARGIN_IN = 0.7


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.frameon": False,
            "legend.fontsize": 11,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "savefig.dpi": 160,
            "savefig.facecolor": "white",
        }
    )


def tint(color, amount: float) -> tuple[float, float, float]:
    """Blend a color towards white.

    Args:
        color: Any matplotlib color.
        amount: 0 gives white, 1 gives the color unchanged.

    Returns:
        The blended RGB tuple.
    """
    r, g, b = mcolors.to_rgb(color)
    return tuple(1 - amount * (1 - c) for c in (r, g, b))


def style_axis_horizontal_bars(ax) -> None:
    """For horizontal bar charts: gridlines on x only, no y-gridlines."""
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)


def legend_below(ax, ncol: int = 4) -> None:
    """Queue a below-plot legend. It is NOT drawn on the axes -- an axes legend
    anchored outside the axes is counted by tight_layout AND again by decorate's
    own reservation, which double-books the space and lets the legend land on top
    of the x-label. Instead we stash the handles and let decorate() draw a
    figure-level legend at a known position above the footnote."""
    handles, labels = ax.get_legend_handles_labels()
    ax.figure._pending_legend = (handles, labels, ncol)


def _wrap(
    text: str, fontsize_pt: float, fig_width_in: float, char_width_factor: float = 0.60
) -> tuple[str, int]:
    """Wrap `text` to fit `fig_width_in`, given a rough average glyph width
    for this font size (in points). Returns (wrapped_text, n_lines)."""
    usable_in = max(fig_width_in - _SIDE_MARGIN_IN, 1.5)
    char_width_in = fontsize_pt * char_width_factor / 72.0
    max_chars = max(12, int(usable_in / char_width_in))
    wrapped = textwrap.fill(text, width=max_chars)
    return wrapped, wrapped.count("\n") + 1


def decorate(
    fig,
    title: str | None,
    subtitle: str | None = None,
    footnote_text: str | None = None,
    has_legend: bool = False,
) -> None:
    """Add a bold title (+ optional subtitle) above the axes and an optional
    italic footnote below, wrapped to the figure width, then lay out the
    figure so nothing overlaps. Call this LAST, after all axes/legends are
    drawn, instead of tight_layout().
    """
    w, h = fig.get_size_inches()
    has_legend = has_legend or getattr(fig, "_pending_legend", None) is not None

    title_wrapped, title_h = "", 0.0
    if title:
        title_wrapped, n_title_lines = _wrap(title, 16, w, char_width_factor=0.62)
        title_h = n_title_lines * _TITLE_LINE_H

    subtitle_wrapped = subtitle_h = None
    n_subtitle_lines = 0
    if subtitle:
        subtitle_wrapped, n_subtitle_lines = _wrap(
            subtitle, 11.5, w, char_width_factor=0.52
        )
        subtitle_h = n_subtitle_lines * _SUBTITLE_LINE_H

    footnote_wrapped = None
    n_footnote_lines = 0
    if footnote_text:
        footnote_wrapped, n_footnote_lines = _wrap(
            footnote_text, 9, w, char_width_factor=0.60
        )

    top_reserved = _TOP_PAD + title_h + (_GAP + subtitle_h if subtitle else 0.0) + 0.05
    bottom_reserved = (
        _BOTTOM_PAD
        + (_LEGEND_H if has_legend else 0.0)
        + (n_footnote_lines * _FOOTNOTE_LINE_H if footnote_text else 0.0)
    )

    # va='top' so the y coordinate is the TOP of the (possibly multi-line) block.
    title_y = 1.0 - _TOP_PAD / h
    if title:
        fig.suptitle(title_wrapped, fontsize=16, fontweight="bold", y=title_y, va="top")

    if subtitle_wrapped:
        subtitle_y = 1.0 - (_TOP_PAD + title_h + _GAP) / h
        fig.text(
            0.5,
            subtitle_y,
            subtitle_wrapped,
            fontsize=11.5,
            ha="center",
            va="top",
            color="#333333",
        )

    if footnote_wrapped:
        footnote_y = _BOTTOM_PAD / h
        fig.text(
            0.01,
            footnote_y,
            footnote_wrapped,
            fontsize=9,
            color=FOOTNOTE_COLOR,
            ha="left",
            va="bottom",
            style="italic",
        )

    # tight_layout() first (no rect) so axis labels/ticks/tick-labels get
    # correct spacing for THIS figure's content -- that spacing is not
    # knowable from figure height alone (a rotated tick label, a multi-line
    # x-label, etc. all change it). We then EXTEND that already-correct
    # margin by our reserved title/footnote height, via subplots_adjust
    # (which always applies, unlike tight_layout's rect which can silently
    # refuse and fall back to defaults when it doesn't fit).
    fig.tight_layout()
    tl_bottom, tl_top = fig.subplotpars.bottom, fig.subplotpars.top
    bottom_frac = min(0.45, tl_bottom + bottom_reserved / h)
    top_frac = max(bottom_frac + 0.05, tl_top - top_reserved / h)
    fig.subplots_adjust(top=top_frac, bottom=bottom_frac)

    # Draw the queued legend LAST, in figure coordinates, in the band we just
    # reserved between the footnote and the bottom of the axes. Figure coords
    # mean it cannot ride on top of the x-label the way an axes-anchored legend
    # does when the axes get short.
    pending = getattr(fig, "_pending_legend", None)
    if pending is not None:
        handles, labels, ncol = pending
        legend_y = (_BOTTOM_PAD + n_footnote_lines * _FOOTNOTE_LINE_H + _GAP) / h
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=ncol,
            frameon=False,
        )
        fig._pending_legend = None
