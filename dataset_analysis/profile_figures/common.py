"""
Shared helpers for the SegTHOR dataset-profile figure scripts in this folder.

Every script here reads one or more tables written by tools/dataset_profile.py
and writes one PNG (or a pair of PNGs) into its own numbered subfolder under
figures/profile/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from plot_style import LABEL_COLORS  # noqa: E402, F401

# Re-exported from tools/plot_style.py so every figure in the repo -- these
# profile figures, explore_data.py, nnunet_planner_checks.py, render_figures.py --
# draws each organ label in the same color. Keys 1-4 cover esophagus, heart,
# trachea and aorta; a dataset missing a label simply never indexes that key.


def apply_ticks_style() -> None:
    """Seaborn "ticks" theme (white background, axis ticks, no grid) for dot-histogram overviews."""
    sns.set_theme(style="ticks", font_scale=1.15)


def title_block(fig, title: str, subtitle: str, title_size: float = 22, subtitle_size: float = 13.5) -> None:
    """Big bold left-aligned figure title with one gray subtitle line underneath.

    Leave room for it with fig.subplots_adjust(top=...) or tight_layout(rect=(..., 0.92)).

    Args:
        fig: Figure to decorate.
        title: Title text.
        subtitle: One-line subtitle text.
        title_size: Title font size; the subtitle line moves up and down with it.
        subtitle_size: Subtitle font size.
    """
    height = fig.get_size_inches()[1]
    fig.suptitle(title, fontsize=title_size, fontweight="bold", x=0.02, ha="left", y=1 - 0.15 / height, va="top")
    fig.text(0.02, 1 - 0.62 * title_size / 22 / height, subtitle, fontsize=subtitle_size, color="#444444", ha="left", va="top")


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser shared by the profile-figure scripts.

    Args:
        description: Script description shown in --help (usually __doc__).

    Returns:
        Parser with --profile-dir already added (default figures/profile).
    """
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-dir", type=Path, default=Path("figures/profile"))
    return ap


def out_subdir(profile_dir: Path, name: str) -> Path:
    """Create (if needed) and return a named output subfolder under profile_dir.

    Args:
        profile_dir: Root profile directory, e.g. figures/profile.
        name: Subfolder name, e.g. "scan_geometry".

    Returns:
        The subfolder path.
    """
    out_dir = profile_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def header(fig, title: str, subtitle: str, subtitle_y: float) -> None:
    """Big bold left-aligned figure title with a lighter subtitle line below it.

    Args:
        fig: Figure to decorate.
        title: Title text, drawn at 22pt bold.
        subtitle: One-line subtitle text, drawn at 13.5pt in gray.
        subtitle_y: Figure-fraction y position of the subtitle, since each
            figure's own layout leaves a different amount of headroom.
    """
    fig.suptitle(title, fontsize=22, fontweight="bold", x=0.02, ha="left", y=0.995)
    fig.text(0.02, subtitle_y, subtitle, fontsize=13.5, color="#444444", ha="left")
