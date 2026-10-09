#!/usr/bin/env python3
"""Presentation figures for the resampling step + the preprocessing A/B results.

Fig 3: per-patient voxel-spacing spread across the SegTHOR training set (why the
       sampling grid is a non-anatomical confound).
Fig 4: schematic — resample+crop/pad preserves mm/pixel, resize cancels it.
Fig 5: A/B summary metrics: no_hu_window vs baseline(+HU window) vs +resampling.
Fig 6: A/B per-organ 3D Dice for the same three configs.

Spacing data read live from the GT volumes; A/B numbers taken from RESULTS.md.
"""
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import glob, os

OUT = Path(__file__).parent
TARGET = (1.95, 1.95, 2.5)   # data.target_spacing for refined_window_resampled

# ---- colours (consistent across all A/B figures) ----
C_NOWIN = "#b0603a"   # no HU window
C_WIN   = "#4a7fb5"   # baseline: HU window
C_RES   = "#2e9e6b"   # HU window + resampling

# --------------------------------------------------------- read real spacings
rows = []
for d in sorted(glob.glob("data/gt/watershed_refined/train/Patient_*")):
    pid = os.path.basename(d)
    z = nib.load(f"{d}/{pid}.nii.gz").header.get_zooms()[:3]
    rows.append((float(z[0]), float(z[2])))
inplane = np.array([r[0] for r in rows])
thick = np.array([r[1] for r in rows])

# =========================================================== Fig 3: spacing spread
fig, ax = plt.subplots(figsize=(8.2, 5.2))
rng = np.random.default_rng(0)
# jitter to separate overlapping identical points; size ~ how many patients share it
from collections import Counter
cnt = Counter(zip(np.round(inplane, 3), thick))
for (xi, yi), n in cnt.items():
    jx = rng.normal(0, 0.004, n)
    jy = rng.normal(0, 0.02, n)
    ax.scatter(xi + jx, yi + jy, s=90, alpha=0.75, color=C_WIN,
               edgecolor="white", linewidth=0.8, zorder=3)
ax.scatter([], [], s=90, color=C_WIN, edgecolor="white", label=f"{len(rows)} patients")

# target spacing marker
ax.scatter(TARGET[0], TARGET[2], marker="*", s=520, color=C_RES, edgecolor="black",
           linewidth=1.0, zorder=5, label=f"resample target ({TARGET[0]}, {TARGET[2]}) mm")

ax.axvspan(inplane.min(), inplane.max(), color="#d8871f", alpha=0.10, zorder=0)
ax.annotate("", xy=(inplane.max(), 2.02), xytext=(inplane.min(), 2.02),
            arrowprops=dict(arrowstyle="<->", color="#a9660f", lw=1.6))
ax.text((inplane.min() + inplane.max()) / 2, 1.99,
        f"in-plane spacing varies\n{inplane.min():.3f}–{inplane.max():.3f} mm  (1.53×)",
        ha="center", va="top", color="#a9660f", fontsize=10)
ax.text(1.30, 2.22, "slice thickness\n2.0 vs 2.5 mm", color="#555", fontsize=9.5, ha="left")

# The model never sees native resolution: the pipeline outputs a 256 grid from the
# 512 acquisition matrix, so the resize baseline already halves in-plane resolution
# (~0.98 -> ~1.95 mm). That is where the target x-value comes from -- it MATCHES the
# baseline's effective spacing, so the A/B changes grid consistency, not resolution.
ax.axvline(TARGET[0], color=C_RES, ls="--", lw=1.4, alpha=0.8, zorder=1)
ax.annotate("native ≈ 0.98 mm", xy=(0.977, 2.47), xytext=(0.90, 2.30),
            color="#2e7d5b", fontsize=8.8, ha="left", va="center",
            arrowprops=dict(arrowstyle="-|>", color="#2e7d5b", lw=1.2))
ax.annotate("the model runs on a 256 grid, so the resize\n"
            "baseline already halves in-plane resolution\n"
            "(512→256):  ≈0.98 → 1.95 mm effective.\n"
            "The target matches it, so the A/B isolates\n"
            "grid consistency, not resolution.",
            xy=(TARGET[0], 2.45), xytext=(1.44, 2.02),
            color="#1d6f4b", fontsize=8.8, ha="left", va="center",
            arrowprops=dict(arrowstyle="-|>", color="#2e9e6b", lw=1.5))

ax.set_xlabel("in-plane spacing (mm/pixel)")
ax.set_ylabel("slice thickness (mm)")
ax.set_title("Voxel spacing is set by the scanner protocol, not anatomy\n"
             "→ the same organ lands on a different pixel grid per patient",
             fontsize=12, fontweight="bold")
ax.set_xlim(0.80, 2.10)
ax.set_ylim(1.85, 2.70)
ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
ax.grid(alpha=0.25, zorder=0)
fig.tight_layout()
fig.savefig(OUT / "fig3_spacing_spread.png", dpi=170, bbox_inches="tight")
print("wrote fig3_spacing_spread.png")

# =========================================================== Fig 4: resize vs crop/pad schematic
# Key idea: the SAME physical organ starts at different PIXEL sizes because mm/px
# differs (wide FOV = coarser mm/px = organ smaller in pixels). resize keeps that
# per-patient scale, so the organ stays different sizes (confound remains).
# resample maps everyone to one mm/px, so the organ becomes the SAME pixel size.
fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.0))
for ax in axes:
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

def organ(ax, cx, cy, r, color="#c65d4b"):
    ax.add_patch(plt.Circle((cx, cy), r, color=color, alpha=0.9, zorder=4))

def body(ax, cx, y, w, h, grid, caption, mmpp, mmpp_col="#1d6f4b", caption_above=False):
    x = cx - w / 2
    ax.add_patch(Rectangle((x, y), w, h, fc="#eef1f4", ec="#333", lw=1.6, zorder=1))
    for i in range(1, grid):
        ax.plot([x + i * w / grid, x + i * w / grid], [y, y + h], color="#c9d2db", lw=0.6, zorder=2)
    for i in range(1, max(2, round(grid * h / w))):
        yy = y + i * w / grid
        if yy < y + h:
            ax.plot([x, x + w], [yy, yy], color="#c9d2db", lw=0.6, zorder=2)
    # spacing tag inside the top-left corner (no arrow crossing)
    ax.text(x + 0.12, y + h - 0.12, mmpp, ha="left", va="top", fontsize=8.6,
            color=mmpp_col, fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=mmpp_col, alpha=0.9))
    if caption_above:
        ax.text(cx, y + h + 0.18, caption, ha="center", va="bottom", fontsize=9.3)
    else:
        ax.text(cx, y - 0.35, caption, ha="center", va="top", fontsize=9.5)

def flow(ax, cx, col):
    ax.add_patch(FancyArrowPatch((cx, 5.7), (cx, 4.9), arrowstyle="-|>",
                                 mutation_scale=20, color=col, lw=2.0, zorder=6))

R_WIDE, R_NARROW, R_COMMON = 0.50, 0.80, 0.66   # organ pixel radii

# LEFT panel: RESIZE (bad)
ax = axes[0]
ax.set_title("resize to fixed pixel count  ✗", fontsize=12.5, fontweight="bold", color="#b0322b")
body(ax, 2.5, 6.0, 4.4, 3.0, 10, "wide FOV patient", "1.37 mm/px", caption_above=True); organ(ax, 2.5, 7.5, R_WIDE)
body(ax, 7.5, 6.0, 2.0, 3.0, 4, "narrow FOV patient", "0.90 mm/px", caption_above=True); organ(ax, 7.5, 7.5, R_NARROW)
flow(ax, 2.5, "#b0322b"); flow(ax, 7.5, "#b0322b")
# both stretched to 256x256, but scale kept per-patient -> organ still different sizes
body(ax, 2.5, 1.3, 3.0, 3.0, 8, "256×256", "1.83 mm/px", "#b0322b", caption_above=True); organ(ax, 2.5, 2.8, R_WIDE)
body(ax, 7.5, 1.3, 3.0, 3.0, 8, "256×256", "1.05 mm/px", "#b0322b", caption_above=True); organ(ax, 7.5, 2.8, R_NARROW)
ax.text(5.0, 0.35, "same organ → STILL different pixel size (mm/px still differs)\n"
        "the non-anatomical confound survives", ha="center", va="bottom", fontsize=9.3, color="#b0322b")

# RIGHT panel: RESAMPLE + CROP/PAD (good)
ax = axes[1]
ax.set_title("resample to fixed spacing → center crop/pad  ✓", fontsize=12.5,
             fontweight="bold", color="#1d6f4b")
body(ax, 2.5, 6.0, 4.4, 3.0, 10, "wide FOV patient", "1.37 mm/px", caption_above=True); organ(ax, 2.5, 7.5, R_WIDE)
body(ax, 7.5, 6.0, 2.0, 3.0, 4, "narrow FOV patient", "0.90 mm/px", caption_above=True); organ(ax, 7.5, 7.5, R_NARROW)
flow(ax, 2.5, "#1d6f4b"); flow(ax, 7.5, "#1d6f4b")
# resampled to one mm/px -> organ SAME pixel size, then crop/pad to 256x256 canvas
body(ax, 2.5, 1.3, 3.0, 3.0, 8, "resampled + crop/pad", "1.95 mm/px", caption_above=True); organ(ax, 2.5, 2.8, R_COMMON)
body(ax, 7.5, 1.3, 3.0, 3.0, 8, "resampled + crop/pad", "1.95 mm/px", caption_above=True); organ(ax, 7.5, 2.8, R_COMMON)
ax.text(5.0, 0.35, "same organ → IDENTICAL pixel size (mm/px = 1.95 for all)\n"
        "crop/pad only changes canvas size, never mm/px", ha="center", va="bottom", fontsize=9.3, color="#1d6f4b")

fig.suptitle("Why resampling is paired with center crop/pad, not resize",
             fontsize=13.5, fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig(OUT / "fig4_resize_vs_croppad.png", dpi=170, bbox_inches="tight")
print("wrote fig4_resize_vs_croppad.png")

# =========================================================== A/B numbers (RESULTS.md)
configs = ["no preprocessing", "HU window", "HU window\n+ resampling"]
colors = [C_NOWIN, C_WIN, C_RES]
# higher is better
dice3d = [0.456, 0.569, 0.683]
nsd    = [0.158, 0.249, 0.305]
# lower is better (mm)
hd95   = [44.39, 43.34, 30.20]
assd   = [12.28, 7.69, 5.30]

x = np.arange(3)

# =========================================================== Fig 5: summary A/B
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

def barpanel(ax, vals, title, ylabel, better, fmt="{:.3f}"):
    bars = ax.bar(x, vals, color=colors, edgecolor="white", width=0.66)
    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=9.5)
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.margins(y=0.18)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(0.98, 0.96, better, transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color="#555", style="italic")
    ax.grid(axis="y", alpha=0.25)

# overlap + tolerance on one panel (both [0,1], higher better)
ax = axes[0]
w = 0.36
b1 = ax.bar(x - w / 2, dice3d, w, color=colors, edgecolor="white")
b2 = ax.bar(x + w / 2, nsd, w, color=colors, edgecolor="white", alpha=0.55, hatch="//")
ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=9.5)
ax.set_title("Overlap & tolerance  (higher is better)", fontsize=11.5, fontweight="bold")
ax.set_ylabel("score  [0–1]")
ax.set_ylim(0, 0.82)
for bars, vals in [(b1, dice3d), (b2, nsd)]:
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8.8)
ax.text(x[0] - w / 2, dice3d[0] + 0.05, "3D Dice", ha="center", fontsize=8.5, color="#333")
ax.text(x[0] + w / 2, nsd[0] + 0.05, "NSD@1mm", ha="center", fontsize=8.5, color="#333")
ax.grid(axis="y", alpha=0.25)

# boundary distances (mm, lower better)
ax = axes[1]
w = 0.36
b1 = ax.bar(x - w / 2, hd95, w, color=colors, edgecolor="white")
b2 = ax.bar(x + w / 2, assd, w, color=colors, edgecolor="white", alpha=0.55, hatch="//")
ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=9.5)
ax.set_title("Boundary error, mm  (lower is better)", fontsize=11.5, fontweight="bold")
ax.set_ylabel("distance (mm)")
ax.set_ylim(0, 52)
for bars, vals in [(b1, hd95), (b2, assd)]:
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.8)
ax.text(x[0] - w / 2, hd95[0] + 3.0, "HD95", ha="center", fontsize=8.5, color="#333")
ax.text(x[0] + w / 2, assd[0] + 3.0, "ASSD", ha="center", fontsize=8.5, color="#333")
ax.grid(axis="y", alpha=0.25)

fig.suptitle("Preprocessing A/B — controlled runs (same GT, split, seed)\n"
             "windowing recovers overlap; resampling drives down boundary error",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig5_ab_summary.png", dpi=170, bbox_inches="tight")
print("wrote fig5_ab_summary.png")

# =========================================================== Fig 6: per-organ 3D Dice
organs = ["esophagus", "heart", "trachea", "aorta"]
per = {
    "no preprocessing":    [0.234, 0.700, 0.509, 0.379],
    "HU window":           [0.276, 0.874, 0.413, 0.712],
    "HU window + resamp.": [0.435, 0.890, 0.678, 0.730],
}
fig, ax = plt.subplots(figsize=(9.6, 5.4))
xo = np.arange(len(organs))
w = 0.26
for i, (name, vals) in enumerate(per.items()):
    bars = ax.bar(xo + (i - 1) * w, vals, w, color=colors[i], edgecolor="white", label=name)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.2)
ax.set_xticks(xo); ax.set_xticklabels(organs, fontsize=11)
ax.set_ylabel("3D Dice")
ax.set_ylim(0, 1.0)
ax.set_title("Per-organ 3D Dice — where each step helps\n"
             "windowing lifts the soft-tissue organs; resampling recovers the trachea",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9.5, loc="upper left")
ax.grid(axis="y", alpha=0.25)
# annotate the trachea trade-off
ax.annotate("windowing alone hurts trachea\n(lumen → 0); resampling fixes it",
            (2, 0.678), xytext=(1.4, 0.93), fontsize=8.6, color="#555",
            arrowprops=dict(arrowstyle="->", color="#888"))
fig.tight_layout()
fig.savefig(OUT / "fig6_ab_per_organ_dice.png", dpi=170, bbox_inches="tight")
print("wrote fig6_ab_per_organ_dice.png")
