#!/usr/bin/env python3
"""Generate presentation figures for the HU-windowing preprocessing step.

Fig 1: intensity histogram of a real SegTHOR volume showing the extreme HU
       outlier, and how min-max vs the mediastinal window spend the 0-255 budget.
Fig 2: the same axial slice under per-volume min-max vs the mediastinal window,
       full frame + a zoom on the mediastinum.

Data: data/gt/watershed_refined/train/Patient_02 (max ~26,613 HU, p99 ~367 HU).
"""
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PID = "Patient_02"
Z = 126
LEVEL, WIDTH = 40.0, 400.0            # mediastinal window
LO, HI = LEVEL - WIDTH / 2, LEVEL + WIDTH / 2   # [-160, 240]
OUT = Path(__file__).parent

d = Path("data/gt/watershed_refined/train") / PID
ct = np.asarray(nib.load(str(d / f"{PID}.nii.gz")).dataobj).astype(np.float32)
gt = np.asarray(nib.load(str(d / "GT.nii.gz")).dataobj)

vmin, vmax, p99 = float(ct.min()), float(ct.max()), float(np.percentile(ct, 99))


def minmax(vol):
    return (vol - vmin) / (vmax - vmin)


def window(vol, lo=LO, hi=HI):
    return np.clip((vol - lo) / (hi - lo), 0, 1)


# orient the axial slice for display (radiological-ish): rotate so head is up
def show_slice(a):
    return np.rot90(a)


ct_sl = ct[:, :, Z]
gt_sl = gt[:, :, Z]

# ---------------------------------------------------------------- Figure 1
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.6))

# Panel A: full HU range, log counts
counts, edges = np.histogram(ct.ravel(), bins=400, range=(vmin, vmax))
centers = (edges[:-1] + edges[1:]) / 2
axA.fill_between(centers, counts + 1, 0.9, step="mid", color="#9aa7b4", alpha=0.9, lw=0)
axA.set_yscale("log")
axA.set_xlim(vmin, vmax)
axA.set_ylim(0.9, counts.max() * 6)
# min-max span (the whole axis) vs mediastinal window
axA.axvspan(LO, HI, color="#2e9e6b", alpha=0.28,
            label=f"mediastinal window [{LO:.0f}, {HI:.0f}] HU")
axA.axvline(p99, color="#d8871f", ls="--", lw=1.5)
axA.annotate(f"99th pct\n= {p99:.0f} HU", (p99, counts.max()),
             xytext=(3500, counts.max() * 2.6),
             color="#a9660f", fontsize=9.5, ha="left",
             arrowprops=dict(arrowstyle="->", color="#a9660f"))
axA.annotate(f"max = {vmax:.0f} HU\n(single outlier)", (vmax, 2), xytext=(vmax - 9000, 40),
             color="#b0322b", fontsize=10, ha="right",
             arrowprops=dict(arrowstyle="->", color="#b0322b"))
axA.annotate("min–max stretches the 0–255 budget\nacross this whole empty span →",
             (12000, 5000), color="#444", fontsize=9.5)
axA.set_title(f"A · Full HU histogram — {PID}", fontsize=12, fontweight="bold")
axA.set_xlabel("Hounsfield Units (HU)")
axA.set_ylabel("voxel count (log)")
axA.legend(loc="upper right", fontsize=9, framealpha=0.95)

# Panel B: zoom into the soft-tissue range (air peak at -1000 is left off-frame
# on purpose, so the soft-tissue distribution the window preserves is visible).
lo_z, hi_z = -300, 500
c2, e2 = np.histogram(ct.ravel(), bins=200, range=(lo_z, hi_z))
cen2 = (e2[:-1] + e2[1:]) / 2
axB.fill_between(cen2, c2, 0, step="mid", color="#9aa7b4", alpha=0.9, lw=0)
axB.axvspan(LO, HI, color="#2e9e6b", alpha=0.28)
axB.axvline(LO, color="#2e9e6b", lw=1.6)
axB.axvline(HI, color="#2e9e6b", lw=1.6)
for hu, name, col in [(-90, "fat", "#c08a3e"),
                      (40, "soft tissue\n(eso / aorta / heart)", "#b0322b")]:
    axB.axvline(hu, color=col, ls=":", lw=1.5)
    axB.annotate(name, (hu, c2.max() * 0.96), rotation=90, va="top", ha="right",
                 fontsize=9, color=col)
axB.set_xlim(lo_z, hi_z)
axB.set_ylim(0, c2.max() * 1.12)
axB.set_title("B · Zoom on soft tissue (air peak off-frame)", fontsize=12, fontweight="bold")
axB.set_xlabel("Hounsfield Units (HU)")
axB.set_ylabel("voxel count")
axB.annotate("← the window spends the\nfull 0–255 range on exactly\nthis soft-tissue band",
             (HI, c2.max() * 0.62), xytext=(255, c2.max() * 0.62),
             fontsize=9, color="#1d6f4b", va="center")

fig.suptitle("Why fixed HU windowing beats per-volume min–max normalization",
             fontsize=13.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig1_hu_histogram.png", dpi=170, bbox_inches="tight")
print("wrote fig1_hu_histogram.png")

# ---------------------------------------------------------------- Figure 2
# mediastinum zoom box from GT bounding box (+ padding)
ys, xs = np.where(gt_sl > 0)
pad = 45
r0, r1 = max(xs.min() - pad, 0), min(xs.max() + pad, ct_sl.shape[0])
c0, c1 = max(ys.min() - pad, 0), min(ys.max() + pad, ct_sl.shape[1])

mm = show_slice(minmax(ct_sl))
wd = show_slice(window(ct_sl))
# transform the box into rot90 coords
H = ct_sl.shape[0]
box = dict(x=c0, y=H - r1, w=(c1 - c0), h=(r1 - r0))

fig2, ax = plt.subplots(2, 2, figsize=(9.6, 10),
                        gridspec_kw=dict(hspace=0.08, wspace=0.06))
for a in ax.ravel():
    a.axis("off")

ax[0, 0].imshow(mm, cmap="gray", vmin=0, vmax=1)
ax[0, 0].set_title("Per-volume min–max (baseline)", fontsize=12, fontweight="bold")
ax[0, 1].imshow(wd, cmap="gray", vmin=0, vmax=1)
ax[0, 1].set_title("Mediastinal HU window", fontsize=12, fontweight="bold", color="#1d6f4b")
for a in (ax[0, 0], ax[0, 1]):
    a.add_patch(Rectangle((box["x"], box["y"]), box["w"], box["h"],
                          ec="#e2b53a", fc="none", lw=1.8))

# zoomed rows
def crop(a):
    return a[box["y"]:box["y"] + box["h"], box["x"]:box["x"] + box["w"]]

ax[1, 0].imshow(crop(mm), cmap="gray", vmin=0, vmax=1)
ax[1, 0].set_title("mediastinum — washed out, low contrast", fontsize=10.5)
ax[1, 1].imshow(crop(wd), cmap="gray", vmin=0, vmax=1)
ax[1, 1].set_title("mediastinum — eso / aorta / heart separable", fontsize=10.5, color="#1d6f4b")

fig2.suptitle(f"Same slice, same voxels — {PID} (z={Z})\n"
              "min–max crushes soft tissue toward one grey level; the window restores it",
              fontsize=12.5, fontweight="bold", y=0.97)
fig2.savefig(OUT / "fig2_window_before_after.png", dpi=170, bbox_inches="tight")
print("wrote fig2_window_before_after.png")
