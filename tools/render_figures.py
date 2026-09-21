#!/usr/bin/env python3
"""Render 3D-style figures (orthogonal planes + surface mesh) from a stitched prediction.

Produces, for one patient:
  <out>/<patient>_planes.png   axial / coronal / sagittal, CT | ground truth | prediction
  <out>/<patient>_surface.png  3D marching-cubes surfaces, ground truth vs prediction

Needs a stitched prediction volume (see stitch.py). Classes present in the data are
detected automatically, so it works whether the ground truth has 3 organs or 4.

Example (from the repo root):

    python tools/render_figures.py \
        --patient Patient_01 \
        --gt_dir data/segthor_part1/train \
        --pred_dir volumes/segthor/ce \
        --out figures
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from scipy.ndimage import zoom as ndzoom  # noqa: E402
from skimage.measure import marching_cubes  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import LABEL_COLORS  # noqa: E402

# Color per label index, from tools/plot_style.py so every figure in the repo
# colors organs the same way. Index 0 is background and is never drawn.
PALETTE = ["#00000000"] + [LABEL_COLORS[i] for i in range(1, 5)]
DEFAULT_NAMES = ["background", "class 1", "class 2", "class 3", "class 4"]


def load(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    obj = nib.load(str(path))
    return obj.get_fdata(dtype=np.float32), obj.header.get_zooms()[:3]


def render_planes(ct, gt, pred, zooms, names, labels, out: Path, title: str) -> None:
    # Show the slice carrying the most labeled voxels, per plane.
    fg = gt > 0
    za, yc, xs = (int(np.argmax(fg.sum(axis=ax))) for ax in [(0, 1), (0, 2), (1, 2)])
    zx, zy, zz = zooms
    window = np.clip((ct + 200) / 600, 0, 1)  # soft-tissue window

    planes = [
        ("Axial", lambda v: v[:, :, za].T, zx / zy),
        ("Coronal", lambda v: v[:, yc, :].T[::-1], zz / zx),
        ("Sagittal", lambda v: v[xs, :, :].T[::-1], zz / zy),
    ]
    cmap = ListedColormap(PALETTE)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for row, (plane, take, aspect) in enumerate(planes):
        base = take(window)
        for col, (label, vol) in enumerate(
            [("CT only", None), ("Ground truth", gt), ("Prediction", pred)]
        ):
            ax = axes[row, col]
            ax.imshow(base, cmap="gray", aspect=aspect)
            if vol is not None:
                sl = take(vol)
                ax.imshow(
                    np.ma.masked_where(sl == 0, sl),
                    cmap=cmap, vmin=0, vmax=len(PALETTE) - 1,
                    alpha=0.55, aspect=aspect, interpolation="nearest",
                )
            ax.set_title(f"{plane} — {label}", fontsize=12)
            ax.axis("off")
    fig.legend(
        handles=[Patch(facecolor=PALETTE[i], label=names[i]) for i in labels],
        loc="lower center", ncol=len(labels), fontsize=13, frameon=False,
    )
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=100, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def render_surface(gt, pred, zooms, names, labels, out: Path, title: str,
                   scale=(0.5, 0.5, 1.0)) -> None:
    # Crop to the union bounding box so the organs fill the frame.
    used = (gt > 0) | (pred > 0)
    if not used.any():
        print("nothing labeled; skipping surface render")
        return
    box = tuple(
        slice(max(0, idx[0] - 6), min(dim, idx[-1] + 6))
        for idx, dim in zip(
            [np.where(used.any(axis=tuple(j for j in range(3) if j != i)))[0]
             for i in range(3)],
            used.shape,
        )
    )
    gt, pred = gt[box], pred[box]
    fx, fy, fz = scale
    spacing = (zooms[0] / fx, zooms[1] / fy, zooms[2] / fz)

    fig = plt.figure(figsize=(15, 9))
    for col, (label, vol) in enumerate([("Ground truth", gt), ("Prediction", pred)]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        for lab in labels:
            mask = ndzoom((vol == lab).astype(np.float32), scale, order=1)
            if mask.max() < 0.5:
                continue
            verts, faces, _, _ = marching_cubes(mask, level=0.5, spacing=spacing)
            mesh = Poly3DCollection(verts[faces], alpha=0.5 if lab == 2 else 0.95)
            mesh.set_facecolor(PALETTE[lab])
            mesh.set_edgecolor("none")
            ax.add_collection3d(mesh)
        extent = [d * s for d, s in zip(ndzoom(vol.astype(np.float32), scale, order=0).shape, spacing)]
        ax.set_xlim(0, extent[0])
        ax.set_ylim(0, extent[1])
        ax.set_zlim(0, extent[2])
        ax.set_box_aspect(tuple(extent))
        ax.view_init(elev=8, azim=-70)
        ax.set_axis_off()
        ax.set_title(label, fontsize=15, pad=0)
    fig.legend(
        handles=[Patch(facecolor=PALETTE[i], label=names[i]) for i in labels],
        loc="lower center", ncol=len(labels), fontsize=13, frameon=False,
    )
    fig.suptitle(title, fontsize=15)
    fig.subplots_adjust(left=0, right=1, top=0.93, bottom=0.06, wspace=0)
    fig.savefig(out, dpi=105, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patient", default="Patient_01")
    p.add_argument("--gt_dir", type=Path, default=Path("data/segthor_part1/train"))
    p.add_argument("--pred_dir", type=Path, default=Path("volumes/segthor/ce"))
    p.add_argument("--out", type=Path, default=Path("figures"))
    p.add_argument("--class_names", nargs="*", default=None,
                   help=f"names per label index, default {DEFAULT_NAMES}")
    p.add_argument("--skip_surface", action="store_true",
                   help="planes only (surface meshing is the slow part)")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pid = args.patient
    ct, _ = load(args.gt_dir / pid / f"{pid}.nii.gz")
    gt_f, zooms = load(args.gt_dir / pid / "GT.nii.gz")
    pred_f, _ = load(args.pred_dir / f"{pid}.nii.gz")
    gt, pred = gt_f.astype(np.uint8), pred_f.astype(np.uint8)

    names = args.class_names or DEFAULT_NAMES
    names = list(names) + DEFAULT_NAMES[len(names):]
    # Only draw labels that actually occur -- avoids inventing an absent organ.
    labels = sorted(set(np.unique(gt)) | set(np.unique(pred)))
    labels = [int(v) for v in labels if v != 0 and v < len(PALETTE)]
    print(f"{pid}: labels present {labels}  voxel {zooms[0]:.2f}x{zooms[1]:.2f}x{zooms[2]:.2f} mm")

    stem = f"{pid} — voxel {zooms[0]:.2f}x{zooms[1]:.2f}x{zooms[2]:.2f} mm"
    render_planes(ct, gt, pred, zooms, names, labels,
                  args.out / f"{pid}_planes.png", f"{stem} — orthogonal planes")
    if not args.skip_surface:
        render_surface(gt, pred, zooms, names, labels,
                       args.out / f"{pid}_surface.png", f"{stem} — 3D surfaces")


if __name__ == "__main__":
    main()
