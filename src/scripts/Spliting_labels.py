from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.measure import perimeter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

MERGED_LABEL = 1
MIN_CIRCULARITY = 0.5 

BEFORE_NAMES = ["background", "esophagus+trachea", "heart", "aorta"]
BEFORE_COLORS = ["#c3c2b7", "#2a78d6", "#eb6834", "#eda100"]
AFTER_NAMES = ["background", "esophagus", "heart", "trachea", "aorta"]
AFTER_COLORS = ["#c3c2b7", "#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def cmap_norm(colors: list[str]):
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1), cmap.N)
    return cmap, norm


def components_per_slice(mask: np.ndarray) -> np.ndarray:
    """Number of connected components in each non-empty axial slice of a 3D mask."""
    counts = [ndi.label(mask[:, :, z])[1] for z in range(mask.shape[2]) if mask[:, :, z].any()]
    return np.array(counts)


def report(name: str, mask: np.ndarray) -> None:
    counts = components_per_slice(mask)
    if len(counts) == 0:
        print(f"  {name:<10} not present")
        return
    print(f"  {name:<10} {len(counts):>4} slices   max {counts.max()} blobs/slice   "
          f"{(counts >= 2).mean():.0%} of slices have >=2 blobs")


def ap_axis_and_sign(affine: np.ndarray) -> tuple[int, int]:
    """Which of the two in-plane array axes is anterior/posterior, and which
    direction along it is anterior."""
    codes = nib.aff2axcodes(affine)
    for axis, code in enumerate(codes[:2]):
        if code in ("A", "P"):
            return axis, (1 if code == "A" else -1)
    raise ValueError(f"expected an anterior/posterior in-plane axis, got orientation {codes}")


def nearest_mask_point(mask: np.ndarray, point: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(mask)
    d2 = (ys - point[0]) ** 2 + (xs - point[1]) ** 2
    i = int(np.argmin(d2))
    return ys[i], xs[i]


def circularity(mask: np.ndarray) -> float:
    """4*pi*area/perimeter^2: 1.0 for a disk, much lower for a thin sliver."""
    area = mask.sum()
    if area < 4:
        return 0.0
    perim = perimeter(mask)
    return 4 * np.pi * area / perim**2 if perim > 0 else 0.0


def split_slice(
    mask: np.ndarray, seed_a: np.ndarray, seed_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Grow two regions from seed_a and seed_b until they cover `mask` or meet.
    If the result doesn't look like two separate round cross-sections anymore,
    leave this slice unresolved (report low confidence, assign nothing)
    instead of guessing -- and don't move either seed, so an ambiguous slice
    can't drag tracking off course for every slice after it."""
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[nearest_mask_point(mask, seed_a)] = 1
    markers[nearest_mask_point(mask, seed_b)] = 2
    labels = watershed(np.zeros(mask.shape), markers=markers, mask=mask)

    # a disconnected fragment (e.g. the trachea splitting at the carina) can't be
    # reached by either seed's flood through mask connectivity -- give it to
    # whichever region is nearest instead of leaving it unlabeled
    holes = mask & (labels == 0)
    if holes.any():
        _, (iy, ix) = ndi.distance_transform_edt(labels == 0, return_indices=True)
        labels = labels.copy()
        labels[holes] = labels[iy[holes], ix[holes]]

    a, b = labels == 1, labels == 2
    if min(circularity(a), circularity(b)) < MIN_CIRCULARITY:
        # can't confidently tell them apart here -- leave it unresolved rather than guess
        return np.zeros_like(mask), np.zeros_like(mask), False
    return a, b, True


def find_bootstrap_slice(merged: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """The cleanest slice to start tracking from: exactly 2 components, as far
    apart as possible (least likely to be mismatched)."""
    best = None
    for z in np.where(merged.any(axis=(0, 1)))[0]:
        lbl, n = ndi.label(merged[:, :, z])
        if n != 2:
            continue
        c1 = np.array(ndi.center_of_mass(lbl == 1))
        c2 = np.array(ndi.center_of_mass(lbl == 2))
        sep = np.linalg.norm(c1 - c2)
        if best is None or sep > best[0]:
            best = (sep, z, c1, c2)
    if best is None:
        raise ValueError("no slice with a clean 2-component split found to bootstrap from")
    _, z, c1, c2 = best
    return z, c1, c2


def split_trachea_from_esophagus(merged: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ap_axis, ap_sign = ap_axis_and_sign(affine)
    z0, c1, c2 = find_bootstrap_slice(merged)
    trachea_seed, esoph_seed = (c1, c2) if c1[ap_axis] * ap_sign > c2[ap_axis] * ap_sign else (c2, c1)

    trachea = np.zeros_like(merged, dtype=bool)
    esophagus = np.zeros_like(merged, dtype=bool)
    zs = np.where(merged.any(axis=(0, 1)))[0]

    for z_range in (range(z0, zs.max() + 1), range(z0 - 1, zs.min() - 1, -1)):
        seed_t, seed_e = trachea_seed, esoph_seed
        for z in z_range:
            mask = merged[:, :, z]
            if not mask.any():
                continue
            t, e, confident = split_slice(mask, seed_t, seed_e)
            trachea[:, :, z], esophagus[:, :, z] = t, e
            if confident:
                if t.any():
                    seed_t = np.array(ndi.center_of_mass(t))
                if e.any():
                    seed_e = np.array(ndi.center_of_mass(e))

    return esophagus, trachea


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("segthor_part1/data/segthor_part1/train"))
    ap.add_argument("--patient", default="Patient_01")
    args = ap.parse_args()

    patient_dir = args.data_dir / args.patient
    gt_img = nib.load(patient_dir / "GT.nii.gz")
    lab = np.asanyarray(gt_img.dataobj).astype(np.uint8)

    print(f"{args.patient}  shape={lab.shape}  orientation={nib.aff2axcodes(gt_img.affine)}")

    merged = lab == MERGED_LABEL
    print("\nBEFORE (label 1 = esophagus+trachea merged):")
    report("label 1", merged)

    esophagus, trachea = split_trachea_from_esophagus(merged, gt_img.affine)
    ambiguous = merged & ~esophagus & ~trachea  # couldn't confidently split -- left as the original merged label
    assert ((esophagus | trachea | ambiguous) == merged).all(), "esophagus + trachea + ambiguous should cover merged"
    assert not (esophagus & trachea).any(), "esophagus/trachea should not overlap"

    out = lab.copy()
    out[lab == 3] = 4  # move the existing aorta out of the way first, so it doesn't collide with the new trachea
    out[merged] = 0
    out[esophagus] = 1
    out[trachea] = 3
    out[ambiguous] = 1  # unresolved: keep the original merged label rather than guess

    print("\nAFTER (split):")
    report("esophagus", out == 1)
    report("trachea", out == 3)
    report("aorta", out == 4)
    print(f"  {'unresolved':<10} {ambiguous.sum():>6} px left as the merged label (could not be split confidently)")

    out_path = patient_dir / "GT_4label.nii.gz"
    nib.save(nib.Nifti1Image(out, gt_img.affine, gt_img.header), out_path)
    print(f"\nwrote {out_path}")

    # a mid-scan slice through the merged region, before vs after
    z = int(np.median(np.where(merged.any(axis=(0, 1)))[0]))
    before_cmap, before_norm = cmap_norm(BEFORE_COLORS)
    after_cmap, after_norm = cmap_norm(AFTER_COLORS)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(lab[:, :, z], cmap=before_cmap, norm=before_norm)
    axes[0].set_title(f"before - slice {z}")
    axes[1].imshow(out[:, :, z], cmap=after_cmap, norm=after_norm)
    axes[1].set_title(f"after - slice {z}")
    for ax, names, colors in [(axes[0], BEFORE_NAMES, BEFORE_COLORS), (axes[1], AFTER_NAMES, AFTER_COLORS)]:
        ax.axis("off")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors[1:]]
        ax.legend(handles, names[1:], loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2, frameon=False, fontsize=8)

    fig_path = patient_dir / "GT_4label_preview.png"
    fig.savefig(fig_path, dpi=110, bbox_inches="tight")
    print(f"wrote {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
