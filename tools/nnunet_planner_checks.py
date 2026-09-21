#!/usr/bin/env python3
"""
nnU-Net planner/fingerprint checks NOT already covered by explore_data.py,
ported as closely as possible to nnU-Net's own source so the numbers here are
literally the same computation nnU-Net would run, not a re-derivation, plus
per-patient distribution figures of the image grid, intensities and organ
geometry.

Reference source (cloned to a scratch dir while writing this,
github.com/MIC-DKFZ/nnUNet, nnunetv2 branch):

  1. Dataset integrity check
     nnunetv2/experiment_planning/verify_dataset_integrity.py
     (check_cases: shape/spacing/affine match; verify_labels: unexpected values)

  2. crop_to_nonzero
     nnunetv2/preprocessing/cropping/cropping.py
     (create_nonzero_mask: per-channel !=0, OR'd, binary_fill_holes;
      crop_to_nonzero: bbox of that mask)
     consumed in nnunetv2/experiment_planning/dataset_fingerprint/fingerprint_extractor.py
     (relative_size_after_cropping = prod(shape_after) / prod(shape_before))
     and in nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py
     (median_relative_size_after_cropping < 0.75 -> mask-restricted normalization)

  3. Anisotropy / target spacing
     nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py
     (determine_fullres_target_spacing)
     nnunetv2/configuration.py (ANISO_THRESHOLD = 3)

  4. Connected-component-per-class rule
     Described in the 2018 MICCAI-BraTS-workshop nnU-Net paper, Sec. 2.5
     "Postprocessing": if a class lies within a single connected component in
     ALL training cases, that is treated as a property of the dataset and all
     but the largest component are removed at inference. (Not literally present
     in nnunetv2, which instead does a CV-Dice-driven search in
     postprocessing/remove_connected_components.py -- reproduced here from the
     paper's stated rule, using the same scipy.ndimage.label primitive.)

Usage:
    python tools/nnunet_planner_checks.py \
        --data-dir data/segthor_part1/train --out-dir figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    generate_binary_structure,
    label,
    map_coordinates,
)
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import PALETTE, apply_style, decorate, legend_below

# Whether label 4 (aorta) has voxels depends on --data-dir: the original
# course release omits it, a full 4-class release doesn't. class 1 is the
# esophagus.
CLASS_NAMES = {0: "background", 1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}
CLASSES = [1, 2, 3, 4]
# Set from the actual data in main(), before any figure is drawn: figures
# skip whichever classes have no voxels in this run rather than assuming.
ORGANS_WITH_VOXELS = [1, 2, 3]
EXPECTED_LABELS = [0, 1, 2, 3, 4]  # what a SegTHOR dataset.json would declare
ANISO_THRESHOLD = 3  # nnunetv2/configuration.py: ANISO_THRESHOLD = 3
MASK_NORM_RELATIVE_SIZE_THRESHOLD = 3 / 4.0  # default_experiment_planner.py:205
# 4095 - 1024 = 3071: the max representable value once a 12-bit CT detector
# reading (0-4095) has the standard -1024 HU offset applied.
SATURATION_CEILING_HU = 3071

# Canonical grid for the cross-patient occupancy envelope: isotropic spacing,
# extent in mm relative to each patient's lowest trachea slice (present in all
# 20 patients -- no rotation/scale correction, just a per-patient translation).
CANONICAL_SPACING_MM = 3.0
CANONICAL_EXTENT_MM = {"x": (-150.0, 150.0), "y": (-170.0, 110.0), "z": (-210.0, 150.0)}

# Air and lung sit below this; soft tissue, bone and the scanner table above.
TISSUE_HU_THRESHOLD = -500
# Signed distance to an organ surface, 1 mm bins; negative is inside the mask.
# No 0 bin: voxel centers are never closer than half a voxel to the surface.
PROFILE_DISTANCES_MM = np.array([d for d in range(-10, 11) if d != 0])
HU_BIN_EDGES = np.arange(-1100, 610, 10)

STRUCT_6 = generate_binary_structure(3, 1)  # face-adjacency only
STRUCT_18 = generate_binary_structure(3, 2)  # face + edge
STRUCT_26 = generate_binary_structure(3, 3)  # face + edge + corner
CONNECTIVITY = {"n6": STRUCT_6, "n18": STRUCT_18, "n26": STRUCT_26}
IN_PLANE_STRUCT = np.zeros((3, 3, 3), dtype=bool)
IN_PLANE_STRUCT[:, :, 1] = generate_binary_structure(2, 1)


# ---------------------------------------------------------------------------
# 1. verify_dataset_integrity.py -- check_cases() + verify_labels()
# ---------------------------------------------------------------------------
def check_case(img: nib.Nifti1Image, seg: nib.Nifti1Image) -> dict:
    """Port of verify_dataset_integrity.check_cases for one image/label pair.

    Args:
        img: CT volume.
        seg: Label volume.

    Returns:
        Shapes, spacings and affine parts of both volumes, and whether each matches.
    """
    shape_image = img.shape
    shape_seg = seg.shape
    shape_match = shape_image == shape_seg

    spacing_images = np.array(img.header.get_zooms()[:3])
    spacing_seg = np.array(seg.header.get_zooms()[:3])
    spacing_match = bool(np.allclose(spacing_images, spacing_seg))

    affine_match = bool(np.allclose(img.affine, seg.affine))

    return {
        "shape_image": shape_image,
        "shape_seg": shape_seg,
        "shape_match": shape_match,
        "spacing_image": spacing_images.tolist(),
        "spacing_seg": spacing_seg.tolist(),
        "spacing_match": spacing_match,
        "affine_match": affine_match,
        "affine_diag_image": np.diag(img.affine)[:3].tolist(),
        "affine_diag_seg": np.diag(seg.affine)[:3].tolist(),
        "affine_translation_image": img.affine[:3, 3].tolist(),
        "affine_translation_seg": seg.affine[:3, 3].tolist(),
    }


def verify_labels(seg_data: np.ndarray, expected_labels: list[int]) -> dict:
    """Port of verify_dataset_integrity.verify_labels, plus the missing labels.

    Args:
        seg_data: Label array.
        expected_labels: Label values the dataset declares.

    Returns:
        Found, unexpected and missing label values.
    """
    found_labels = sorted(int(x) for x in np.unique(seg_data))
    unexpected = [v for v in found_labels if v not in expected_labels]
    missing = [v for v in expected_labels if v not in found_labels]
    return {"found_labels": found_labels, "unexpected_labels": unexpected, "missing_labels": missing}


# ---------------------------------------------------------------------------
# 2. cropping.py -- create_nonzero_mask() + crop_to_nonzero()
# ---------------------------------------------------------------------------
def create_nonzero_mask(data: np.ndarray) -> np.ndarray:
    """Verbatim port of cropping.create_nonzero_mask.

    Args:
        data: (C, X, Y, Z) array.

    Returns:
        Hole-filled mask of voxels that are nonzero in any channel.
    """
    assert data.ndim == 4
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    return binary_fill_holes(nonzero_mask)


def crop_to_nonzero(data: np.ndarray) -> dict:
    """Bounding-box crop of the nonzero mask, as fingerprint_extractor records it.

    Args:
        data: (C, X, Y, Z) array.

    Returns:
        Shape before and after cropping and the kept volume fraction.
    """
    shape_before_crop = data.shape[1:]
    mask = create_nonzero_mask(data)
    if not mask.any():
        return {
            "shape_before_crop": shape_before_crop,
            "shape_after_crop": (0, 0, 0),
            "relative_size_after_cropping": 0.0,
        }
    idx = np.nonzero(mask)
    mins = tuple(int(a.min()) for a in idx)
    maxs = tuple(int(a.max()) for a in idx)
    shape_after_crop = tuple(hi - lo + 1 for lo, hi in zip(mins, maxs))
    relative_size = float(np.prod(shape_after_crop) / np.prod(shape_before_crop))
    return {
        "shape_before_crop": shape_before_crop,
        "shape_after_crop": shape_after_crop,
        "relative_size_after_cropping": relative_size,
    }


# ---------------------------------------------------------------------------
# 3. default_experiment_planner.py -- determine_fullres_target_spacing()
# ---------------------------------------------------------------------------
def determine_fullres_target_spacing(
    spacings: np.ndarray, sizes: np.ndarray, anisotropy_threshold: int = ANISO_THRESHOLD
) -> dict:
    """Verbatim port of default_experiment_planner.determine_fullres_target_spacing.

    Args:
        spacings: (N, 3) voxel spacing per case.
        sizes: (N, 3) array shape per case.
        anisotropy_threshold: nnU-Net's ANISO_THRESHOLD.

    Returns:
        Median spacing/shape, anisotropy flags and the final target spacing.
    """
    target = np.percentile(spacings, 50, axis=0)
    target_size = np.percentile(sizes, 50, axis=0)

    worst_spacing_axis = int(np.argmax(target))
    other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
    other_spacings = [target[i] for i in other_axes]
    other_sizes = [target_size[i] for i in other_axes]

    has_aniso_spacing = bool(target[worst_spacing_axis] > (anisotropy_threshold * max(other_spacings)))
    has_aniso_voxels = bool(target_size[worst_spacing_axis] * anisotropy_threshold < min(other_sizes))

    overridden = False
    final_target = target.copy()
    if has_aniso_spacing and has_aniso_voxels:
        spacings_of_that_axis = spacings[:, worst_spacing_axis]
        target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
        if target_spacing_of_that_axis < max(other_spacings):
            target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
        final_target[worst_spacing_axis] = target_spacing_of_that_axis
        overridden = True

    return {
        "median_spacing": target.tolist(),
        "median_shape": target_size.tolist(),
        "worst_spacing_axis": worst_spacing_axis,
        "has_aniso_spacing": has_aniso_spacing,
        "has_aniso_voxels": has_aniso_voxels,
        "cascade_override_applied": overridden,
        "final_target_spacing": final_target.tolist(),
    }


# ---------------------------------------------------------------------------
# 4. Connected-component-per-class rule (paper Sec. 2.5) and organ geometry
# ---------------------------------------------------------------------------
def _bbox_slices(mask: np.ndarray, margin) -> tuple[slice, ...]:
    """Bounding box of a non-empty mask, grown by `margin` voxels per axis."""
    box = []
    for axis, pad in enumerate(np.broadcast_to(margin, (mask.ndim,))):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        hits = np.flatnonzero(mask.any(axis=other))
        box.append(slice(max(int(hits[0]) - int(pad), 0), int(hits[-1]) + int(pad) + 1))
    return tuple(box)


def connected_components_per_class(seg: np.ndarray, classes: list[int]) -> dict:
    """Component count per class under 6-, 18- and 26-connectivity.

    Args:
        seg: Label array.
        classes: Class ids to count.

    Returns:
        {class: {"n6": count, "n18": count, "n26": count}}, zeros for absent classes.
    """
    out = {}
    for c in classes:
        mask = seg == c
        if not mask.any():
            out[c] = dict.fromkeys(CONNECTIVITY, 0)
            continue
        out[c] = {key: int(label(mask, structure=struct)[1]) for key, struct in CONNECTIVITY.items()}
    return out


def organ_pair_adjacency(seg: np.ndarray, classes: list[int]) -> dict:
    """Contact size between every ordered pair of classes.

    Args:
        seg: Label array.
        classes: Class ids to pair up.

    Returns:
        {"a_b": number of class-b voxels inside class a dilated by one voxel (26-conn)}.
    """
    masks = {c: seg == c for c in classes}
    dilated = {c: binary_dilation(m, structure=STRUCT_26) if m.any() else m for c, m in masks.items()}
    out = {}
    for a in classes:
        for b in classes:
            if a != b:
                out[f"{a}_{b}"] = int((dilated[a] & masks[b]).sum())
    return out


def organ_centroid_and_bbox(seg: np.ndarray, classes: list[int], zooms: np.ndarray) -> dict:
    """Per-class centroid and bounding box in mm from the array origin (index * spacing).

    Args:
        seg: Label array.
        classes: Class ids.
        zooms: Voxel spacing (mm).

    Returns:
        {class: {"centroid_mm", "bbox_min_mm", "bbox_max_mm"}}, None for absent classes.
    """
    out = {}
    for c in classes:
        idx = np.argwhere(seg == c)
        if idx.size == 0:
            out[c] = None
            continue
        out[c] = {
            "centroid_mm": (idx.mean(axis=0) * zooms).tolist(),
            "bbox_min_mm": (idx.min(axis=0) * zooms).tolist(),
            "bbox_max_mm": (idx.max(axis=0) * zooms).tolist(),
        }
    return out


def trachea_inferior_end_mm(seg: np.ndarray, zooms: np.ndarray) -> list[float] | None:
    """Centroid of the lowest axial slice containing trachea, in mm from the array origin.

    The volumes are LPS, so the lowest z index is the most inferior slice.

    Args:
        seg: Label array.
        zooms: Voxel spacing (mm).

    Returns:
        [x, y, z] in mm, or None if the trachea is absent.
    """
    idx = np.argwhere(seg == 3)
    if idx.size == 0:
        return None
    lowest = idx[idx[:, 2] == idx[:, 2].min()]
    return (lowest.mean(axis=0) * zooms).tolist()


def check_orientation(img: nib.Nifti1Image, seg: nib.Nifti1Image) -> dict:
    """Axis codes (e.g. LPS) of image and label, and whether they agree.

    Args:
        img: CT volume.
        seg: Label volume.

    Returns:
        Axis-code strings for both volumes and whether they match.
    """
    img_codes = "".join(nib.aff2axcodes(img.affine))
    seg_codes = "".join(nib.aff2axcodes(seg.affine))
    return {"image_axcodes": img_codes, "seg_axcodes": seg_codes, "orientation_match": img_codes == seg_codes}


def contrast_spike_stats(ct: np.ndarray, zooms: np.ndarray) -> dict:
    """Location and value of the brightest voxel, and counts at/above the 12-bit ceiling.

    Args:
        ct: CT array (HU).
        zooms: Voxel spacing (mm).

    Returns:
        Max HU, its location in mm, and voxel counts at and above 3071 HU.
    """
    idx = np.unravel_index(np.argmax(ct), ct.shape)
    n_exceeds_ceiling = int((ct > SATURATION_CEILING_HU).sum())
    return {
        "max_hu": float(ct[idx]),
        "max_hu_location_mm": [float(i * s) for i, s in zip(idx, zooms)],
        "n_voxels_at_saturation_ceiling": int((ct >= SATURATION_CEILING_HU - 1).sum()),
        "n_voxels_exceeding_ceiling": n_exceeds_ceiling,
        "exceeds_ceiling": n_exceeds_ceiling > 0,
    }


def pairwise_min_surface_distance(seg: np.ndarray, classes: list[int], zooms: np.ndarray) -> dict:
    """Minimum distance (mm) between every unordered pair of present classes.

    Args:
        seg: Label array.
        classes: Class ids.
        zooms: Voxel spacing (mm).

    Returns:
        {"a_b": distance in mm} for a < b; 0.0 means the masks touch.
    """
    masks = {c: seg == c for c in classes}
    present = [c for c in classes if masks[c].any()]
    out = {}
    for a in present:
        dist_from_a = distance_transform_edt(~masks[a], sampling=zooms)
        for b in present:
            if a < b:
                out[f"{a}_{b}"] = float(dist_from_a[masks[b]].min())
    return out


def slice_profiles(seg: np.ndarray, classes: list[int], zooms: np.ndarray) -> dict:
    """Per-axial-slice area and largest in-plane width of each class.

    Width is 2 * max(EDT) - 1 voxels: exact for squares and straight strips
    (a single voxel is 1, a 7x7 square is 7), slightly under the diameter for
    round shapes.

    Args:
        seg: (X, Y, Z) label array.
        classes: Class ids.
        zooms: Voxel spacing (mm).

    Returns:
        {class: {"area_mm2": [...], "width_vox": [...]}} with one entry per
        slice (0 where absent), or None for absent classes.
    """
    out = {}
    for c in classes:
        mask = seg == c
        if not mask.any():
            out[c] = None
            continue
        box = _bbox_slices(mask, (1, 1, 0))
        sub = mask[box]
        area = np.zeros(mask.shape[2])
        width = np.zeros(mask.shape[2])
        for k in range(sub.shape[2]):
            sl = sub[:, :, k]
            if sl.any():
                z = box[2].start + k
                area[z] = sl.sum() * zooms[0] * zooms[1]
                width[z] = 2 * distance_transform_edt(sl).max() - 1
        out[c] = {"area_mm2": area.tolist(), "width_vox": width.tolist()}
    return out


def boundary_hu_profile(ct: np.ndarray, mask: np.ndarray, zooms: np.ndarray) -> list[float] | None:
    """Median HU as a function of in-plane signed distance to the mask outline.

    Distances are measured within each axial slice, so the 2-2.5 mm slice
    spacing does not mix voxels from neighboring slices into the 1 mm bins.

    Args:
        ct: CT array (HU).
        mask: Boolean organ mask, same shape as ct.
        zooms: Voxel spacing (mm).

    Returns:
        One median per entry of PROFILE_DISTANCES_MM (negative = inside, NaN
        where no voxel falls in the bin), or None for an empty mask.
    """
    if not mask.any():
        return None
    margin = np.ceil(np.abs(PROFILE_DISTANCES_MM).max() / zooms[:2]).astype(int) + 1
    box = _bbox_slices(mask, (*margin, 0))
    m, c = mask[box], ct[box]
    bins = np.zeros(m.shape, dtype=int)  # 0 is not a profile bin
    for k in range(m.shape[2]):
        sl = m[:, :, k]
        if sl.any():
            signed = distance_transform_edt(~sl, sampling=zooms[:2]) - distance_transform_edt(sl, sampling=zooms[:2])
            bins[:, :, k] = np.rint(signed)
    out = []
    for d in PROFILE_DISTANCES_MM:
        vals = c[bins == d]
        out.append(float(np.median(vals)) if vals.size else float("nan"))
    return out


def interior_hu_stats(ct: np.ndarray, seg: np.ndarray) -> dict:
    """HU statistics away from partial-volume edges: heart interior and trachea lumen.

    Args:
        ct: CT array (HU).
        seg: Label array.

    Returns:
        Heart p25/median/p75 HU after 3 voxels of 3D erosion and trachea HU std
        after 2 voxels of in-plane erosion; NaN where the eroded mask is empty.
    """
    out = {"heart_p25": np.nan, "heart_median": np.nan, "heart_p75": np.nan, "trachea_lumen_std": np.nan}
    heart, trachea = seg == 2, seg == 3
    if heart.any():
        box = _bbox_slices(heart, 1)
        vals = ct[box][binary_erosion(heart[box], STRUCT_6, iterations=3)]
        if vals.size:
            out["heart_p25"], out["heart_median"], out["heart_p75"] = (
                float(v) for v in np.percentile(vals, [25, 50, 75])
            )
    if trachea.any():
        box = _bbox_slices(trachea, 1)
        vals = ct[box][binary_erosion(trachea[box], IN_PLANE_STRUCT, iterations=2)]
        if vals.size:
            out["trachea_lumen_std"] = float(vals.std())
    return out


INTENSITY_GROUPS = {"background": [0], "label 1": [1], "label 2": [2], "label 3": [3], "label 4": [4], "all labels": [1, 2, 3, 4]}


def summary_stats(vals: np.ndarray) -> dict | None:
    """Percentile, center and spread statistics of a set of HU values.

    Args:
        vals: 1D array of HU values.

    Returns:
        p0.5, median, p99.5, mean, std and count, or None if vals is empty.
    """
    if vals.size == 0:
        return None
    p_lo, median, p_hi = np.percentile(vals, [0.5, 50, 99.5])
    return {
        "p0.5": float(p_lo),
        "median": float(median),
        "p99.5": float(p_hi),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "n": int(vals.size),
    }


def intensity_stats(ct: np.ndarray, seg: np.ndarray) -> dict:
    """HU statistics of one patient for each group in INTENSITY_GROUPS.

    Args:
        ct: CT array (HU).
        seg: Label array, same shape as ct.

    Returns:
        {group name: summary_stats of the voxels whose label is in that group}.
    """
    return {name: summary_stats(ct[np.isin(seg, labels)]) for name, labels in INTENSITY_GROUPS.items()}


def resample_to_canonical(
    mask: np.ndarray, zooms: np.ndarray, anchor_mm: np.ndarray, canonical_shape: tuple, canonical_coords_mm: np.ndarray
) -> np.ndarray:
    """Nearest-neighbor resample a mask onto the shared canonical grid around an anchor.

    Args:
        mask: Boolean mask in patient voxel space.
        zooms: Voxel spacing (mm).
        anchor_mm: Patient-space point (mm) mapped to the canonical origin.
        canonical_shape: Shape of the canonical grid.
        canonical_coords_mm: (3, N) canonical voxel coordinates in mm.

    Returns:
        Resampled mask as float32 in canonical_shape.
    """
    patient_vox = (canonical_coords_mm + anchor_mm[:, None]) / zooms[:, None]
    sampled = map_coordinates(mask.astype(np.float32), patient_vox, order=0, mode="constant", cval=0.0)
    return sampled.reshape(canonical_shape)


def find_patients(data_dir: Path) -> list[Path]:
    """Sorted Patient_* directories under data_dir.

    Args:
        data_dir: Directory holding one folder per patient.

    Returns:
        Patient directories in name order.
    """
    return sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("Patient_"))


CACHE_JSON = "_checks_cache.json"
CACHE_NPZ = "_checks_cache.npz"
CACHE_ARRAYS = ("spacings", "sizes", "all_fg", "all_bg", "tissue_footprints", "organ_footprints")
# rows whose class-id keys become strings in JSON
CLASS_KEYED_ROWS = ("cc_rows", "centroid_rows", "slice_rows", "boundary_rows", "hist_rows")


def compute(data_dir: Path) -> dict:
    """Per-patient pass over every CT/label pair (a few minutes for 20 patients).

    Args:
        data_dir: Directory holding Patient_* folders with CT and GT.nii.gz.

    Returns:
        Everything the figure functions and the markdown report need, in a
        form save_cache() can write to disk.
    """
    patients = find_patients(data_dir)
    print(f"Found {len(patients)} patients in {data_dir}")

    integrity_rows, crop_rows, cc_rows, adjacency_rows = [], [], [], []
    centroid_rows, orientation_rows, contrast_rows, distance_rows = [], [], [], []
    slice_rows, boundary_rows, interior_rows, hist_rows = [], [], [], []
    spacings, sizes, affine_diffs = [], [], []
    tissue_footprints, organ_footprints = [], []
    fg_by_class = {c: [] for c in CLASSES}
    bg_intensities, intensity_rows = [], []
    bg_sample_per_patient = 50_000  # background is ~99% of voxels; subsample to keep this tractable

    axes_1d = [np.arange(lo, hi + 1e-6, CANONICAL_SPACING_MM) for lo, hi in CANONICAL_EXTENT_MM.values()]
    canonical_shape = tuple(len(a) for a in axes_1d)
    canonical_coords_mm = np.stack([g.ravel() for g in np.meshgrid(*axes_1d, indexing="ij")], axis=0)
    occupancy_sum = {c: np.zeros(canonical_shape, dtype=np.float32) for c in CLASSES}
    occupancy_n = 0

    rs = np.random.RandomState(1234)  # matches fingerprint_extractor.py's seed
    for p in patients:
        pid = p.name
        img = nib.load(str(p / f"{pid}.nii.gz"))
        seg = nib.load(str(p / "GT.nii.gz"))
        zooms = np.array(img.header.get_zooms()[:3])

        check = check_case(img, seg)
        seg_data = np.asarray(seg.dataobj).astype(np.int16)
        labels_info = verify_labels(seg_data, EXPECTED_LABELS)
        integrity_rows.append({"pid": pid, **check, **labels_info})
        affine_diffs.append(float(np.max(np.abs(img.affine - seg.affine))))
        orientation_rows.append({"pid": pid, **check_orientation(img, seg)})

        ct_data = np.asarray(img.dataobj)[None].astype(np.float32)
        crop = crop_to_nonzero(ct_data)
        crop_rows.append({"pid": pid, **crop})
        ct = ct_data[0]

        spacings.append(zooms.tolist())
        sizes.append(list(img.shape))

        cc = connected_components_per_class(seg_data, CLASSES)
        cc_rows.append({"pid": pid, **cc})
        adjacency_rows.append({"pid": pid, **organ_pair_adjacency(seg_data, CLASSES)})
        distance_rows.append({"pid": pid, **pairwise_min_surface_distance(seg_data, CLASSES, zooms)})

        centroid_bbox = organ_centroid_and_bbox(seg_data, CLASSES, zooms)
        trachea_end = trachea_inferior_end_mm(seg_data, zooms)
        centroid_rows.append({"pid": pid, "trachea_end_mm": trachea_end, **centroid_bbox})

        slice_rows.append(
            {
                "pid": pid,
                "n_slices": int(seg_data.shape[2]),
                "z_spacing": float(zooms[2]),
                "trachea_end_mm": trachea_end,
                **slice_profiles(seg_data, CLASSES, zooms),
            }
        )
        boundary_rows.append({"pid": pid, **{c: boundary_hu_profile(ct, seg_data == c, zooms) for c in CLASSES}})
        interior_rows.append({"pid": pid, **interior_hu_stats(ct, seg_data)})
        hist_rows.append(
            {"pid": pid, **{c: np.histogram(ct[seg_data == c], HU_BIN_EDGES)[0].tolist() for c in CLASSES}}
        )
        tissue_footprints.append((ct > TISSUE_HU_THRESHOLD).any(axis=2))
        organ_footprints.append([(seg_data == c).any(axis=2) for c in CLASSES])

        if trachea_end is not None:
            anchor_mm = np.array(trachea_end)
            for c in CLASSES:
                occupancy_sum[c] += resample_to_canonical(
                    seg_data == c, zooms, anchor_mm, canonical_shape, canonical_coords_mm
                )
            occupancy_n += 1

        # nnU-Net's CTNormalization clips/z-scores on ALL foreground pooled.
        intensity_rows.append({"pid": pid, **intensity_stats(ct, seg_data)})
        for c in CLASSES:
            fg_by_class[c].append(ct[seg_data == c])
        contrast_rows.append({"pid": pid, **contrast_spike_stats(ct, zooms)})

        bg_vals = ct[seg_data == 0]
        if len(bg_vals) > bg_sample_per_patient:
            bg_vals = rs.choice(bg_vals, bg_sample_per_patient, replace=False)
        bg_intensities.append(bg_vals)

        print(
            f"  {pid}: shape_match={check['shape_match']} spacing_match={check['spacing_match']} "
            f"affine_match={check['affine_match']} unexpected_labels={labels_info['unexpected_labels']} "
            f"rel_size_after_crop={crop['relative_size_after_cropping']:.4f} cc={cc}"
        )

    spacings = np.array(spacings)
    sizes = np.array(sizes)
    aniso = determine_fullres_target_spacing(spacings, sizes)
    occupancy = {c: occupancy_sum[c] / max(occupancy_n, 1) for c in CLASSES}

    all_axcodes = {r["image_axcodes"] for r in orientation_rows}
    orientation_consistent = len(all_axcodes) == 1 and all(r["orientation_match"] for r in orientation_rows)
    median_rel_size = float(np.median([r["relative_size_after_cropping"] for r in crop_rows]))

    cc_summary = {}
    for c in CLASSES:
        summary = {}
        n_present = 0
        for key in CONNECTIVITY:
            counts = [r[c][key] for r in cc_rows if r[c][key] > 0]
            n_present = len(counts)
            summary[f"always_single_component_{key}"] = all(n == 1 for n in counts) if counts else None
            summary[f"max_components_seen_{key}"] = max(counts) if counts else 0
        cc_summary[c] = {"n_patients_present": n_present, **summary}

    pooled_by_class = {c: np.concatenate(v) for c, v in fg_by_class.items()}
    all_fg = np.concatenate(list(pooled_by_class.values()))
    pooled_intensity = {
        name: summary_stats(np.concatenate([pooled_by_class[c] for c in labels]))
        for name, labels in INTENSITY_GROUPS.items()
        if name != "background"
    }
    p00_5, p99_5 = np.percentile(all_fg, [0.5, 99.5])
    fg_mean, fg_std = float(all_fg.mean()), float(all_fg.std())
    all_bg = np.concatenate(bg_intensities)
    pooled_intensity = {"background": summary_stats(all_bg), **pooled_intensity}

    results = {
        "integrity": integrity_rows,
        "integrity_summary": {
            "all_shape_match": all(r["shape_match"] for r in integrity_rows),
            "all_spacing_match": all(r["spacing_match"] for r in integrity_rows),
            "all_affine_match": all(r["affine_match"] for r in integrity_rows),
            "any_unexpected_labels": any(r["unexpected_labels"] for r in integrity_rows),
            "labels_missing_in_every_patient": sorted(
                set.intersection(*[set(r["missing_labels"]) for r in integrity_rows])
            ),
            "max_affine_abs_diff_per_patient": affine_diffs,
        },
        "crop_to_nonzero": crop_rows,
        "median_relative_size_after_cropping": median_rel_size,
        "mask_restricted_normalization_would_trigger": median_rel_size < MASK_NORM_RELATIVE_SIZE_THRESHOLD,
        "anisotropy": aniso,
        "connected_components": cc_rows,
        "connected_components_summary": cc_summary,
        "organ_pair_adjacency": adjacency_rows,
        "organ_centroids": centroid_rows,
        "orientation": orientation_rows,
        "orientation_consistent": orientation_consistent,
        "contrast_spikes": contrast_rows,
        "any_contrast_spike": any(r["exceeds_ceiling"] for r in contrast_rows),
        "pairwise_min_distance_mm": distance_rows,
        "interior_hu": interior_rows,
        "intensity_per_patient": intensity_rows,
        "intensity_pooled": pooled_intensity,
        "ct_normalization": {
            "clip_lower_p0_5": float(p00_5),
            "clip_upper_p99_5": float(p99_5),
            "mean": fg_mean,
            "std": fg_std,
            "n_foreground_voxels_pooled": int(len(all_fg)),
            "n_background_voxels_sampled": int(len(all_bg)),
        },
    }

    return {
        "results": results,
        "crop_rows": crop_rows,
        "cc_rows": cc_rows,
        "adjacency_rows": adjacency_rows,
        "centroid_rows": centroid_rows,
        "contrast_rows": contrast_rows,
        "distance_rows": distance_rows,
        "slice_rows": slice_rows,
        "boundary_rows": boundary_rows,
        "interior_rows": interior_rows,
        "intensity_rows": intensity_rows,
        "pooled_intensity": pooled_intensity,
        "hist_rows": hist_rows,
        "occupancy": occupancy,
        "canonical_shape": canonical_shape,
        "spacings": spacings,
        "sizes": sizes,
        "all_fg": all_fg,
        "all_bg": all_bg,
        "tissue_footprints": np.array(tissue_footprints),
        "organ_footprints": np.array(organ_footprints),
        "p00_5": float(p00_5),
        "p99_5": float(p99_5),
        "fg_mean": fg_mean,
        "fg_std": fg_std,
    }


def save_cache(data: dict, out_dir: Path) -> None:
    """Write compute() output to a JSON + NPZ pair in out_dir.

    Args:
        data: Output of compute().
        out_dir: Directory for the cache files.
    """
    occupancy = {f"occupancy_class_{c}": v for c, v in data["occupancy"].items()}
    np.savez_compressed(out_dir / CACHE_NPZ, **{k: data[k] for k in CACHE_ARRAYS}, **occupancy)
    json_part = {k: v for k, v in data.items() if k not in (*CACHE_ARRAYS, "occupancy")}
    with open(out_dir / CACHE_JSON, "w") as f:
        json.dump(json_part, f, default=str)
    print(f"Cached per-patient computation to {out_dir / CACHE_JSON} and {out_dir / CACHE_NPZ}")


def _int_keys(d: dict) -> dict:
    return {int(k) if k.isdigit() else k: v for k, v in d.items()}


def cache_is_for(data: dict, data_dir: Path) -> bool:
    """Whether a cached compute() dict was computed from data_dir.

    Args:
        data: The dict read from the cache's JSON file.
        data_dir: Directory of the Patient_* folders now being analyzed.

    Returns:
        True if the cache records this same directory; caches that record none are not trusted.
    """
    return data.get("data_dir") == str(data_dir.resolve())


def load_cache(out_dir: Path, data_dir: Path) -> dict | None:
    """Load a cache written by save_cache().

    Args:
        out_dir: Directory holding the cache files.
        data_dir: Directory of the Patient_* folders now being analyzed.

    Returns:
        The compute() dict, or None if the cache is missing, from an older version of
        this script, or computed from a different data directory.
    """
    json_path, npz_path = out_dir / CACHE_JSON, out_dir / CACHE_NPZ
    if not json_path.is_file() or not npz_path.is_file():
        return None
    with open(json_path) as f:
        data = json.load(f)
    if not cache_is_for(data, data_dir):
        print(f"Cache in {out_dir} was not computed from {data_dir}; recomputing")
        return None
    npz = np.load(npz_path)
    if not all(k in data for k in (*CLASS_KEYED_ROWS, "intensity_rows")) or not all(
        k in npz.files for k in CACHE_ARRAYS
    ):
        print("Cache is from an older version of this script; recomputing")
        return None
    data.update({k: npz[k] for k in CACHE_ARRAYS})
    data["occupancy"] = {int(k.split("_")[-1]): npz[k] for k in npz.files if k.startswith("occupancy_class_")}
    for key in CLASS_KEYED_ROWS:
        data[key] = [_int_keys(row) for row in data[key]]
    data["results"]["connected_components_summary"] = _int_keys(data["results"]["connected_components_summary"])
    print(f"Loaded cached per-patient computation from {json_path} (skip with --recompute)")
    return data


# key -> figure fn taking (data, out_dir, data_dir); only connected_components
# reads data_dir, to reload a few GT files for a real anatomical render.
FIGURES = {
    "grid_geometry": lambda d, o, dd: _fig_grid_geometry(
        d["sizes"], d["spacings"], [r["pid"] for r in d["crop_rows"]], o
    ),
    "tissue_footprint": lambda d, o, dd: _fig_tissue_footprint(d["tissue_footprints"], d["organ_footprints"], o),
    "z_coverage": lambda d, o, dd: _fig_z_coverage(d["slice_rows"], o),
    "slice_area": lambda d, o, dd: _fig_slice_area(d["slice_rows"], o),
    "organ_width": lambda d, o, dd: _fig_organ_width(d["slice_rows"], o),
    "ct_normalization": lambda d, o, dd: _fig_ct_normalization(
        d["all_fg"], d["all_bg"], d["p00_5"], d["p99_5"], d["fg_mean"], d["fg_std"], o
    ),
    "intensity_stats": lambda d, o, dd: _fig_intensity_stats(d["intensity_rows"], d["pooled_intensity"], o),
    "hu_ridge": lambda d, o, dd: _fig_hu_ridge(d["hist_rows"], o),
    "boundary_profile": lambda d, o, dd: _fig_boundary_profile(d["boundary_rows"], o),
    "heart_hu_vs_noise": lambda d, o, dd: _fig_heart_hu_vs_noise(d["interior_rows"], d["spacings"], o),
    "connected_components": lambda d, o, dd: _fig_connected_components(d["cc_rows"], dd, o),
    "centroids_3d": lambda d, o, dd: _fig_centroids_3d(d["centroid_rows"], o),
    "occupancy": lambda d, o, dd: _fig_occupancy(d["occupancy"], o),
    "contrast_spikes": lambda d, o, dd: _fig_contrast_spikes(
        d["contrast_rows"], d["slice_rows"], d["tissue_footprints"], d["spacings"], o
    ),
}


def main():
    """Compute (or load cached) per-patient data, then write figures, JSON and markdown."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_part1/train"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help=f"Comma-separated figure keys to (re)generate, skipping the rest. Choices: {','.join(FIGURES)}.",
    )
    ap.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore any cached per-patient data and recompute from --data-dir.",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    data = None if args.recompute else load_cache(args.out_dir, args.data_dir)
    if data is None:
        data = compute(args.data_dir)
        data["data_dir"] = str(args.data_dir.resolve())
        save_cache(data, args.out_dir)

    global ORGANS_WITH_VOXELS
    missing = set(data["results"]["integrity_summary"]["labels_missing_in_every_patient"])
    ORGANS_WITH_VOXELS = [c for c in CLASSES if c not in missing]

    with open(args.out_dir / "nnunet_checks.json", "w") as f:
        json.dump(data["results"], f, indent=2, default=str)

    keys = [k.strip() for k in args.only.split(",")] if args.only else list(FIGURES)
    for k in keys:
        if k not in FIGURES:
            print(f"  skip unknown figure key {k!r} (choices: {', '.join(FIGURES)})")
            continue
        FIGURES[k](data, args.out_dir, args.data_dir)
        print(f"  wrote figure: {k}")

    _write_markdown(data["results"], args.out_dir)
    print(f"\nWrote nnunet_checks.json, nnunet_checks.md and {len(keys)} figure(s) to {args.out_dir}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _present(rows, c):
    return [r for r in rows if r.get(c) is not None]


def _equal_aspect_3d(ax, pts):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center, half = (lo + hi) / 2, max((hi - lo).max() / 2, 1.0)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))


def _runs(flags):
    edges = np.diff(np.concatenate([[0], flags.astype(int), [0]]))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _organ_label(c):
    return f"label {c}"


def _pid_num(pid):
    return pid.replace("Patient_", "P")


def _fig_grid_geometry(sizes, spacings, pids, out_dir):
    fov = sizes * spacings
    df = pd.DataFrame(
        {
            "patient": [_pid_num(p) for p in pids],
            "in-plane pixel (mm)": spacings[:, 0].round(2),
            "slice spacing (mm)": spacings[:, 2].round(2),
            "slices": sizes[:, 2],
            "in-plane field of view (mm)": fov[:, 0],
            "scan length (mm)": fov[:, 2],
        }
    ).sort_values(["in-plane pixel (mm)", "slice spacing (mm)", "scan length (mm)"])
    columns = list(df.columns[1:])
    g = sns.PairGrid(df, x_vars=columns, y_vars=["patient"], height=7, aspect=0.38)
    for k, (ax, col) in enumerate(zip(g.axes.flat, columns)):
        sns.stripplot(
            data=df,
            x=col,
            y="patient",
            ax=ax,
            color=PALETTE[k],
            size=9,
            jitter=False,
            linewidth=0.6,
            edgecolor="#333333",
        )
        ax.set(xlabel="", ylabel="")
        ax.set_title(col, fontsize=11.5)
        ax.tick_params(axis="y", left=False)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, color="#E6E6E6")
    sns.despine(left=True)
    decorate(
        g.figure,
        "Scan Grid per Patient",
        subtitle="One row per patient, sorted by pixel size, slice spacing, then scan length. Every scan is 512x512 "
        "pixels in-plane, so a larger pixel means a wider field of view.",
    )
    g.figure.savefig(out_dir / "grid_geometry.png")
    plt.close(g.figure)


def _tissue_background(ax, tissue_footprints):
    """Gray image of the fraction of patients with tissue at each pixel, anterior at the top."""
    cmap = LinearSegmentedColormap.from_list("tissue", ["white", "#8C8C8C"])
    im = ax.imshow(tissue_footprints.mean(axis=0).T, cmap=cmap, vmin=0, vmax=1, origin="upper", interpolation="nearest")
    ax.grid(False)
    return im


def _fig_tissue_footprint(tissue_footprints, organ_footprints, out_dir):
    fig, ax = plt.subplots(figsize=(9, 9))
    im = _tissue_background(ax, tissue_footprints)
    for i, c in enumerate(CLASSES):
        for fp in organ_footprints[:, i]:
            if fp.any():
                ax.contour(fp.T.astype(float), levels=[0.5], colors=[PALETTE[i]], linewidths=0.9, alpha=0.8)
    ax.grid(False)
    ax.set_xlabel("x pixel index (patient right -> left)")
    ax.set_ylabel("y pixel index (anterior -> posterior)")
    fig.colorbar(
        im, ax=ax, fraction=0.046, pad=0.03, label=f"fraction of patients with any voxel > {TISSUE_HU_THRESHOLD} HU"
    )
    handles = [Line2D([], [], color=PALETTE[i], lw=2, label=_organ_label(c)) for i, c in enumerate(ORGANS_WITH_VOXELS)]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    decorate(
        fig,
        "In-Plane Tissue and Organ Footprint",
        subtitle="Gray: how many patients have tissue at each pixel in any slice. Lines: each patient's "
        "organ outline projected through all slices, on the native 512x512 pixel grid.",
    )
    fig.savefig(out_dir / "tissue_footprint.png")
    plt.close(fig)


HEIGHT_LABEL = "height (mm above lowest slice of label 3)"


def _slice_heights(r):
    return np.arange(r["n_slices"]) * r["z_spacing"] - r["trachea_end_mm"][2]


def _draw_scan_rows(ax, slice_rows, organs, offsets, band_lw=14, organ_lw=3.5):
    """One row per patient: gray scanned range and colored labeled ranges, in trachea-relative height."""
    for row, r in enumerate(slice_rows):
        h, dz = _slice_heights(r), r["z_spacing"]
        ax.plot([h[0], h[-1] + dz], [row, row], color="#E3E3E3", lw=band_lw, solid_capstyle="butt", zorder=1)
        for c in organs:
            for start, stop in _runs(np.asarray(r[c]["area_mm2"]) > 0):
                y = row + offsets[c]
                color = PALETTE[ORGANS_WITH_VOXELS.index(c)]
                ax.plot([h[start], h[stop - 1] + dz], [y, y], color=color, lw=organ_lw, solid_capstyle="butt", zorder=2)
    ax.set_yticks(range(len(slice_rows)), [_pid_num(r["pid"]) for r in slice_rows])
    ax.set_ylim(len(slice_rows) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x")


def _fig_z_coverage(slice_rows, out_dir):
    fig, (count_ax, ax) = plt.subplots(2, 1, figsize=(11, 11), sharex=True, gridspec_kw={"height_ratios": [1, 3.2]})
    offsets = dict(zip(ORGANS_WITH_VOXELS, np.linspace(0.3, -0.3, len(ORGANS_WITH_VOXELS))))
    _draw_scan_rows(ax, slice_rows, ORGANS_WITH_VOXELS, offsets)
    lo = min(_slice_heights(r)[0] for r in slice_rows)
    hi = max(_slice_heights(r)[-1] for r in slice_rows)
    edges = np.arange(np.floor(lo / 5) * 5, hi + 10, 5.0)
    centers = edges[:-1] + 2.5
    scanned = np.zeros(len(centers))
    for r in slice_rows:
        h = _slice_heights(r)
        scanned += (centers >= h[0]) & (centers <= h[-1] + r["z_spacing"])
    count_ax.fill_between(centers, scanned, step="mid", color="#E3E3E3", label="scanned")
    for i, c in enumerate(ORGANS_WITH_VOXELS):
        present = np.zeros(len(centers))
        for r in slice_rows:
            h = _slice_heights(r)[np.asarray(r[c]["area_mm2"]) > 0]
            present += np.histogram(h, edges)[0] > 0
        count_ax.step(centers, present, where="mid", color=PALETTE[i], lw=2, label=_organ_label(c))
    count_ax.set_ylabel("patients")
    count_ax.set_ylim(0, len(slice_rows) + 1)
    count_ax.set_title("All patients: how many have each organ (and any scan) at this height", fontsize=11.5)
    ax.set_title("Each patient: gray = scanned range, colors = labeled organ", fontsize=11.5)
    ax.set_xlabel(HEIGHT_LABEL)
    legend_below(count_ax, ncol=4)
    decorate(
        fig,
        "Organ Coverage Along the Body Axis",
        subtitle="All patients aligned so their lowest slice of label 3 is at 0; left is toward the feet, right toward "
        "the head.",
    )
    fig.savefig(out_dir / "z_coverage.png")
    plt.close(fig)


HEIGHT_BIN_EDGES_MM = np.arange(-210.0, 155.0, 5.0)


def _plot_slice_profile(slice_rows, key, unit, title, subtitle, path):
    """Patient x height heatmap per organ of a per-slice measurement, blank where the label is absent."""
    n_bins = len(HEIGHT_BIN_EDGES_MM) - 1
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)
    for ax, (i, c) in zip(axes, enumerate(ORGANS_WITH_VOXELS)):
        grid = np.full((len(slice_rows), n_bins), np.nan)
        for row, r in enumerate(slice_rows):
            if r.get(c) is None:
                continue
            vals = np.asarray(r[c][key])
            z = _slice_heights(r)
            idx = np.digitize(z, HEIGHT_BIN_EDGES_MM) - 1
            ok = (vals > 0) & (idx >= 0) & (idx < n_bins)
            for b in np.unique(idx[ok]):
                grid[row, b] = vals[ok][idx[ok] == b].mean()
        cmap = LinearSegmentedColormap.from_list(key, ["#F1F1F1", PALETTE[i], "#1F1F1F"])
        cmap.set_bad("white")
        extent = (HEIGHT_BIN_EDGES_MM[0], HEIGHT_BIN_EDGES_MM[-1], len(slice_rows) - 0.5, -0.5)
        im = ax.imshow(grid, aspect="auto", cmap=cmap, extent=extent, interpolation="nearest", vmin=0)
        ax.axvline(0, color="#999999", lw=0.8)
        ax.grid(False)
        ax.set_title(_organ_label(c), fontsize=12)
        ax.set_xlabel(HEIGHT_LABEL)
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.12, fraction=0.05, label=unit)
    axes[0].set_yticks(range(len(slice_rows)), [_pid_num(r["pid"]) for r in slice_rows])
    decorate(fig, title, subtitle=subtitle)
    fig.savefig(path)
    plt.close(fig)


def _fig_slice_area(slice_rows, out_dir):
    _plot_slice_profile(
        slice_rows,
        "area_mm2",
        "cross-section area (mm²)",
        "Organ Cross-Section by Patient and Height",
        "Each row is one patient; color is the organ's area on the slices at that height (5 mm bins), blank "
        "where the organ is not labeled. Heights are relative to the lowest slice of label 3.",
        out_dir / "slice_area_profile.png",
    )


def _fig_organ_width(slice_rows, out_dir):
    _plot_slice_profile(
        slice_rows,
        "width_vox",
        "width (pixels, 512x512 image)",
        "Organ Width by Patient and Height",
        "Each row is one patient; color is the organ's widest in-plane extent in image pixels at that height "
        "(5 mm bins), blank where the organ is not labeled.",
        out_dir / "organ_width.png",
    )


def _fig_ct_normalization(all_fg, all_bg, p00_5, p99_5, mean, std, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True)
    xlim = (-1200, 600)

    ax = axes[0]
    ax.hist(np.clip(all_bg, *xlim), bins=200, color=PALETTE[4], edgecolor="none")
    ax.set_title("Background voxels (random 50k per patient)", fontsize=12)
    ax.set_ylabel("voxel count")

    ax2 = axes[1]
    ax2.hist(np.clip(all_fg, *xlim), bins=200, color=PALETTE[2], edgecolor="none")
    ax2.axvline(p00_5, color=PALETTE[5], linestyle="--", linewidth=1.5)
    ax2.axvline(p99_5, color=PALETTE[5], linestyle="--", linewidth=1.5)
    ax2.axvspan(mean - std, mean + std, color=PALETTE[0], alpha=0.15)
    top = ax2.get_ylim()[1]
    ax2.text(p00_5, top, f" p0.5 = {p00_5:.0f}", fontsize=9, color=PALETTE[5], va="top", ha="left")
    ax2.text(p99_5, top, f"p99.5 = {p99_5:.0f} ", fontsize=9, color=PALETTE[5], va="top", ha="right")
    ax2.text(
        0.6, 0.8, f"shaded: mean {mean:.0f} ± std {std:.0f}", fontsize=9, color=PALETTE[0], transform=ax2.transAxes
    )
    ax2.set_title("Labeled organ voxels (all)", fontsize=12)

    for a in axes:
        a.set_xlabel("Hounsfield Units (clipped to -1200..600 for display)")
        a.set_xlim(*xlim)

    decorate(
        fig,
        "CT Intensity: Background vs Labeled Organs",
        subtitle="Dashed lines and shaded band are the clip percentiles, mean and std that nnU-Net's "
        "CTNormalization computes from the pooled organ voxels.",
    )
    fig.savefig(out_dir / "ct_normalization.png")
    plt.close(fig)


def _fig_intensity_stats(intensity_rows, pooled, out_dir):
    # Only organs with voxels in this dataset get their own panel; "background"
    # and "all labels" always do.
    groups = ["background"] + [f"label {c}" for c in ORGANS_WITH_VOXELS] + ["all labels"]
    colors = {"background": "#8C8C8C", "all labels": "#333333"}
    colors |= {f"label {c}": PALETTE[i % len(PALETTE)] for i, c in enumerate(ORGANS_WITH_VOXELS)}
    rows = [(_pid_num(r["pid"]), r) for r in intensity_rows] + [("pooled", pooled)]
    fig, axes = plt.subplots(1, len(groups), figsize=(18, 9), sharey=True)
    for ax, g in zip(axes, groups):
        for y, (_, r) in enumerate(rows):
            st = r.get(g)
            if st is None:
                continue
            color = colors[g]
            ax.hlines(y, st["p0.5"], st["p99.5"], color=color, lw=1.2)
            ax.hlines(y, st["mean"] - st["std"], st["mean"] + st["std"], color=color, lw=6, alpha=0.35)
            ax.scatter(st["median"], y, color=color, s=40, zorder=3)
            ax.scatter(st["mean"], y, facecolor="white", edgecolor=color, marker="D", s=34, lw=1.3, zorder=4)
        ax.axhline(len(rows) - 1.5, color="#999999", lw=0.8)
        title = g if g in ("background", "all labels") else _organ_label(int(g.split()[1]))
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("HU")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x")
    axes[0].set_yticks(range(len(rows)), [name for name, _ in rows])
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    axes[0].plot([], [], color="#555555", lw=1.2, label="0.5th to 99.5th percentile")
    axes[0].plot([], [], color="#555555", lw=6, alpha=0.35, label="mean ± std")
    axes[0].scatter([], [], color="#555555", s=40, label="median")
    axes[0].scatter([], [], facecolor="white", edgecolor="#555555", marker="D", s=34, label="mean")
    legend_below(axes[0], ncol=4)
    decorate(
        fig,
        "Intensity Statistics per Label, per Patient",
        subtitle="Each row uses only that patient's voxels; the bottom row pools all patients (background pooled "
        "from 50k random voxels per patient). Exact values: intensity_stats_per_patient.csv.",
    )
    fig.savefig(out_dir / "intensity_stats_per_patient.png")
    plt.close(fig)
    table = [{"patient": name, "group": g, **r[g]} for name, r in rows for g in groups if r.get(g) is not None]
    pd.DataFrame(table).to_csv(out_dir / "intensity_stats_per_patient.csv", index=False, float_format="%.1f")


def _fig_hu_ridge(hist_rows, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    centers = (HU_BIN_EDGES[:-1] + HU_BIN_EDGES[1:]) / 2
    width = np.diff(HU_BIN_EDGES)
    for ax, (i, c) in zip(axes, enumerate(ORGANS_WITH_VOXELS)):
        for r in hist_rows:
            counts = np.asarray(r[c], dtype=float)
            if counts.sum():
                ax.plot(centers, counts / (counts.sum() * width), color=PALETTE[i], lw=1, alpha=0.7)
        ax.set_yticks([])
        ax.set_ylabel(_organ_label(c), rotation=0, ha="right", va="center", fontsize=12)
        ax.grid(axis="x")
    axes[-1].set_xlabel("Hounsfield Units")
    decorate(
        fig,
        "HU Distribution per Organ, per Patient",
        subtitle="One line per patient: normalized histogram (10 HU bins) of every voxel inside the label.",
    )
    fig.savefig(out_dir / "hu_ridge_by_organ.png")
    plt.close(fig)


def _fig_boundary_profile(boundary_rows, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), sharey=True)
    for ax, (i, c) in zip(axes, enumerate(ORGANS_WITH_VOXELS)):
        for r in _present(boundary_rows, c):
            ax.plot(PROFILE_DISTANCES_MM, r[c], color=PALETTE[i], lw=1, alpha=0.7, marker="o", ms=2)
        ax.axvline(0, color="#BBBBBB", lw=1, zorder=0)
        ax.set_title(_organ_label(c), fontsize=12)
        ax.set_xlabel("distance from the annotated outline (mm)\n<- inside the organ    outside ->")
        ax.grid(axis="x")
    axes[0].set_ylabel("median HU at that distance")
    decorate(
        fig,
        "Brightness Across Each Organ's Outline",
        subtitle="0 is the outline the annotator drew on each slice. One line per patient: median HU of the voxels "
        "at each distance inside (left) or outside (right) that outline. A steep jump at 0 means the edge is "
        "visible in the image.",
    )
    fig.savefig(out_dir / "boundary_hu_profile.png")
    plt.close(fig)


def _fig_heart_hu_vs_noise(interior_rows, spacings, out_dir):
    fig, ax = plt.subplots(figsize=(9, 7))
    med = np.array([r["heart_median"] for r in interior_rows])
    p25 = np.array([r["heart_p25"] for r in interior_rows])
    p75 = np.array([r["heart_p75"] for r in interior_rows])
    noise = np.array([r["trachea_lumen_std"] for r in interior_rows])
    z_sp = np.round(spacings[:, 2], 2)
    for k, v in enumerate(np.unique(z_sp)):
        sel = z_sp == v
        ax.hlines(noise[sel], p25[sel], p75[sel], color=PALETTE[k], lw=2, alpha=0.6)
        ax.scatter(med[sel], noise[sel], color=PALETTE[k], s=60, edgecolor="white", zorder=3, label=f"{v:g} mm slices")
    for r, x, y in zip(interior_rows, med, noise):
        ax.annotate(
            _pid_num(r["pid"]), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8.5, color="#555555"
        )
    ax.set_xlabel("label 2 interior HU (dot = median, line = 25th-75th percentile)")
    ax.set_ylabel("HU std inside eroded label 3 (log scale)")
    ax.set_yscale("log")
    ax.set_yticks([20, 30, 50, 100, 200, 300], ["20", "30", "50", "100", "200", "300"])
    ax.minorticks_off()
    ax.grid(axis="x")
    legend_below(ax, ncol=2)
    decorate(
        fig,
        "Label 2 Interior vs Label 3 Core Variation per Patient",
        subtitle="Label 2 eroded 3 voxels to skip its edge; label 3 eroded 2 voxels in-plane to keep only its core.",
    )
    fig.savefig(out_dir / "heart_hu_vs_noise.png")
    plt.close(fig)


def _fig_connected_components(cc_rows, data_dir, out_dir):
    cases = [(r["pid"], c) for r in cc_rows for c in CLASSES if r[c]["n6"] > 1]
    n_present = sum(r[c]["n6"] > 0 for r in cc_rows for c in CLASSES)
    subtitle = (
        f"A label counts as one piece if its voxels connect through shared faces. {n_present - len(cases)} of "
        f"{n_present} patient-organ labels are one piece; these {len(cases)} are not. Colors are pieces, not organs."
    )
    if not cases:
        fig = plt.figure(figsize=(7, 3))
        decorate(fig, "Every Organ Label Is One Piece", subtitle=subtitle)
        fig.savefig(out_dir / "connected_components.png")
        plt.close(fig)
        return

    piece_colors = ["#4D4D4D", PALETTE[4], PALETTE[6]]
    fig = plt.figure(figsize=(5 * len(cases), 6))
    for i, (pid, c) in enumerate(cases):
        gt = nib.load(str(data_dir / pid / "GT.nii.gz"))
        zooms = np.array(gt.header.get_zooms()[:3])
        mask = np.asarray(gt.dataobj) == c
        labeled, n = label(mask, structure=STRUCT_6)
        crop = labeled[_bbox_slices(mask, 3)]
        sizes_ml = np.bincount(crop.ravel(), minlength=n + 1)[1:] * np.prod(zooms) / 1000

        ax = fig.add_subplot(1, len(cases), i + 1, projection="3d")
        for comp_id in np.argsort(-sizes_ml) + 1:
            sub = (crop == comp_id).astype(np.float32)
            if sub.sum() < 8:
                continue
            verts, faces, _, _ = marching_cubes(sub, level=0.5, spacing=tuple(zooms))
            mesh = Poly3DCollection(verts[faces], alpha=0.7)
            mesh.set_facecolor(piece_colors[min(comp_id - 1, len(piece_colors) - 1)])
            mesh.set_edgecolor("none")
            ax.add_collection3d(mesh)
        extent = [d * s for d, s in zip(crop.shape, zooms)]
        ax.set_xlim(0, extent[0])
        ax.set_ylim(0, extent[1])
        ax.set_zlim(0, extent[2])
        ax.set_box_aspect(extent)
        ax.view_init(elev=5, azim=-90)
        ax.set_axis_off()
        sizes = " + ".join(f"{v:.0f}" for v in sorted(sizes_ml, reverse=True))
        ax.set_title(f"{_pid_num(pid)}, {_organ_label(c)}\n{n} pieces: {sizes} mL", fontsize=11)

    decorate(fig, "Organ Labels That Are Not One Piece", subtitle=subtitle)
    fig.savefig(out_dir / "connected_components.png")
    plt.close(fig)


def _fig_centroids_3d(centroid_rows, out_dir):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    all_pts = []
    for i, c in enumerate(ORGANS_WITH_VOXELS):
        rows = [r for r in _present(centroid_rows, c) if r["trachea_end_mm"] is not None]
        pts = np.array([np.subtract(r[c]["centroid_mm"], r["trachea_end_mm"]) for r in rows])
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color=PALETTE[i],
            s=45,
            alpha=0.85,
            edgecolor="white",
            label=_organ_label(c),
        )
        all_pts.append(pts)
    ax.scatter([0], [0], [0], color="#333333", marker="x", s=60, label="lowest trachea slice")
    _equal_aspect_3d(ax, np.vstack([*all_pts, np.zeros((1, 3))]))
    ax.set_xlabel("x (mm, R -> L)")
    ax.set_ylabel("y (mm, A -> P)")
    ax.set_zlabel("z (mm, I -> S)")
    ax.view_init(elev=15, azim=-60)
    legend_below(ax, ncol=4)
    decorate(
        fig,
        "3D Organ Centroids",
        subtitle="One dot per organ per patient, in mm from that patient's lowest slice of label 3; equal scale on all axes.",
    )
    fig.savefig(out_dir / "organ_centroids_3d.png")
    plt.close(fig)


def _fig_occupancy(occupancy, out_dir):
    levels = [0.25, 0.5, 0.75]
    widths = [0.8, 1.6, 2.8]
    ax_names = {"x": "x (mm, R -> L)", "y": "y (mm, A -> P)", "z": "z (mm, I -> S)"}
    views = [("Front view", "x", "z", 1), ("Side view", "y", "z", 0), ("Top view", "x", "y", 2)]
    extent = CANONICAL_EXTENT_MM
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    for ax, (title, h, v, collapse) in zip(axes, views):
        for i, c in enumerate(ORGANS_WITH_VOXELS):
            ax.contour(
                occupancy[c].max(axis=collapse).T,
                levels=levels,
                colors=[PALETTE[i]],
                linewidths=widths,
                extent=(*extent[h], *extent[v]),
                origin="lower",
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(ax_names[h])
        ax.set_ylabel(ax_names[v])
        ax.set_aspect("equal")
        ax.plot(0, 0, marker="x", color="#333333", ms=7)
        if v == "y":
            ax.invert_yaxis()
    for i, c in enumerate(ORGANS_WITH_VOXELS):
        axes[0].plot([], [], color=PALETTE[i], lw=2.5, label=_organ_label(c))
    for lvl, w in zip(levels, widths):
        axes[0].plot([], [], color="#555555", lw=w, label=f"in {lvl:.0%} of patients")
    legend_below(axes[0], ncol=6)
    decorate(
        fig,
        "Where Each Organ Sits Across Patients",
        subtitle="All patients shifted so their lowest slice of label 3 is at the x. Thin to thick lines enclose "
        "the space that is inside the label in at least 25%, 50% and 75% of patients.",
    )
    fig.savefig(out_dir / "occupancy_envelope.png")
    plt.close(fig)


def _fig_contrast_spikes(rows, slice_rows, tissue_footprints, spacings, out_dir):
    hu = np.array([r["max_hu"] for r in rows])
    loc_mm = np.array([r["max_hu_location_mm"] for r in rows])
    loc_px = loc_mm[:, :2] / spacings[:, :2]
    cmap = LinearSegmentedColormap.from_list("hu", [PALETTE[0], PALETTE[6], PALETTE[5]])
    norm = LogNorm(vmin=hu.min(), vmax=hu.max())

    fig, (top, side) = plt.subplots(1, 2, figsize=(16, 8.5), gridspec_kw={"width_ratios": [1, 1.1]})
    _tissue_background(top, tissue_footprints)
    top.scatter(loc_px[:, 0], loc_px[:, 1], c=hu, cmap=cmap, norm=norm, s=80, edgecolor="black", lw=0.6, zorder=3)
    for r, (x, y) in zip(rows, loc_px):
        top.annotate(_pid_num(r["pid"]), (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    top.set_title("Seen from above (all slices stacked)", fontsize=12)
    top.set_xlabel("x pixel index (patient right -> left)")
    top.set_ylabel("y pixel index (anterior -> posterior)")

    _draw_scan_rows(side, slice_rows, [2], {2: 0.0}, band_lw=12, organ_lw=3)
    heights = loc_mm[:, 2] - np.array([r["trachea_end_mm"][2] for r in slice_rows])
    sc = side.scatter(heights, range(len(rows)), c=hu, cmap=cmap, norm=norm, s=80, edgecolor="black", lw=0.6, zorder=3)
    side.plot([], [], color="#E3E3E3", lw=8, label="whole scan")
    side.plot([], [], color=PALETTE[1], lw=3, label="label 2")
    side.set_title("Height along the body", fontsize=12)
    side.set_xlabel(HEIGHT_LABEL)
    legend_below(side, ncol=2)

    cb = fig.colorbar(sc, ax=side, fraction=0.05, pad=0.02)
    cb.set_label("brightest voxel value (HU, log scale)")
    ticks = [t for t in (SATURATION_CEILING_HU, 10000, 30000) if hu.min() <= t <= hu.max()]
    cb.set_ticks(ticks, labels=[f"{t}\n(12-bit max)" if t == SATURATION_CEILING_HU else str(t) for t in ticks])
    decorate(
        fig,
        "Brightest Voxel per Scan",
        subtitle="Each patient's single highest-HU voxel: where it is seen from above (gray = body in how many "
        "patients) and at what height relative to their lowest slice of label 3. Color is its value.",
    )
    fig.savefig(out_dir / "contrast_spikes.png")
    plt.close(fig)


def _write_markdown(results, out_dir):
    lines = ["# nnU-Net planner/fingerprint checks -- ported from source\n"]

    lines.append("## 1. Dataset integrity (verify_dataset_integrity.py)\n")
    s = results["integrity_summary"]
    lines.append(f"- All image/label shapes match: **{s['all_shape_match']}**")
    lines.append(f"- All image/label spacings match: **{s['all_spacing_match']}**")
    lines.append(f"- All image/label affines match: **{s['all_affine_match']}**")
    lines.append(f"- Any unexpected label values found (outside {{0..4}}): **{s['any_unexpected_labels']}**")
    missing_names = [CLASS_NAMES[v] for v in s["labels_missing_in_every_patient"]]
    lines.append(
        f"- Label(s) missing in EVERY patient: **{s['labels_missing_in_every_patient']}** "
        f"(= {', '.join(missing_names) or 'none'}; verify_labels() only reports unexpected labels, "
        f"so this is checked separately against the 5 expected classes)"
    )
    axcodes = sorted({r["image_axcodes"] for r in results["orientation"]})
    lines.append(
        f"- Coordinate orientation consistent across all patients: "
        f"**{results['orientation_consistent']}** (axis codes seen: {axcodes})"
    )
    lines.append(
        f"- Any voxel above the 12-bit ceiling (>{SATURATION_CEILING_HU} HU): **{results['any_contrast_spike']}**\n"
    )

    lines.append("## 2. crop_to_nonzero (cropping.py)\n")
    rel = [r["relative_size_after_cropping"] for r in results["crop_to_nonzero"]]
    lines.append(f"- Relative size after crop, per patient: min **{min(rel):.4f}**, max **{max(rel):.4f}**")
    lines.append(
        f"- Triggers mask-restricted normalization (median < 0.75)? "
        f"**{results['mask_restricted_normalization_would_trigger']}**"
    )
    lines.append(
        "- CT air is about -1000 HU, not 0, so the nonzero mask covers the whole volume; "
        "see tissue_footprint.png and z_coverage.png for where tissue and organs actually are.\n"
    )

    lines.append("## 3. Anisotropy / target spacing (default_experiment_planner.py)\n")
    a = results["anisotropy"]
    lines.append(f"- median spacing (x,y,z): {[round(v, 4) for v in a['median_spacing']]}")
    lines.append(f"- worst_spacing_axis: {a['worst_spacing_axis']}")
    lines.append(f"- has_aniso_spacing: **{a['has_aniso_spacing']}**")
    lines.append(f"- has_aniso_voxels: **{a['has_aniso_voxels']}**")
    lines.append(f"- cascade override applied: **{a['cascade_override_applied']}**")
    lines.append(f"- final target spacing: {[round(v, 4) for v in a['final_target_spacing']]}\n")

    lines.append("## 4. Connected components per class (paper Sec. 2.5 rule)\n")
    lines.append(
        "| class | n patients present | always single (6-conn) | always single (18-conn) | "
        "always single (26-conn) | max seen (6-conn) | max seen (18-conn) | max seen (26-conn) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for c, cs in results["connected_components_summary"].items():
        lines.append(
            f"| {_organ_label(c)} | {cs['n_patients_present']} | {cs['always_single_component_n6']} | "
            f"{cs['always_single_component_n18']} | {cs['always_single_component_n26']} | "
            f"{cs['max_components_seen_n6']} | {cs['max_components_seen_n18']} | "
            f"{cs['max_components_seen_n26']} |"
        )
    lines.append("")

    lines.append("## 5. Affine consistency, per patient\n")
    lines.append(
        f"- Max |image_affine - seg_affine| across all 20 patients: "
        f"**{max(s['max_affine_abs_diff_per_patient']):.6f}**\n"
    )

    lines.append("## 6. Organ-pair adjacency (fraction of patients where organ A touches organ B)\n")
    rows = results["organ_pair_adjacency"]
    lines.append("| touches -> | " + " | ".join(_organ_label(c) for c in CLASSES) + " |")
    lines.append("|" + "---|" * (len(CLASSES) + 1))
    for a_ in CLASSES:
        cells = ["-" if a_ == b else f"{np.mean([r[f'{a_}_{b}'] > 0 for r in rows]):.0%}" for b in CLASSES]
        lines.append(f"| {CLASS_NAMES[a_]} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 7. CT normalization parameters (CTNormalization.run(), pooled foreground)\n")
    n = results["ct_normalization"]
    lines.append(f"- clip lower (p0.5): **{n['clip_lower_p0_5']:.1f} HU**")
    lines.append(f"- clip upper (p99.5): **{n['clip_upper_p99_5']:.1f} HU**")
    lines.append(f"- mean: **{n['mean']:.1f} HU**, std: **{n['std']:.1f} HU**")
    lines.append(f"- n foreground voxels pooled: {n['n_foreground_voxels_pooled']:,}\n")

    (out_dir / "nnunet_checks.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
