#!/usr/bin/env python3
"""Split the merged esophagus/aorta label in SegTHOR ground truths.

The original ``segthor_part1`` masks encode both the esophagus and aorta as
label 1.  A direct 3-D connected-component split is insufficient because the
two structures touch in the merged mask.  This script therefore:

1. progressively erodes the merged mask;
2. labels eroded components with :func:`skimage.measure.label` using strict
   6-neighbour connectivity;
3. uses the two largest components as watershed seeds;
4. evaluates every plausible split instead of accepting the first one; and
5. prefers a split with anatomically consistent axial endpoints: the aorta
   starts at the low-z end and the esophagus extends to the high-z end.

The larger final component is assigned to the aorta (label 4), and the
smaller component to the esophagus (label 1).  Raw data are never modified:
the corrected patient tree is written to a separate destination.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

from fix_patient15 import make_patient15_split


@dataclass
class SplitReport:
    patient: str
    erosion_iterations: int
    candidate_count: int
    merged_voxels: int
    esophagus_voxels: int
    aorta_voxels: int
    esophagus_volume_cm3: float
    aorta_volume_cm3: float
    esophagus_fraction: float
    esophagus_z_span: int
    merged_z_span: int
    z_coverage: float
    endpoint_error_slices: int
    endpoint_error_fraction: float
    reference_difference_voxels: int | None
    esophagus_reference_dice: float | None
    aorta_reference_dice: float | None
    qc_warning: str


def bounding_box(mask: np.ndarray, margin: int = 1) -> tuple[slice, ...]:
    coordinates = np.where(mask)
    if not coordinates[0].size:
        raise ValueError("The merged-label mask is empty")

    return tuple(
        slice(
            max(0, int(axis_coordinates.min()) - margin),
            min(mask.shape[axis], int(axis_coordinates.max()) + margin + 1),
        )
        for axis, axis_coordinates in enumerate(coordinates)
    )


def axial_span(mask: np.ndarray) -> int:
    occupied = np.where(mask.any(axis=(0, 1)))[0]
    if not occupied.size:
        return 0
    return int(occupied[-1] - occupied[0] + 1)


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    denominator = int(mask_a.sum() + mask_b.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * int(np.logical_and(mask_a, mask_b).sum()) / denominator


def choose_split(
    mask: np.ndarray,
    *,
    min_seed_size: int,
    max_erosion: int,
    min_smaller_fraction: float,
    max_smaller_fraction: float,
    target_smaller_fraction: float,
) -> tuple[np.ndarray, int, int, float, int, int, int]:
    """Return a two-region split and its QC statistics.

    Candidate ranking is lexicographic:

    1. minimize endpoint disagreement (aorta at low z, esophagus at high z);
    2. prefer a smaller-region fraction near ``target_smaller_fraction``;
    3. prefer less erosion when the previous scores tie.
    """

    crop_slices = bounding_box(mask)
    cropped_mask = mask[crop_slices]
    merged_span = axial_span(cropped_mask)
    distance = ndimage.distance_transform_edt(cropped_mask)
    eroded = cropped_mask.copy()

    best_score: tuple[float, float, int] | None = None
    best_split: np.ndarray | None = None
    best_iteration = 0
    best_fraction = 0.0
    best_small_span = 0
    best_endpoint_error = 0
    candidate_count = 0

    for iteration in range(1, max_erosion + 1):
        eroded = ndimage.binary_erosion(eroded, iterations=1)
        components = label(eroded, connectivity=1)
        regions = sorted(
            (region for region in regionprops(components) if region.area >= min_seed_size),
            key=lambda region: region.area,
            reverse=True,
        )
        if len(regions) < 2:
            continue

        markers = np.zeros(cropped_mask.shape, dtype=np.uint8)
        markers[components == regions[0].label] = 1
        markers[components == regions[1].label] = 2
        grown = watershed(-distance, markers, mask=cropped_mask)

        volumes = [int((grown == component_id).sum()) for component_id in (1, 2)]
        total = sum(volumes)
        smaller_component = int(np.argmin(volumes)) + 1
        larger_component = int(np.argmax(volumes)) + 1
        smaller_fraction = min(volumes) / total
        if not min_smaller_fraction <= smaller_fraction <= max_smaller_fraction:
            continue

        smaller_span = axial_span(grown == smaller_component)
        union_z = np.where(cropped_mask.any(axis=(0, 1)))[0]
        aorta_z = np.where((grown == larger_component).any(axis=(0, 1)))[0]
        esophagus_z = np.where((grown == smaller_component).any(axis=(0, 1)))[0]
        endpoint_error = int(
            (aorta_z[0] - union_z[0]) + (union_z[-1] - esophagus_z[-1])
        )
        score = (
            -endpoint_error,
            -abs(smaller_fraction - target_smaller_fraction),
            -iteration,
        )
        candidate_count += 1

        if best_score is None or score > best_score:
            best_score = score
            best_split = grown.copy()
            best_iteration = iteration
            best_fraction = smaller_fraction
            best_small_span = smaller_span
            best_endpoint_error = endpoint_error

    if best_split is None:
        raise RuntimeError(
            "No plausible two-component split was found. "
            "Inspect the mask and consider adjusting the erosion or fraction limits."
        )

    full_split = np.zeros(mask.shape, dtype=np.uint8)
    full_split[crop_slices] = best_split
    return (
        full_split,
        best_iteration,
        candidate_count,
        best_fraction,
        best_small_span,
        merged_span,
        best_endpoint_error,
    )


def correct_patient(
    patient_dir: Path,
    destination_dir: Path,
    *,
    split_label: int,
    esophagus_label: int,
    aorta_label: int,
    min_seed_size: int,
    max_erosion: int,
    min_smaller_fraction: float,
    max_smaller_fraction: float,
    target_smaller_fraction: float,
) -> SplitReport:
    patient = patient_dir.name
    gt_path = patient_dir / "GT.nii.gz"
    ct_path = patient_dir / f"{patient}.nii.gz"
    if not gt_path.exists() or not ct_path.exists():
        raise FileNotFoundError(f"Missing CT or GT for {patient}")

    gt_image = nib.load(str(gt_path))
    original = np.asanyarray(gt_image.dataobj).astype(np.uint8)
    merged_mask = original == split_label

    if patient == "Patient_15":
        # Patient 15's annotations remain joined under progressive 3-D erosion.
        # Track its visible axial esophagus components instead, then use them
        # as dense watershed markers only where the two organs touch.
        split, candidate_count = make_patient15_split(merged_mask)
        erosion_iterations = 0
        esophagus_component = 1
        aorta_component = 2
        smaller_fraction = float((split == esophagus_component).sum() / merged_mask.sum())
        smaller_span = axial_span(split == esophagus_component)
        merged_span = axial_span(merged_mask)
        union_z = np.where(merged_mask.any(axis=(0, 1)))[0]
        aorta_z = np.where((split == aorta_component).any(axis=(0, 1)))[0]
        esophagus_z = np.where((split == esophagus_component).any(axis=(0, 1)))[0]
        endpoint_error = int(
            (aorta_z[0] - union_z[0]) + (union_z[-1] - esophagus_z[-1])
        )
    else:
        (
            split,
            erosion_iterations,
            candidate_count,
            smaller_fraction,
            smaller_span,
            merged_span,
            endpoint_error,
        ) = choose_split(
            merged_mask,
            min_seed_size=min_seed_size,
            max_erosion=max_erosion,
            min_smaller_fraction=min_smaller_fraction,
            max_smaller_fraction=max_smaller_fraction,
            target_smaller_fraction=target_smaller_fraction,
        )

        component_ids = [1, 2]
        component_volumes = {
            component_id: int((split == component_id).sum())
            for component_id in component_ids
        }
        aorta_component = max(component_volumes, key=component_volumes.get)
        esophagus_component = min(component_volumes, key=component_volumes.get)

    corrected = original.copy()
    corrected[merged_mask] = 0
    corrected[split == aorta_component] = aorta_label
    corrected[split == esophagus_component] = esophagus_label

    if not np.array_equal(merged_mask, (corrected == esophagus_label) | (corrected == aorta_label)):
        raise AssertionError(f"{patient}: corrected organ union differs from the merged input")
    for class_id in sorted(set(np.unique(original).astype(int)) - {split_label}):
        if not np.array_equal(original == class_id, corrected == class_id):
            raise AssertionError(f"{patient}: non-target class {class_id} changed")

    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ct_path, destination_dir / ct_path.name)
    output_header = gt_image.header.copy()
    output_header.set_data_dtype(np.uint8)
    output_image = nib.Nifti1Image(corrected, gt_image.affine, output_header)
    nib.save(output_image, str(destination_dir / "GT.nii.gz"))

    reference_difference: int | None = None
    esophagus_reference_dice: float | None = None
    aorta_reference_dice: float | None = None
    reference_path = patient_dir / "GT2.nii.gz"
    if reference_path.exists():
        reference = np.asanyarray(nib.load(str(reference_path)).dataobj).astype(np.uint8)
        if reference.shape != corrected.shape:
            raise ValueError(f"{patient}: GT2 shape differs from corrected output")
        reference_difference = int((corrected != reference).sum())
        esophagus_reference_dice = dice(corrected == esophagus_label, reference == esophagus_label)
        aorta_reference_dice = dice(corrected == aorta_label, reference == aorta_label)

    esophagus_voxels = int((corrected == esophagus_label).sum())
    aorta_voxels = int((corrected == aorta_label).sum())
    voxel_volume_cm3 = float(np.prod(gt_image.header.get_zooms()[:3]) / 1000.0)
    esophagus_volume_cm3 = esophagus_voxels * voxel_volume_cm3
    aorta_volume_cm3 = aorta_voxels * voxel_volume_cm3
    z_coverage = smaller_span / merged_span
    endpoint_error_fraction = endpoint_error / merged_span
    warnings: list[str] = []
    if endpoint_error_fraction > 0.20:
        warnings.append(
            f"endpoint inconsistency ({endpoint_error} slices; "
            f"{endpoint_error_fraction:.2f} of merged z-span)"
        )
    if smaller_fraction < 0.08 or smaller_fraction > 0.40:
        warnings.append(f"unusual smaller-component fraction ({smaller_fraction:.2f})")

    return SplitReport(
        patient=patient,
        erosion_iterations=erosion_iterations,
        candidate_count=candidate_count,
        merged_voxels=int(merged_mask.sum()),
        esophagus_voxels=esophagus_voxels,
        aorta_voxels=aorta_voxels,
        esophagus_volume_cm3=esophagus_volume_cm3,
        aorta_volume_cm3=aorta_volume_cm3,
        esophagus_fraction=smaller_fraction,
        esophagus_z_span=smaller_span,
        merged_z_span=merged_span,
        z_coverage=z_coverage,
        endpoint_error_slices=endpoint_error,
        endpoint_error_fraction=endpoint_error_fraction,
        reference_difference_voxels=reference_difference,
        esophagus_reference_dice=esophagus_reference_dice,
        aorta_reference_dice=aorta_reference_dice,
        qc_warning="; ".join(warnings),
    )


def write_report(reports: list[SplitReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(report) for report in reports]
    with path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_cohort_warnings(reports: list[SplitReport]) -> None:
    """Flag physical-volume outliers after the whole cohort is processed."""

    esophagus_median = float(np.median([report.esophagus_volume_cm3 for report in reports]))
    aorta_median = float(np.median([report.aorta_volume_cm3 for report in reports]))

    for report in reports:
        warnings = [warning for warning in report.qc_warning.split("; ") if warning]
        esophagus_ratio = report.esophagus_volume_cm3 / esophagus_median
        aorta_ratio = report.aorta_volume_cm3 / aorta_median
        if esophagus_ratio < 0.35 or esophagus_ratio > 2.5:
            warnings.append(
                f"esophagus volume outlier ({report.esophagus_volume_cm3:.1f} cm3; "
                f"{esophagus_ratio:.1f}x cohort median)"
            )
        if aorta_ratio < 0.35 or aorta_ratio > 2.5:
            warnings.append(
                f"aorta volume outlier ({report.aorta_volume_cm3:.1f} cm3; "
                f"{aorta_ratio:.1f}x cohort median)"
            )
        report.qc_warning = "; ".join(warnings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/segthor_part1"),
        help="Source dataset root containing train/Patient_XX directories.",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path("data/segthor_part1_clean_cc"),
        help="New dataset root. Raw source data are never modified.",
    )
    parser.add_argument("--split-label", type=int, default=1)
    parser.add_argument("--esophagus-label", type=int, default=1)
    parser.add_argument("--aorta-label", type=int, default=4)
    parser.add_argument("--min-seed-size", type=int, default=500)
    parser.add_argument("--max-erosion", type=int, default=12)
    parser.add_argument(
        "--min-smaller-fraction",
        type=float,
        default=0.02,
        help="Allow genuinely small esophagus components; endpoint QC rejects short fragments.",
    )
    parser.add_argument("--max-smaller-fraction", type=float, default=0.45)
    parser.add_argument("--target-smaller-fraction", type=float, default=0.20)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing destination. No files are deleted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_train = args.source_root / "train"
    destination_train = args.dest_root / "train"

    if not source_train.is_dir():
        raise FileNotFoundError(f"Source train directory does not exist: {source_train}")
    if args.dest_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"Destination already exists: {args.dest_root}. "
            "Choose a new destination or pass --overwrite explicitly."
        )

    patient_dirs = sorted(source_train.glob("Patient_*"))
    if not patient_dirs:
        raise RuntimeError(f"No Patient_* directories found in {source_train}")

    reports: list[SplitReport] = []
    for index, patient_dir in enumerate(patient_dirs, start=1):
        print(f"[{index:02d}/{len(patient_dirs):02d}] Correcting {patient_dir.name}...", flush=True)
        report = correct_patient(
            patient_dir,
            destination_train / patient_dir.name,
            split_label=args.split_label,
            esophagus_label=args.esophagus_label,
            aorta_label=args.aorta_label,
            min_seed_size=args.min_seed_size,
            max_erosion=args.max_erosion,
            min_smaller_fraction=args.min_smaller_fraction,
            max_smaller_fraction=args.max_smaller_fraction,
            target_smaller_fraction=args.target_smaller_fraction,
        )
        reports.append(report)
        warning = f" WARNING: {report.qc_warning}" if report.qc_warning else ""
        print(
            f"    erosion={report.erosion_iterations}, "
            f"esophagus={report.esophagus_voxels:,}, aorta={report.aorta_voxels:,}, "
            f"endpoint-error={report.endpoint_error_slices} slices.{warning}",
            flush=True,
        )

    add_cohort_warnings(reports)
    report_path = args.dest_root / "split_report.csv"
    write_report(reports, report_path)
    warnings = [report.patient for report in reports if report.qc_warning]
    print(f"\nSaved corrected dataset to: {args.dest_root}")
    print(f"Saved QC report to: {report_path}")
    if warnings:
        print("Patients requiring visual review: " + ", ".join(warnings))
    else:
        print("No automatic QC warnings. Visual review is still recommended.")


if __name__ == "__main__":
    main()
