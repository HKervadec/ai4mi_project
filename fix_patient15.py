#!/usr/bin/env python3
"""Repair only Patient 15 in the corrected SegTHOR dataset.

The generic 3-D erosion method fails for Patient 15 because short fragments at
the superior and inferior ends disconnect before the full esophagus separates
from the aorta.  In this patient, most axial slices already contain a small,
separate esophagus component.  This repair tracks those components between
adjacent slices and uses them as dense markers for a 3-D watershed.  Watershed
is therefore needed only on the slices where the annotations touch.

The original dataset is read-only.  Only
``segthor_part1_clean_cc/train/Patient_15/GT.nii.gz`` is replaced, and its
previous version is backed up in ``results`` first.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.segmentation import watershed


def axial_span(mask: np.ndarray) -> int:
    occupied = np.where(mask.any(axis=(0, 1)))[0]
    if not occupied.size:
        return 0
    return int(occupied[-1] - occupied[0] + 1)


def make_patient15_split(
    merged_mask: np.ndarray,
    *,
    max_seed_area: int = 700,
    max_centroid_step: float = 18.0,
) -> tuple[np.ndarray, int]:
    """Split Patient 15 using tracked axial components as watershed seeds."""

    markers = np.zeros(merged_mask.shape, dtype=np.uint8)
    previous_centroid: np.ndarray | None = None
    seeded_slices: list[int] = []

    # Start at the superior end, where only the esophagus is present, and
    # follow its small 2-D component inferiorly.  Large or implausibly distant
    # components are deliberately left unseeded for the 3-D watershed.
    for z in range(merged_mask.shape[2] - 1, -1, -1):
        components = label(merged_mask[:, :, z], connectivity=1)
        regions = regionprops(components)
        if not regions:
            continue

        candidates = [region for region in regions if region.area <= max_seed_area]
        if not candidates:
            continue

        if previous_centroid is None:
            chosen = min(candidates, key=lambda region: region.area)
            step = 0.0
        else:
            chosen = min(
                candidates,
                key=lambda region: float(
                    np.sum((np.asarray(region.centroid) - previous_centroid) ** 2)
                ),
            )
            step = float(
                np.linalg.norm(np.asarray(chosen.centroid) - previous_centroid)
            )

        if previous_centroid is not None and step >= max_centroid_step:
            continue

        esophagus_seed = components == chosen.label
        markers[:, :, z][esophagus_seed] = 1
        markers[:, :, z][merged_mask[:, :, z] & ~esophagus_seed] = 2
        previous_centroid = np.asarray(chosen.centroid)
        seeded_slices.append(z)

    if not seeded_slices:
        raise RuntimeError("No axial esophagus seeds were found for Patient 15")

    # Everything below the first observed esophagus slice is unambiguous aorta.
    for z in range(min(seeded_slices)):
        markers[:, :, z][merged_mask[:, :, z]] = 2

    if not np.any(markers == 1) or not np.any(markers == 2):
        raise RuntimeError("Both esophagus and aorta seeds are required")

    distance = ndimage.distance_transform_edt(merged_mask)
    split = watershed(-distance, markers, mask=merged_mask).astype(np.uint8)

    if not np.array_equal(merged_mask, (split == 1) | (split == 2)):
        raise AssertionError("Watershed output does not reproduce the merged mask")

    # Patient 15's esophagus must be one continuous 3-D structure and must be
    # present without gaps between its first and last annotated axial slices.
    esophagus = split == 1
    if int(label(esophagus, connectivity=1).max()) != 1:
        raise AssertionError("The repaired esophagus is not one 3-D component")
    occupied_z = np.where(esophagus.any(axis=(0, 1)))[0]
    expected_z = np.arange(occupied_z[0], occupied_z[-1] + 1)
    if not np.array_equal(occupied_z, expected_z):
        raise AssertionError("The repaired esophagus has missing axial slices")

    return split, len(seeded_slices)


def update_qc_report(
    report_path: Path,
    *,
    patient: str,
    original: np.ndarray,
    corrected: np.ndarray,
    voxel_volume_cm3: float,
    seed_count: int,
) -> None:
    """Replace only Patient 15's stale statistics in the existing report."""

    if not report_path.exists():
        return

    with report_path.open(newline="", encoding="utf-8") as report_file:
        reader = csv.DictReader(report_file)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"QC report has no header: {report_path}")

    merged = original == 1
    esophagus = corrected == 1
    aorta = corrected == 4
    union_z = np.where(merged.any(axis=(0, 1)))[0]
    aorta_z = np.where(aorta.any(axis=(0, 1)))[0]
    esophagus_z = np.where(esophagus.any(axis=(0, 1)))[0]
    endpoint_error = int(
        (aorta_z[0] - union_z[0]) + (union_z[-1] - esophagus_z[-1])
    )

    replacement = {
        "patient": patient,
        "erosion_iterations": "0",
        "candidate_count": str(seed_count),
        "merged_voxels": str(int(merged.sum())),
        "esophagus_voxels": str(int(esophagus.sum())),
        "aorta_voxels": str(int(aorta.sum())),
        "esophagus_volume_cm3": str(float(esophagus.sum()) * voxel_volume_cm3),
        "aorta_volume_cm3": str(float(aorta.sum()) * voxel_volume_cm3),
        "esophagus_fraction": str(float(esophagus.sum() / merged.sum())),
        "esophagus_z_span": str(axial_span(esophagus)),
        "merged_z_span": str(axial_span(merged)),
        "z_coverage": str(axial_span(esophagus) / axial_span(merged)),
        "endpoint_error_slices": str(endpoint_error),
        "endpoint_error_fraction": str(endpoint_error / axial_span(merged)),
        "reference_difference_voxels": "",
        "esophagus_reference_dice": "",
        "aorta_reference_dice": "",
        "qc_warning": "",
    }

    matches = 0
    for row in rows:
        if row.get("patient") == patient:
            row.update({key: value for key, value in replacement.items() if key in row})
            matches += 1
    if matches != 1:
        raise ValueError(f"Expected one {patient} row in {report_path}, found {matches}")

    temporary_report = report_path.with_suffix(".tmp.csv")
    with temporary_report.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_report, report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/segthor_part1"))
    parser.add_argument(
        "--dest-root", type=Path, default=Path("data/segthor_part1_clean_cc")
    )
    parser.add_argument("--patient", default="Patient_15")
    parser.add_argument("--max-seed-area", type=int, default=700)
    parser.add_argument("--max-centroid-step", type=float, default=18.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.patient != "Patient_15":
        raise ValueError("This targeted repair is validated only for Patient_15")

    source_path = args.source_root / "train" / args.patient / "GT.nii.gz"
    destination_path = args.dest_root / "train" / args.patient / "GT.nii.gz"
    if not source_path.is_file() or not destination_path.is_file():
        raise FileNotFoundError("Patient 15 source or corrected GT is missing")

    source_image = nib.load(str(source_path))
    original = np.asanyarray(source_image.dataobj).astype(np.uint8)
    previous_image = nib.load(str(destination_path))
    previous = np.asanyarray(previous_image.dataobj).astype(np.uint8)

    if original.shape != previous.shape or not np.array_equal(
        source_image.affine, previous_image.affine
    ):
        raise ValueError("Source and corrected Patient 15 geometries differ")

    split, seed_count = make_patient15_split(
        original == 1,
        max_seed_area=args.max_seed_area,
        max_centroid_step=args.max_centroid_step,
    )
    corrected = original.copy()
    corrected[original == 1] = 0
    corrected[split == 1] = 1
    corrected[split == 2] = 4

    if not np.array_equal(original == 1, (corrected == 1) | (corrected == 4)):
        raise AssertionError("The repaired organ union differs from the source")
    for class_id in (0, 2, 3):
        if not np.array_equal(original == class_id, corrected == class_id):
            raise AssertionError(f"Non-target class {class_id} changed")
    if set(np.unique(corrected).astype(int)) != {0, 1, 2, 3, 4}:
        raise AssertionError("The repaired GT does not contain exactly five classes")

    backup_path = Path("results") / "Patient_15_GT_before_targeted_fix.nii.gz"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(destination_path, backup_path)

    header = source_image.header.copy()
    header.set_data_dtype(np.uint8)
    output_image = nib.Nifti1Image(corrected, source_image.affine, header)
    temporary_path = destination_path.with_name("GT.patient15-fix.tmp.nii.gz")
    nib.save(output_image, str(temporary_path))
    os.replace(temporary_path, destination_path)

    # Reload the file so validation also covers NIfTI serialization.
    saved_image = nib.load(str(destination_path))
    saved = np.asanyarray(saved_image.dataobj).astype(np.uint8)
    if not np.array_equal(saved, corrected):
        raise AssertionError("Saved Patient 15 GT differs from the validated array")
    if not np.array_equal(saved_image.affine, source_image.affine):
        raise AssertionError("Patient 15 affine changed during saving")

    voxel_volume_cm3 = float(np.prod(source_image.header.get_zooms()[:3]) / 1000.0)
    update_qc_report(
        args.dest_root / "split_report.csv",
        patient=args.patient,
        original=original,
        corrected=corrected,
        voxel_volume_cm3=voxel_volume_cm3,
        seed_count=seed_count,
    )

    changed_voxels = int((previous != corrected).sum())
    print(f"Repaired: {destination_path}")
    print(f"Backup:   {backup_path}")
    print(f"Tracked axial seed slices: {seed_count}")
    print(f"Changed class assignments: {changed_voxels:,} voxels")
    print(
        f"Esophagus: {int((corrected == 1).sum()):,} voxels "
        f"({float((corrected == 1).sum()) * voxel_volume_cm3:.1f} cm3)"
    )
    print(
        f"Aorta:     {int((corrected == 4).sum()):,} voxels "
        f"({float((corrected == 4).sum()) * voxel_volume_cm3:.1f} cm3)"
    )


if __name__ == "__main__":
    main()
