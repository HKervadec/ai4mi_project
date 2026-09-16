#!/usr/bin/env python3

"""
Convert TotalSegmentator (small v2.0.1) into a new dataset with the same layout as data/segthor_part1:

    <dest_dir>/train/s0011/s0011.nii.gz   (the CT, originally ct.nii.gz)
    <dest_dir>/train/s0011/GT.nii.gz      (0 background, 1 esophagus, 2 heart, 3 trachea, 4 aorta)

Only subjects where all four thoracic organs are annotated are kept, each volume is cropped along the
body axis to the thorax, and the in-plane field of view is padded/cropped to a fixed physical size so
that the slices end up at the same mm/pixel as the SegTHOR ones. The source dataset is only read,
never modified.
"""

import os
import time
import shutil
import argparse
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable

import numpy as np
import nibabel as nib
from tqdm import tqdm
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

# Not imported from utils.py, to avoid pulling torch into a pure data preparation script
tqdm_ = partial(tqdm, dynamic_ncols=True, leave=True)


# SegTHOR label convention
LABELS: dict[str, int] = {"esophagus": 1, "heart": 2, "trachea": 3, "aorta": 4}
PAINT_ORDER: list[str] = ["heart", "esophagus", "trachea", "aorta"]
assert set(PAINT_ORDER) == set(LABELS)
# SegTHOR volumes are stored as LPS, TotalSegmentator as RAS. Without reorienting, the 2D slices
# would end up rotated by 180 degrees compared to the SegTHOR ones.
TARGET_AXCODES: tuple[str, str, str] = ("L", "P", "S")

# Since TotalSegmentator contains slices along the lower body, we crop around the heart since we focus on thoracic organs.
ABOVE_HEART_MM: float = 220
BELOW_HEART_MM: float = 95

# There were some outliers with abnormally large aortas. Upon inspection this was due to very, very generous aorta annotations which
# majorly inflated the size. Hence we drop subject with aorta volumes above threshold.
MAX_AORTA_CM3: float = 600

# Since TotalSegmentor has many outliers with very low HU values, we clip the CTs to a minimum value. This is due to these low hu values
# occuring mainly in voxels which are not annotated as one of the thoracic organs.
CLIP_HU_MIN: int = -1000

# slice_segthor.py resizes every slice to 256x256, so a volume with a smaller field of view ends up with a
# smaller mm/pixel: the same organ would cover more pixels than in SegTHOR. The majority of SegTHOR patients were
# scanned with a 500 mm field of view (1.95 mm/pixel at 256), so we pad/crop TotalSegmentator to match.
TARGET_FOV_MM: float = 500


def usable_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def rename_with_retry(src: Path, dest: Path, attempts: int = 10, delay: float = 0.2) -> None:
    for attempt in range(attempts):
        try:
            src.rename(dest)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay * (attempt + 1))


def reorient(img: nib.Nifti1Image) -> nib.Nifti1Image:
    # Helper function for pure axis flips without resampling
    transform = ornt_transform(io_orientation(img.affine), axcodes2ornt(TARGET_AXCODES))
    return img.as_reoriented(transform)


def crop_to_thorax(ct: nib.Nifti1Image, gt: nib.Nifti1Image,
                   above_mm: float, below_mm: float) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    # Both are already reoriented, so the last axis runs from the feet to the head
    z_heart = np.flatnonzero((np.asarray(gt.dataobj) == LABELS["heart"]).any(axis=(0, 1)))
    dz: float = float(gt.header.get_zooms()[2])
    z_start: int = max(0, int(z_heart.min() - round(below_mm / dz)))
    z_stop: int = min(gt.shape[2], int(z_heart.max() + 1 + round(above_mm / dz)))

    return ct.slicer[:, :, z_start:z_stop], gt.slicer[:, :, z_start:z_stop]


def set_fov(img: nib.Nifti1Image, fov_mm: float, pad_value: int) -> nib.Nifti1Image:
    # Centered pad (or crop, if the scan is wider than fov_mm) of the two in-plane axes, leaving the
    # voxel size untouched: only the number of voxels changes, so nothing is resampled.
    data: np.ndarray = np.asarray(img.dataobj)
    zooms: tuple[float, ...] = img.header.get_zooms()

    shape: list[int] = list(data.shape)
    src: list[slice] = [slice(None)] * 3
    dest: list[slice] = [slice(None)] * 3
    offsets: list[int] = [0, 0, 0]

    for axis in (0, 1):
        n: int = data.shape[axis]
        n_target: int = int(round(fov_mm / zooms[axis]))
        offset: int = (n - n_target) // 2

        start: int = max(offset, 0)
        stop: int = min(offset + n_target, n)
        src[axis] = slice(start, stop)
        dest[axis] = slice(start - offset, start - offset + stop - start)
        offsets[axis] = offset
        shape[axis] = n_target

    out: np.ndarray = np.full(shape, pad_value, dtype=data.dtype)
    out[tuple(dest)] = data[tuple(src)]

    affine: np.ndarray = img.affine.copy()
    affine[:3, 3] += img.affine[:3, :3] @ np.asarray(offsets, dtype=float)
    
    new_img = nib.Nifti1Image(out, affine, header=img.header)
    new_img.set_data_dtype(data.dtype)

    return new_img


def convert_subject(subject_dir: Path, dest_path: Path, above_mm: float, below_mm: float,
                    max_aorta_cm3: float, clip_hu_min: int, fov_mm: float) -> tuple[str, str]:
    id_: str = subject_dir.name
    dest_subject: Path = dest_path / id_
    if dest_subject.exists():
        return id_, "exists"

    ct_nib = nib.load(subject_dir / "ct.nii.gz")

    masks: dict[str, np.ndarray] = {}
    for organ in LABELS:
        mask_nib = nib.load(subject_dir / "segmentations" / f"{organ}.nii.gz")
        assert mask_nib.shape == ct_nib.shape, (id_, organ, mask_nib.shape, ct_nib.shape)
        # Some affines differ by float rounding noise only (e.g. ~3e-4 for s1389)
        assert np.allclose(mask_nib.affine, ct_nib.affine, atol=1e-3), (id_, organ)
        masks[organ] = np.asarray(mask_nib.dataobj) > 0

    if not all(mask.any() for mask in masks.values()):
        return id_, "incomplete"

    gt = np.zeros(ct_nib.shape, dtype=np.uint8)
    for organ in PAINT_ORDER:
        gt[masks[organ]] = LABELS[organ]
    assert set(np.unique(gt)) == set(range(5)), (id_, np.unique(gt))

    gt_nib = nib.Nifti1Image(gt, ct_nib.affine, header=ct_nib.header)
    gt_nib.set_data_dtype(np.uint8)
    gt_nib.header.set_slope_inter(1, 0)

    ct_out = reorient(ct_nib)
    gt_out = reorient(gt_nib)
    assert ct_out.shape == gt_out.shape
    assert nib.aff2axcodes(ct_out.affine) == TARGET_AXCODES

    ct_out, gt_out = crop_to_thorax(ct_out, gt_out, above_mm, below_mm)
    if fov_mm:
        # The CT is padded with air, so the padding survives the clip below unchanged
        ct_out = set_fov(ct_out, fov_mm, clip_hu_min)
        gt_out = set_fov(gt_out, fov_mm, 0)
        assert ct_out.shape == gt_out.shape

    cropped_gt: np.ndarray = np.asarray(gt_out.dataobj)
    # A crop can cut away an organ entirely (e.g. an esophagus annotated far down into the abdomen)
    if set(np.unique(cropped_gt)) != set(range(len(LABELS) + 1)):
        return id_, "organ lost in crop"

    aorta_cm3: float = (cropped_gt == LABELS["aorta"]).sum() * np.prod(gt_out.header.get_zooms()) / 1000
    if aorta_cm3 > max_aorta_cm3:
        return id_, f"aorta too large ({aorta_cm3:.0f} cm3)"

    clipped_ct: np.ndarray = np.clip(np.asarray(ct_out.dataobj), clip_hu_min, None).astype(np.int16)
    ct_out = nib.Nifti1Image(clipped_ct, ct_out.affine, header=ct_out.header)
    ct_out.set_data_dtype(np.int16)

    # Write to a temporary folder first, so an interrupted run never leaves a half-written subject behind
    tmp_subject: Path = dest_path / f"{id_}_tmp"
    if tmp_subject.exists():
        shutil.rmtree(tmp_subject)
    tmp_subject.mkdir(parents=True)
    nib.save(ct_out, tmp_subject / f"{id_}.nii.gz")
    nib.save(gt_out, tmp_subject / "GT.nii.gz")
    rename_with_retry(tmp_subject, dest_subject)

    return id_, "converted"


def main(args: argparse.Namespace) -> None:
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir) / "train"
    assert src_path.exists(), src_path
    # Everything is written to dest_dir, which must not end up inside the source dataset
    assert not dest_path.resolve().is_relative_to(src_path.resolve()), (src_path, dest_path)

    subjects: list[Path] = sorted(p for p in src_path.glob("s[0-9]*") if p.is_dir())
    print(f"Found {len(subjects)} subjects in {src_path}")
    assert subjects

    dest_path.mkdir(parents=True, exist_ok=True)

    pfun: Callable = partial(convert_subject, dest_path=dest_path,
                             above_mm=args.above_heart_mm, below_mm=args.below_heart_mm,
                             max_aorta_cm3=args.max_aorta_cm3, clip_hu_min=args.clip_hu_min,
                             fov_mm=args.target_fov_mm)
    results: list[tuple[str, str]]
    match args.process:
        case 1:
            results = list(map(pfun, tqdm_(subjects)))
        case _ as p:
            processes: int = usable_cpus() if p == -1 else p
            print(f"Converting with {processes} processes")
            with Pool(processes) as pool:
                results = list(tqdm_(pool.imap_unordered(pfun, subjects), total=len(subjects)))

    for status in ["converted", "exists", "incomplete", "organ lost in crop"]:
        ids: list[str] = [id_ for id_, s in results if s == status]
        print(f"{status:>20}: {len(ids)}" + (f" {ids}" if status != "converted" and ids else ""))
    rejected: list[tuple[str, str]] = [(id_, s) for id_, s in results if s.startswith("aorta too large")]
    print(f"{'aorta too large':>20}: {len(rejected)}" + (f" {rejected}" if rejected else ""))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TotalSegmentator to the segthor_part1 layout")
    parser.add_argument('--source_dir', type=str, default="data/Totalsegmentator_dataset_small_v201",
                        help="The original TotalSegmentator dataset (read only)")
    parser.add_argument('--dest_dir', type=str, default="data/totalseg_part1",
                        help="The new dataset, subjects are saved in <dest_dir>/train/")
    parser.add_argument('--above_heart_mm', type=float, default=ABOVE_HEART_MM,
                        help="How much to keep above the top of the heart")
    parser.add_argument('--below_heart_mm', type=float, default=BELOW_HEART_MM,
                        help="How much to keep below the bottom of the heart")
    parser.add_argument('--max_aorta_cm3', type=float, default=MAX_AORTA_CM3,
                        help="Drop subjects whose cropped aorta is larger than this")
    parser.add_argument('--clip_hu_min', type=int, default=CLIP_HU_MIN,
                        help="Clip CT intensities below this value (SegTHOR is clipped at -1000 HU)")
    parser.add_argument('--target_fov_mm', type=float, default=TARGET_FOV_MM,
                        help="Pad/crop the in-plane field of view to this many mm (0 to keep it as is)")
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="The number of cores to use for processing (-1 for all)")
    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
