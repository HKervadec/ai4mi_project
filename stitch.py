#!/usr/bin/env python3

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import numpy as np
import nibabel as nib
from collections import defaultdict
from PIL import Image
from skimage.transform import resize

ACCEPTED_EXTENSIONS = {".png", ".npy"}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch 2D slices back into 3D volumes")
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Folder containing sliced data (e.g., data/prediction/best_epoch/val)")
    parser.add_argument("--dest_folder", type=str, required=True,
                        help="Destination folder for stitched predictions (e.g., val/pred)")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of classes (e.g., 5)")
    parser.add_argument("--grp_regex", type=str, required=True,
                        help="Pattern for filename grouping "
                             "(e.g., '(Patient\\d+)_\\d{4}' or '(Patient_\\d\\d)_\\d\\d\\d\\d')")
    parser.add_argument("--source_scan_pattern", type=str, required=True,
                        help="Pattern to reference scans for metadata (use {id_} placeholder).")
    parser.add_argument("--copy-gt", action="store_true",
                        help="If set, copy the original GT files into val/gt/")
    return parser.parse_args()


# --- file discovery & parsing -------------------------------------------------

def _walk_slice_files(root: Path) -> Iterable[Path]:
    """Yield all supported slice files under root (recursively), in a stable order."""
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in ACCEPTED_EXTENSIONS:
            yield p


def _parse_slice_index(stem: str) -> int:
    """Read the axial index from the trailing integer in the stem."""
    m = re.search(r"(\d+)$", stem)
    if not m:
        raise ValueError(f"Could not infer slice index from filename '{stem}'")
    return int(m.group(1))


def group_slices_by_patient(data_folder: Path, grp_regex: str) -> Dict[str, List[Tuple[int, Path]]]:
    """
    Group slice files by patient id using the provided regex.
    - Searches recursively
    - Accepts .png and .npy
    - Uses the trailing digits as the slice index
    """
    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    pattern = re.compile(grp_regex)
    buckets: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)

    found_any = False
    for f in _walk_slice_files(data_folder):
        found_any = True
        m = pattern.search(f.stem)
        if not m:
            raise ValueError(f"File '{f.name}' does not match grouping regex '{grp_regex}'")
        pid = m.group(1) if m.groups() else m.group(0)
        idx = _parse_slice_index(f.stem)
        buckets[pid].append((idx, f))

    if not found_any:
        exts = ", ".join(sorted(ACCEPTED_EXTENSIONS))
        raise ValueError(f"No slice files found in '{data_folder}' (accepted: {exts})")

    # sort by parsed index
    for pid in buckets:
        buckets[pid].sort(key=lambda t: t[0])

    return dict(buckets)


# --- label handling ------------------------------------------------------

def _load_slice_2d(path: Path) -> np.ndarray:
    """Load a 2D label map from .png or .npy; ensure uint8 and 2D."""
    if path.suffix.lower() == ".png":
        with Image.open(path) as img:
            arr = np.array(img)
    elif path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        raise ValueError(f"Unsupported slice extension '{path.suffix}'")

    if arr.ndim != 2:
        raise ValueError(f"Slice '{path.name}' is expected to be 2D, got shape {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    return arr


def _undo_png_scaling_gcd(labels: np.ndarray) -> np.ndarray:
    """
    If PNG scaling was applied (e.g., labels multiplied to fit into 8-bit),
    undo it using the GCD of positive unique values.
    """
    if labels.size == 0:
        return labels
    uniq = np.unique(labels)
    pos = uniq[uniq > 0]
    if pos.size == 0:
        return labels
    gcd_val = int(np.gcd.reduce(pos.astype(np.int64)))
    gcd_val = max(gcd_val, 1)
    return np.round(labels / gcd_val).astype(np.uint8)


def _resize_labels_nn(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    if arr.shape == target_hw:
        return arr
    out = resize(arr, target_hw, order=0, mode="edge", preserve_range=True, anti_aliasing=False)
    return np.round(out).astype(np.uint8)


def _load_ref_scan(pattern: str, pid: str) -> Tuple[Tuple[int, int, int], np.ndarray, nib.nifti1.Nifti1Header]:
    ref_path = Path(pattern.format(id_=pid))
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference scan not found for '{pid}': {ref_path}")
    nii = nib.load(str(ref_path))
    shape = nii.shape
    if len(shape) != 3:
        raise ValueError(f"Reference scan for '{pid}' must be 3D, got shape {shape}")
    return (int(shape[0]), int(shape[1]), int(shape[2])), nii.affine, nii.header


# --- validation ---------------------------------------------------------------

def _check_volume(volume: np.ndarray, ref_shape: Tuple[int, int, int], num_classes: int) -> None:
    if volume.shape != ref_shape:
        raise ValueError(f"Volume shape {volume.shape} != reference {ref_shape}")
    if volume.dtype != np.uint8:
        raise ValueError(f"Volume dtype {volume.dtype} != uint8")
    vmin = int(volume.min(initial=0))
    vmax = int(volume.max(initial=0))
    if vmin < 0 or vmax > 255:
        raise ValueError(f"Volume values out of [0,255]: min={vmin}, max={vmax}")
    if num_classes is not None and num_classes > 0 and vmax >= num_classes:
        raise ValueError(f"Found label {vmax} which exceeds allowed range [0, {num_classes-1}]")


def _sanity_check_written(path: Path, ref_shape: Tuple[int, int, int], num_classes: int) -> None:
    img = nib.load(str(path))
    if img.shape != ref_shape:
        raise ValueError(f"Written NIfTI '{path}' has shape {img.shape}, expected {ref_shape}")
    data = np.asanyarray(img.dataobj)
    if data.dtype != np.uint8:
        raise ValueError(f"Written NIfTI '{path}' dtype is {data.dtype}, expected uint8")
    vmin = int(data.min(initial=0))
    vmax = int(data.max(initial=0))
    if vmin < 0 or vmax > 255:
        raise ValueError(f"Written NIfTI '{path}' values out of [0,255]: min={vmin}, max={vmax}")
    if num_classes is not None and num_classes > 0 and vmax >= num_classes:
        raise ValueError(f"Written NIfTI '{path}' contains label {vmax} (>= num_classes={num_classes})")


# --- stitching & save ---------------------------------------------------------

def stitch_patient_slices(slice_files: List[Tuple[int, Path]],
                          num_classes: int,
                          original_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Build a (H, W, D) volume using parsed indices for placement.
    Enforces exact slice count match with reference depth.
    """
    H, W, D = original_shape

    if len(slice_files) != D:
        raise ValueError(f"Expected {D} slices, found {len(slice_files)}")

    vol = np.zeros((H, W, D), dtype=np.uint8)

    for z, fpath in slice_files:
        sl = _load_slice_2d(fpath)
        sl = _undo_png_scaling_gcd(sl)
        sl = _resize_labels_nn(sl, (H, W))

        if sl.max(initial=0) >= num_classes:
            raise ValueError(
                f"Slice '{fpath.name}' contains label {int(sl.max())} which exceeds num_classes={num_classes}"
            )

        if not (0 <= z < D):
            raise ValueError(f"Slice index {z} out of bounds for depth {D} (file '{fpath.name}')")

        vol[:, :, z] = sl

    return vol


def save_stitched_volume(volume: np.ndarray,
                         affine: np.ndarray,
                         header: nib.nifti1.Nifti1Header,
                         dest_path: Path,
                         patient_id: str,
                         ref_shape: Tuple[int, int, int],
                         num_classes: int) -> None:
    _check_volume(volume, ref_shape, num_classes)
    dest_path.mkdir(parents=True, exist_ok=True)
    out_path = dest_path / f"{patient_id}.nii.gz"
    nib.save(nib.Nifti1Image(volume.astype(np.uint8), affine=affine, header=header), str(out_path))
    _sanity_check_written(out_path, ref_shape, num_classes)
    print(f"Saved stitched volume: {out_path}")


def copy_gt_files(patient_ids: List[str], source_scan_pattern: str, dest_folder: Path):
    """Copy original GT files to the destination folder for evaluation (optional)."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    for pid in patient_ids:
        src = Path(source_scan_pattern.format(id_=pid))
        if not src.exists():
            print(f"Warning: GT not found for {pid} at {src}")
            continue
        dst = dest_folder / f"{pid}.nii.gz"
        shutil.copy(src, dst)
        print(f"Copied GT for {pid} to {dst}")


# --- main --------------------------------------------------------------------

def main():
    args = get_args()

    data_folder = Path(args.data_folder)
    dest_folder = Path(args.dest_folder)

    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    print(f"Stitching slices from: {data_folder}")
    print(f"Output destination: {dest_folder}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Grouping regex: {args.grp_regex}")
    print(f"Source scan pattern: {args.source_scan_pattern}")
    print(f"Copy GT enabled: {args.copy_gt}")

    patient_slices = group_slices_by_patient(data_folder, args.grp_regex)

    if not patient_slices:
        raise ValueError("No patient slices found. Check your data folder and regex pattern.")

    print(f"Found {len(patient_slices)} patients")

    for pid, files in patient_slices.items():
        print(f"\nProcessing {pid} with {len(files)} slices...")
        try:
            ref_shape, ref_affine, ref_header = _load_ref_scan(args.source_scan_pattern, pid)
        except FileNotFoundError as e:
            print(f"{e}. Skipping '{pid}'.")
            continue

        try:
            print(f"Reference shape: {ref_shape}")
            vol = stitch_patient_slices(files, args.num_classes, ref_shape)
            print(f"Stitched volume shape: {vol.shape}")
            save_stitched_volume(vol, ref_affine, ref_header, dest_folder, pid, ref_shape, args.num_classes)
        except Exception as e:
            print(f"Error processing {pid}: {e}")
            continue

    if args.copy_gt:
        gt_dest = dest_folder.parent / "gt"
        print(f"\nCopying GT files to {gt_dest}...")
        copy_gt_files(list(patient_slices.keys()), args.source_scan_pattern, gt_dest)

    print(f"\nDone. Results saved to: {dest_folder}")
    if args.copy_gt:
        print(f"GT files copied to: {gt_dest}")


if __name__ == "__main__":
    main()