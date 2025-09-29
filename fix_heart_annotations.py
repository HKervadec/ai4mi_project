#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
from tqdm import tqdm
import time

HEART = 2  # heart label id

# -------------------- transformation -------------------- #
def create_forward_transform_neg_offset() -> np.ndarray:
    """
    Build and return the forward transformation.
    """
    # T1: translation
    T1 = np.array([
        [1, 0, 0, 275],
        [0, 1, 0, 200],
        [0, 0, 1,   0],
        [0, 0, 0,   1]
    ], dtype=np.float64)

    # R2: rotation (change from - to to + since we invert the transform later)
    phi = (27.0 / 180.0) * np.pi
    c, s = np.cos(phi), np.sin(phi)
    R2 = np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ], dtype=np.float64)

    # T3 = inverse of T1
    T3 = np.linalg.inv(T1)

    # T4: final translation
    T4 = np.array([
        [1, 0, 0, 50],
        [0, 1, 0, 40],
        [0, 0, 1, 15],
        [0, 0, 0,  1]
    ], dtype=np.float64)

    # combine transforms: T1 -> R2 -> T3 -> T4
    forward = T4 @ T3 @ R2 @ T1
    return forward


# -------------------- dice metric -------------------- #
def dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    eps = 1e-8
    inter = np.count_nonzero(mask_a & mask_b)
    total = mask_a.sum() + mask_b.sum() + eps
    return ((2.0 * inter + eps) / total )


# -------------------- fix function -------------------- #
def fix_tampered_annotation(gt_data: np.ndarray, transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the transform to fix the heart annotation.
    Returns:
        fixed volume and the restored heart mask.
    """
    heart_mask = (gt_data == HEART).astype(np.float32)
    matrix = transform[:3, :3]
    offset = transform[:3, 3]

    restored = affine_transform(
        heart_mask,
        matrix=matrix,
        offset=-offset,
        output_shape=heart_mask.shape,
        order=0, # NN
        prefilter=False
    )

    restored_bin = restored > 0.5
    fixed = gt_data.copy()
    fixed[gt_data == HEART] = 0
    fixed[restored_bin] = HEART

    return fixed, restored_bin


# -------------------- dice (compared to clean GT) -------------------- #
def compute_dice_against_clean(patient_folder: Path, restored_mask: np.ndarray) -> None:
    """
    If there's a clean GT in this folder, compute and print the Dice score.
    """
    clean_gt_candidates = [p for p in patient_folder.glob("GT*.nii*")
                           if p.name not in {"GT.nii.gz", "GT_fixed.nii.gz"}]
    if not clean_gt_candidates:
        return
    else:
        print(f"  - {patient_folder.name}: Found clean reference GT: {clean_gt_candidates[0].name}")
    clean_gt_path = clean_gt_candidates[0]
    clean_data = nib.load(str(clean_gt_path)).get_fdata().astype(np.int16)
    clean_heart = (clean_data == HEART)

    d = dice_score(restored_mask, clean_heart)
    print(f"  - {patient_folder.name}: Dice score (heart vs reference) = {d:.4f} "
          f"[restored voxels={int(restored_mask.sum())}, ref voxels={int(clean_heart.sum())}]")


# -------------------- processing -------------------- #
def process_patient_folder(patient_folder: Path, transform: np.ndarray) -> None:
    gt_path = patient_folder / "GT.nii.gz"
    output_path = patient_folder / "GT_fixed.nii.gz"

    if not gt_path.exists():
        print(f"  - Skipping (no GT.nii.gz): {patient_folder.name}")
        return

    try:
        gt_nii = nib.load(str(gt_path))
        gt_data = gt_nii.get_fdata().astype(np.int16)

        fixed_data, restored_mask = fix_tampered_annotation(gt_data, transform)

        # save the fixed GT
        out_img = nib.Nifti1Image(fixed_data.astype(np.uint8), gt_nii.affine, gt_nii.header)
        out_img.set_data_dtype(np.uint8)
        nib.save(out_img, str(output_path))

        # compute dice if a clean GT exists
        compute_dice_against_clean(patient_folder, restored_mask)

    except Exception as e:
        print(f"  ! Error processing {patient_folder.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fix tampered heart annotations using fixed forward transform with negated offset (fwd/noswap/negOffset).")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="data/segthor_train/train",
        help="Path to the source directory containing patient folders"
    )
    args = parser.parse_args()
    # track time
    start_time = time.time()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return

    transform = create_forward_transform_neg_offset()

    # process all Patient_* folders
    patient_folders = sorted(
        d for d in source_dir.iterdir()
        if d.is_dir() and d.name.startswith("Patient_")
    )

    if not patient_folders:
        print("No patient folders found!")
        return

    print("*"*50)
    print(f"Found {len(patient_folders)} patient folders under {source_dir}")
    for patient_folder in tqdm(patient_folders, desc="Processing patients"):
        process_patient_folder(patient_folder, transform)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[INFO] All patients processed. Total processing time: {elapsed_time:.2f} seconds")
    print("*"*50)


if __name__ == "__main__":
    main()
