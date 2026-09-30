import argparse
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    total = np.count_nonzero(pred) + np.count_nonzero(gt)
    if total == 0:
        return float("nan")
    return 2 * np.count_nonzero(pred & gt) / total

def hausdorff(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, ...]) -> float:
    if not pred.any() and not gt.any():
        return float("nan")
    if not pred.any() or not gt.any():
        return float("inf")
    pred_edge = pred & ~binary_erosion(pred)
    gt_edge = gt & ~binary_erosion(gt)
    pred_to_gt = distance_transform_edt(~gt_edge, sampling=spacing)[pred_edge].max()
    gt_to_pred = distance_transform_edt(~pred_edge, sampling=spacing)[gt_edge].max()
    return float(max(pred_to_gt, gt_to_pred))

def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.ndim != 2 or np.any(mask > 252) or np.any(mask % 63):
        raise ValueError(f"error with {path}")
    return mask // 63

def main() -> None:
    parser = argparse.ArgumentParser(description="evaluate segthor predictions")
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    gt_files = {path.name: path for path in args.gt_dir.glob("*.png")}
    pred_files = {path.name: path for path in args.pred_dir.glob("*.png")}
    if not gt_files or gt_files.keys() != pred_files.keys():
        raise ValueError("matching error")

    with open(args.gt_dir.parent.parent / "spacing.pkl", "rb") as file:
        spacings = pickle.load(file)

    patients = defaultdict(list)
    for name in gt_files:
        patient, _, index = Path(name).stem.rpartition("_")
        patients[patient].append((int(index), name))

    dice_2d, hausdorff_2d, dice_3d, hausdorff_3d = {}, {}, {}, {}
    for patient, slices in sorted(patients.items()):
        slices.sort()
        if [index for index, _ in slices] != list(range(len(slices))):
            raise ValueError(f"error with {patient}")

        gt_slices = [read_mask(gt_files[name]) for _, name in slices]
        pred_slices = [read_mask(pred_files[name]) for _, name in slices]
        if any(gt.shape != pred.shape for gt, pred in zip(gt_slices, pred_slices)):
            raise ValueError(f"mismatch {patient}")

        shape = gt_slices[0].shape
        if any(gt.shape != shape for gt in gt_slices):
            raise ValueError(f"error {patient}")

        dx, dy, dz = spacings[patient]
        spacing_2d = (dx * 512 / shape[0], dy * 512 / shape[1])
        spacing_3d = (*spacing_2d, dz)

        d2 = np.full((5, len(slices)), np.nan)
        h2 = np.full((5, len(slices)), np.nan)
        for z, (pred, gt) in enumerate(zip(pred_slices, gt_slices)):
            for cls in range(1, 5):
                pred_mask, gt_mask = pred == cls, gt == cls
                d2[cls, z] = dice(pred_mask, gt_mask)
                h2[cls, z] = hausdorff(pred_mask, gt_mask, spacing_2d)

        pred_volume = np.stack(pred_slices, axis=-1)
        gt_volume = np.stack(gt_slices, axis=-1)
        d3, h3 = np.full(5, np.nan), np.full(5, np.nan)
        for cls in range(1, 5):
            pred_mask, gt_mask = pred_volume == cls, gt_volume == cls
            d3[cls] = dice(pred_mask, gt_mask)
            h3[cls] = hausdorff(pred_mask, gt_mask, spacing_3d)

        dice_2d[patient], hausdorff_2d[patient] = d2, h2
        dice_3d[patient], hausdorff_3d[patient] = d3, h3
        print(f"{patient}: 3D Dice {d3[1:]}; 3D Hausdorff (mm) {h3[1:]}")

    args.dest.mkdir(parents=True, exist_ok=True)
    np.savez(args.dest / "dice_2d.npz", **dice_2d)
    np.savez(args.dest / "hausdorff_2d.npz", **hausdorff_2d)
    np.savez(args.dest / "dice_3d.npz", **dice_3d)
    np.savez(args.dest / "hausdorff_3d.npz", **hausdorff_3d)


if __name__ == "__main__":
    main()