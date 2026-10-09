#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import label as connected_components
from torch.utils.data import DataLoader, Subset

from dataset import SliceDataset
from ENet import ENet
from utils import class2one_hot, probs2class

ORGAN_NAMES: dict[int, str] = {
    1: "Esophagus",
    2: "Heart",
    3: "Trachea",
    4: "Aorta",
}


def img_transform(img: Image.Image) -> torch.Tensor:
    img = img.convert("L")
    arr = np.array(img)[np.newaxis, ...] / 255.0
    return torch.tensor(arr, dtype=torch.float32)


def gt_transform(K: int, img: Image.Image) -> torch.Tensor:
    img = img.convert("L")
    arr = np.array(img)[...]
    arr = arr / (255 / (K - 1)) if K != 5 else arr / 63
    tensor = torch.tensor(arr, dtype=torch.int64)[None, ...]
    return class2one_hot(tensor, K=K)[0]


def find_empty_slice_indices(val_set: SliceDataset) -> list[int]:
    """Return dataset indices of validation slices whose GT mask is completely empty."""
    empty_indices: list[int] = []
    for idx, (_, gt_path) in enumerate(val_set.files):
        if gt_path is None:
            raise ValueError(f"Missing ground-truth path at index {idx}")
        with Image.open(gt_path) as gt_img:
            gt_arr = np.asarray(gt_img.convert("L"))
            if np.all(gt_arr == 0):
                empty_indices.append(idx)
    return empty_indices


def load_enet_model(
    weights_path: Path,
    device: torch.device,
    num_classes: int = 5,
    kernels: int = 8,
    factor: int = 2,
) -> torch.nn.Module:
    """Load an ENet model from a state_dict (.pt) or pickled model (.pkl)."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {weights_path}")

    if weights_path.suffix == ".pkl":
        net = torch.load(weights_path, map_location=device, weights_only=False)
    else:
        net = ENet(1, num_classes, kernels=kernels, factor=factor)
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        net.load_state_dict(state_dict)

    net.to(device)
    net.eval()
    return net


def run_empty_slices_check(args: argparse.Namespace) -> None:
    gpu = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda" if gpu else "cpu")
    print(f">> Using device: {device}")

    root_dir = args.data_root / args.dataset
    val_set = SliceDataset(
        "val",
        root_dir,
        img_transform=img_transform,
        gt_transform=lambda img: gt_transform(args.num_classes, img),
    )

    empty_indices = find_empty_slice_indices(val_set)
    total_val = len(val_set)
    num_empty = len(empty_indices)
    print(f">> Empty validation slices: {num_empty} / {total_val} ({100.0 * num_empty / max(total_val, 1):.1f}%)")

    if num_empty == 0:
        print(">> No completely empty validation slices found in this dataset.")
        return

    empty_subset = Subset(val_set, empty_indices)
    empty_loader = DataLoader(
        empty_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f">> Loading ENet weights from: {args.weights}")
    net = load_enet_model(
        args.weights,
        device=device,
        num_classes=args.num_classes,
        kernels=args.kernels,
        factor=args.factor,
    )

    blank_slices = 0
    hallucinated_slices = 0
    fg_pixels_per_slice: list[int] = []
    blobs_per_slice: list[int] = []
    hallucinated_details: list[tuple[str, int, int, dict[int, int]]] = []

    class_slice_counts: dict[int, int] = {k: 0 for k in range(1, args.num_classes)}
    class_pixel_counts: dict[int, int] = {k: 0 for k in range(1, args.num_classes)}

    with torch.no_grad():
        for batch in empty_loader:
            imgs = batch["images"].to(device)
            gts = batch["gts"].to(device)
            stems = batch["stems"]

            # Sanity checks matching main.py
            assert 0 <= imgs.min() and imgs.max() <= 1
            assert torch.all(gts[:, 1:, ...] == 0), "Expected completely empty ground-truth masks"

            logits = net(imgs)
            probs = F.softmax(logits, dim=1)
            pred_classes = probs2class(probs).cpu().numpy()

            for stem, pred_mask in zip(stems, pred_classes):
                fg_mask = pred_mask != 0
                fg_pixels = int(fg_mask.sum())
                fg_pixels_per_slice.append(fg_pixels)

                if fg_pixels == 0:
                    blank_slices += 1
                    blobs_per_slice.append(0)
                else:
                    hallucinated_slices += 1
                    _, num_blobs = connected_components(fg_mask)
                    blobs_per_slice.append(int(num_blobs))

                    per_class_fg: dict[int, int] = {}
                    for k in range(1, args.num_classes):
                        c_pixels = int((pred_mask == k).sum())
                        if c_pixels > 0:
                            class_slice_counts[k] += 1
                            class_pixel_counts[k] += c_pixels
                            per_class_fg[k] = c_pixels

                    hallucinated_details.append((stem, fg_pixels, int(num_blobs), per_class_fg))

    fg_arr = np.asarray(fg_pixels_per_slice)
    hallucinated_fg_arr = fg_arr[fg_arr > 0]
    blobs_arr = np.asarray(blobs_per_slice)
    hallucinated_blobs_arr = blobs_arr[fg_arr > 0]

    print("\n=== Empty Validation Slices Inference Summary ===")
    print(f"Model checkpoint      : {args.weights}")
    print(f"Dataset               : {args.dataset} ({root_dir / 'val'})")
    print(f"Empty GT slices       : {num_empty} / {total_val}")
    print(f"Entirely blank preds  : {blank_slices} / {num_empty} ({100.0 * blank_slices / num_empty:.2f}%)")
    print(f"Hallucinated slices   : {hallucinated_slices} / {num_empty} ({100.0 * hallucinated_slices / num_empty:.2f}%)")
    print(f"Mean FG pixels (all)  : {fg_arr.mean():.2f} px")
    print(f"Max FG pixels         : {fg_arr.max(initial=0):,} px")

    if hallucinated_slices > 0:
        print(
            f"FG pixels (halluc.)   : min={hallucinated_fg_arr.min()}, "
            f"median={np.median(hallucinated_fg_arr):.1f}, "
            f"mean={hallucinated_fg_arr.mean():.1f}, "
            f"max={hallucinated_fg_arr.max()}"
        )
        print(
            f"Blobs per halluc. slc : min={hallucinated_blobs_arr.min()}, "
            f"median={np.median(hallucinated_blobs_arr):.1f}, "
            f"mean={hallucinated_blobs_arr.mean():.2f}, "
            f"max={hallucinated_blobs_arr.max()}"
        )

        print("\n--- Breakdown by Foreground Class ---")
        for k in range(1, args.num_classes):
            organ = ORGAN_NAMES.get(k, f"Class {k}")
            s_cnt = class_slice_counts[k]
            p_cnt = class_pixel_counts[k]
            mean_when_present = p_cnt / s_cnt if s_cnt > 0 else 0.0
            print(
                f"  Class {k} ({organ:<9}): "
                f"{s_cnt:3d} / {num_empty} slices ({100.0 * s_cnt / num_empty:5.2f}%), "
                f"total={p_cnt:6,d} px, mean(when >0)={mean_when_present:6.1f} px"
            )

        hallucinated_details.sort(key=lambda x: x[1], reverse=True)
        top_n = min(args.show_top, len(hallucinated_details))
        print(f"\n--- Top {top_n} Hallucinated Empty Slices (by foreground pixel count) ---")
        for stem, fg_px, n_blobs, per_cls in hallucinated_details[:top_n]:
            cls_str = ", ".join(
                f"{ORGAN_NAMES.get(k, f'c{k}')}:{v}px" for k, v in sorted(per_cls.items())
            )
            print(f"  {stem}: {fg_px:4d} FG px across {n_blobs:2d} blob(s) [{cls_str}]")

        print(
            "\n>> Conclusion: The ENet model does NOT output entirely blank masks on all empty "
            f"validation slices ({hallucinated_slices}/{num_empty} slices contain hallucinated foreground blobs)."
        )
    else:
        print(
            "\n>> Conclusion: Confirmed! The ENet model outputs entirely blank masks on all "
            f"{num_empty} empty validation slices."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on completely empty validation slices to check if ENet outputs blank masks."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="SEGTHOR_FULL",
        help="Dataset folder name under --data_root (default: SEGTHOR_FULL).",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data"),
        help="Root data directory (default: data).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("results/segthor_train_full/ce/enet_baseline/bestweights.pt"),
        help="Path to ENet model weights (.pt) or model file (.pkl).",
    )
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--kernels", type=int, default=8)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--show_top", type=int, default=15)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_empty_slices_check(parse_args())

