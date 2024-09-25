#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os

# MPS issue: aten::max_unpool2d' not available for MPS devices
# Solution: set fallback to 1 before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import os
import warnings
from pathlib import Path
from shutil import copytree, rmtree
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from PIL import Image
from skimage.transform import resize
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import transforms

import wandb
from dataset import SliceDataset
from models import get_model
from utils.losses import get_loss
from utils.metrics import dice_batch, dice_coef
from utils.tensor_utils import (
    Dcm,
    class2one_hot,
    probs2class,
    probs2one_hot,
    save_images,
    tqdm_,
    print_args,
    set_seed
)

torch.set_float32_matmul_precision("medium")

# Initialize a new W&B run
def setup_wandb(args):
    wandb.init(
        project=args.wandb_project_name,
        config={
            "epochs": args.epochs,
            "dataset": args.dataset,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "mode": args.mode,
            "seed": args.seed,
            "model": args.model_name,
            "loss": args.loss,
            "precision": args.precision,
            "include_background": args.include_background,
        },
    )

def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int, Fabric]:
    
    # Seed and Fabric initialization
    set_seed(args.seed)
    fabric = Fabric(precision=args.precision, accelerator="cpu" if args.cpu else "auto")

    # Networks and scheduler
    device = fabric.device
    print(f">> Running on device '{device}'")
    K: int = args.datasets_params[args.dataset]["K"]
    net = args.model(1, K)
    net.init_weights()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    net, optimizer = fabric.setup(net, optimizer)

    # Dataset part
    B: int = args.datasets_params[args.dataset]["B"]  # Batch size
    root_dir: Path = Path(args.data_dir) / str(args.dataset)

    # Transforms for images and ground truth
    img_transform = transforms.Compose(
        [
            lambda img: img.convert("L"), # Convert to grayscale
            transforms.PILToTensor(),     # Convert to tensor 
            lambda img: img / 255,        # Normalize to [0, 1]
        ]
    ) # img_tensor.shape = [1, H, W]
    gt_transform = transforms.Compose(
        [
            lambda img: np.array(img), # img values are in [0, 255]
            # For 2 classes, the classes are mapped to {0, 255}.
            # For 4 classes, the classes are mapped to {0, 85, 170, 255}.
            # For 6 classes, the classes are mapped to {0, 51, 102, 153, 204, 255}.
            lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # Normalization
            lambda nd: torch.from_numpy(nd).to(dtype=torch.int64, device=fabric.device)[
                None, ...
            ],  # Add one dimension to simulate batch
            lambda t: class2one_hot(t, K=K)[0], # Tensor: One-hot encoding [B, K, H, W]
        ]
    )

    # Datasets and loaders
    train_set = SliceDataset(
        "train",
        root_dir,
        img_transform=img_transform,
        gt_transform=gt_transform,
        debug=args.debug,
    )
    train_loader = DataLoader(
        train_set, batch_size=B, num_workers=args.num_workers, shuffle=True
    )

    val_set = SliceDataset(
        "val",
        root_dir,
        img_transform=img_transform,
        gt_transform=gt_transform,
        debug=args.debug,
    )
    val_loader = DataLoader(
        val_set, batch_size=B, num_workers=args.num_workers, shuffle=False
    )

    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    args.dest.mkdir(parents=True, exist_ok=True)

    # For each patient in dataset, get the ground truth volume shape
    gt_shape = {"train": {}, "val": {}}
    for split in gt_shape:
        directory = root_dir / split / "gt"
        split_patient_ids = set(x.stem.split("_")[1] for x in directory.iterdir())

        for patient_number in split_patient_ids:
            patient_id = f"Patient_{patient_number}"
            patients = list(directory.glob(patient_id + "*"))

            H, W = Image.open(patients[0]).size
            D = len(patients)
            gt_shape[split][patient_id] = (H, W, D)

    return (net, optimizer, device, train_loader, val_loader, K, gt_shape, fabric)


def runTraining(args):

    print(f">>> Setting up to train on '{args.dataset}' with '{args.mode}'")
    net, optimizer, device, train_loader, val_loader, K, gt_shape, fabric = setup(args)
    
    # Logging loss per batch for each epoch = [epochs, num_batches], num_batches = data_size/B
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))

    # Logging 2D Dice loss per sample and class = [epochs, num_samples, num_classes]
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K)) #[e, 5453, K]
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))   #[e, 1967, K]

    # Logging 3D Dice loss per patient and class = [epochs, num_patients, num_classes]
    log_dice_3d_tra: Tensor = torch.zeros(((args.epochs, len(gt_shape["train"].keys()), K))) #[e, 30, K]
    log_dice_3d_val: Tensor = torch.zeros(((args.epochs, len(gt_shape["val"].keys()), K)))   #[e, 10, K]

    best_dice: float = 0
    loss_fn = get_loss(args.loss, K, include_background=args.include_background)

    for e in range(args.epochs):
        for m in ["train", "val"]:
            match m:
                case "train":
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                    log_dice_3d = log_dice_3d_tra
                case "val":
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val
                    log_dice_3d = log_dice_3d_val

            # Initialize 3D volumes [Z, K, X, Y] for each patient 
            # Z: Depth, K: Number of classes, (X, Y): Spatial dimensions
            gt_volumes = {
                p: np.zeros((Z, K, X, Y), dtype=np.int16)
                for p, (X, Y, Z) in gt_shape[m].items()
            }
            pred_volumes = {
                p: np.zeros((Z, K, X, Y), dtype=np.int16)
                for p, (X, Y, Z) in gt_shape[m].items()
            }

            with cm():  # Train: dummy context manager, Val: torch.no_grad 
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data["images"]
                    gt = data["gts"]

                    # print(f"Shape of img at epoch {e}, batch {i}: {img.shape}")
                    # Shape of img at epoch 0, batch 0: torch.Size([8, 1, 256, 256])

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(pred_logits / args.temperature, dim=1)

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j : j + B, :] = dice_coef(
                        pred_seg, gt
                    )  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = (
                        loss.item()
                    )  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        fabric.backward(loss)
                        opt.step()

                    # Save predictions and gt slice for 3D Dice computation
                    for i, seg_class in enumerate(pred_seg):
                        stem = data["stems"][i]
                        _, patient_n, z = stem.split("_")
                        patient_id = f"Patient_{patient_n}"

                        X, Y, _ = gt_shape[m][patient_id]

                        resize_and_save_slice(
                            seg_class, K, X, Y, z, pred_volumes[patient_id]
                        )
                        resize_and_save_slice(gt[i], K, X, Y, z, gt_volumes[patient_id])

                    if m == "val":
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))

                            # Save predictions in logging
                            save_images(
                                predicted_class * mult,
                                data["stems"],
                                args.dest / f"iter{e:03d}" / m,
                            )

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {
                        "Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                        "Loss": f"{log_loss[e, :i + 1].mean():5.2e}",
                    }
                    if K > 2:
                        postfix_dict |= {
                            f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                            for k in range(1, K)
                        }
                    tq_iter.set_postfix(postfix_dict)

                log_dict = {
                    m: {
                        "loss": log_loss[e].mean().item(),
                        "dice": log_dice[e, :, 1:].mean().item(),
                        "dice_class": get_dice_per_class(args, log_dice, K, e),
                    }
                }

                # Compute 3D Dice
                if m == "val":
                    print("Computing 3D dice...")
                    for i, (patient_id, pred_vol) in tqdm_(
                        enumerate(pred_volumes.items()), total=len(pred_volumes)
                    ):
                        gt_vol = torch.from_numpy(gt_volumes[patient_id]).to(device)
                        pred_vol = torch.from_numpy(pred_vol).to(device)

                        dice_3d = dice_batch(gt_vol, pred_vol)
                        log_dice_3d[e, i, :] = dice_3d
                    log_dict["dice_3d"] = (log_dice_3d[e, :, 1:].mean().item(),)
                    log_dict["dice_3d_class"] = get_dice_per_class(
                        args, log_dice_3d, K, e
                    )

                # Log the metrics after each 'e' epoch
                if args.wandb_project_name:
                    wandb.log(log_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            print(
                f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            )
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", "w") as f:
                f.write(str(e))

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")

            # Log model checkpoint
            if args.wandb_project_name:
                wandb.save(str(args.dest / "bestmodel.pkl"))
                wandb.save(str(args.dest / "bestweights.pt"))


def get_dice_per_class(args, log, K, e):
    if args.dataset == "SEGTHOR":
        class_names = [
            (1, "background"),
            (2, "esophagus"),
            (3, "heart"),
            (4, "trachea"),
            (5, "aorta"),
        ]
        dice_per_class = {
            f"dice_{k}_{n}": log[e, :, k - 1].mean().item() for k, n in class_names
        }
    else:
        dice_per_class = {f"dice_{k}": log[e, :, k].mean().item() for k in range(1, K)}

    return dice_per_class


def resize_and_save_slice(arr, K, X, Y, z, target_arr):
    resized_arr = resize(
        arr.cpu().numpy(),
        (K, X, Y),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
        order=0,
    )
    target_arr[int(z), :, :, :] = resized_arr[...]


def get_args():

    # Group 1: Dataset & Model configuration
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="shallowCNN",
        choices=["shallowCNN", "ENet", "UDBRNet"],
        help="Model to use for training",
    )
    parser.add_argument(
        "--mode",
        default="full",
        choices=["partial", "full"],
        help="Whether to supervise all the classes ('full') or, "
        "only a subset of them ('partial').",
    )
    parser.add_argument(
        "--dataset",
        default="SEGTHOR",
        choices=["SEGTHOR", "TOY2"],
        help="Which dataset to use for the training.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="data",
        help="Path to get the GT scan, in order to get the correct number of slices",
    )

    # Group 2: Training parameters
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument(
        "--lr", type=float, default=0.0005, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--loss",
        choices=["ce", "dice", "dicece", "dicefocal", "ce_torch"],
        default="dicefocal",
        help="Loss function to use for training.",
    )
    parser.add_argument(
        "--include_background",
        action="store_true",
        help="Whether to include the background class in the loss computation.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducibility."
    )
    parser.add_argument(
        "--precision",
        default=32,
        type=str,
        choices=[
            "bf16",
            "bf16-mixed",
            "bf16-true",
            "16",
            "16-mixed",
            "16-true",
            "32",
            "64",
        ],
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force the code to run on CPU, even if a GPU is available.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. "
        "Default 0 to avoid pickle lambda error.",
    )

    # Group 3: Output directory
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory to save the results (predictions and weights).",
    )

    # Group 4: Debugging and logging settings
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep only a fraction (10 samples) of the datasets, "
        "to test the logic around epochs and logging easily.",
    )
    parser.add_argument(
        "--wandb_project_name",  # clean code dictates I leave this as "--wandb" but I'm not breaking people's flows yet
        type=str,
        help="Project wandb will be logging run to.",
    )

    args = parser.parse_args()

    # If dest not provided, create one
    if args.dest is None:
        # CE: 'args.mode = full'
        # Other: 'args.mode = partial'
        args.dest = Path(f"results/{args.dataset}/{args.mode}/{args.model_name}")

    # Model selection
    args.model = get_model(args.model_name)
    print_args(args)

    args.datasets_params = {
        # K = number of classes, B = batch size
        "TOY2": {"K": 2, "B": args.batch_size},
        "SEGTHOR": {"K": 5, "B": args.batch_size},
    }
    return args


def main():
    args = get_args()
    if args.wandb_project_name: 
        setup_wandb(args)
    runTraining(args)


if __name__ == "__main__":
    main()
