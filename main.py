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
from lightning import seed_everything
from lightning.fabric import Fabric
from PIL import Image
from skimage.transform import resize
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import wandb
from dataset import SliceDataset
from models import get_model
from utils.losses import get_loss
from utils.metrics import dice_batch, dice_coef
import pytorch_lightning as pl
from utils.tensor_utils import (
    Dcm,
    class2one_hot,
    get_device,
    probs2class,
    probs2one_hot,
    save_images,
    tqdm_,
)


def setup_wandb(args):
    # Initialize a new W&B run
    wandb.init(
        project=args.wandb_project_name,
        config={
            "epochs": args.epochs,
            "dataset": args.dataset,
            "learning_rate": args.lr,
            "batch_size": args.datasets_params[args.dataset]["B"],
            "mode": args.mode,
            "seed": args.seed,
            "model": args.model_name,
            "loss": args.loss,
            "precision": args.precision,
            "include_background": args.include_background,
        },
    )


class MyModel(pl.LightningModule):
    def __init__(self, args, batch_size, K, train_loader, val_loader):
        super().__init__()
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.K = K

        self.log_loss_tra = torch.zeros((args.epochs, len(self.train_loader)))
        self.log_dice_tra = torch.zeros(
            (args.epochs, len(self.train_loader.dataset), self.K)
        )
        self.log_loss_val = torch.zeros((args.epochs, len(self.val_loader)))
        self.log_dice_val = torch.zeros(
            (args.epochs, len(self.val_loader.dataset), self.K)
        )
        self.best_dice = 0

        self.K: int = args.datasets_params[args.dataset]["K"]
        self.net = args.model(1, self.K)
        self.net.init_weights()
        self.loss_fn = get_loss(
            args.loss, self.K, include_background=args.include_background
        )

        # Dataset part
        self.batch_size: int = args.datasets_params[args.dataset]["B"]  # Batch size
        self.root_dir: Path = Path(args.data_dir) / str(args.dataset)
        self.gt_shape = self.get_gt_shape()

        self.log_dice_3d_tra = torch.zeros(
            (args.epochs, len(self.gt_shape["train"].keys()), self.K)
        )
        self.log_dice_3d_val = torch.zeros(
            (args.epochs, len(self.gt_shape["val"].keys()), self.K)
        )
        args.dest.mkdir(parents=True, exist_ok=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            filter(lambda x: x.requires_grad, self.net.parameters()), lr=self.args.lr
        )

    def get_gt_shape(self):
        # For each patient in dataset, get the ground truth volume shape
        self.gt_shape = {"train": {}, "val": {}}
        for split in self.gt_shape:
            directory = self.root_dir / split / "gt"
            split_patient_ids = set(x.stem.split("_")[1] for x in directory.iterdir())

            for patient_number in split_patient_ids:
                patient_id = f"Patient_{patient_number}"
                patients = list(directory.glob(patient_id + "*"))

                H, W = Image.open(patients[0]).size
                D = len(patients)
                self.gt_shape[split][patient_id] = (H, W, D)
        return self.gt_shape

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img = batch["images"]
        gt = batch["gts"]
        pred_logits = self(img)
        pred_probs = F.softmax(1 * pred_logits, dim=1)
        pred_seg = probs2one_hot(pred_probs)
        loss = self.loss_fn(pred_probs, gt)
        self.log_loss_tra[self.current_epoch, batch_idx] = loss.detach()
        self.log_dice_tra[
            self.current_epoch, batch_idx : batch_idx + img.size(0), :
        ] = dice_coef(pred_seg, gt)
        return loss

    def validation_step(self, batch, batch_idx):
        img = batch["images"]
        gt = batch["gts"]
        pred_logits = self(img)
        pred_probs = F.softmax(1 * pred_logits, dim=1)
        pred_seg = probs2one_hot(pred_probs)
        loss = self.loss_fn(pred_probs, gt)
        self.log_loss_val[self.current_epoch, batch_idx] = loss.detach()
        self.log_dice_val[
            self.current_epoch, batch_idx : batch_idx + img.size(0), :
        ] = dice_coef(pred_seg, gt)

    def on_validation_epoch_end(self):
        log_dict = {
            "val/loss": self.log_loss_val[self.current_epoch].mean().detach(),
            "val/dice": self.log_dice_val[self.current_epoch, :, 1:].mean().detach(),
            # "val/dice_class": self.get_dice_per_class(self.log_dice_val, self.K, self.current_epoch)
        }
        for k, v in self.get_dice_per_class(
            self.log_dice_val, self.K, self.current_epoch
        ).items():
            log_dict[f"val/dice_class/{k}"] = v
        if self.args.dataset == "SEGTHOR":
            log_dict["val/dice_3d"] = (
                self.log_dice_3d_val[self.current_epoch, :, 1:].mean().detach()
            )
            # log_dict["val/dice_3d_class"] = self.get_dice_per_class(self.log_dice_3d_val, self.K, self.current_epoch)
            for k, v in self.get_dice_per_class(
                self.log_dice_3d_val, self.K, self.current_epoch
            ).items():
                log_dict[f"val/dice_3d_class/{k}"] = v
        self.log_dict(log_dict)

        current_dice = self.log_dice_val[self.current_epoch, :, 1:].mean().detach()
        if current_dice > self.best_dice:
            self.best_dice = current_dice
            self.save_model()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def get_dice_per_class(self, log, K, e):
        if self.args.dataset == "SEGTHOR":
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
            dice_per_class = {
                f"dice_{k}": log[e, :, k].mean().item() for k in range(1, K)
            }
        return dice_per_class

    def save_model(self):
        torch.save(self.net, self.args.dest / "bestmodel.pkl")
        torch.save(self.net.state_dict(), self.args.dest / "bestweights.pt")
        if not self.args.wandb_project_name:
            self.logger.save_checkpoint(str(self.args.dest / "bestweights.pt"))


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")

    K = args.datasets_params[args.dataset]["K"]
    root_dir = Path(args.data_dir) / args.dataset
    batch_size = args.datasets_params[args.dataset]["B"]
    # Transforms
    from torchvision.transforms.v2 import Compose
    from torchvision import transforms

    img_transform = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    
    # class 
    
    gt_transform = v2.Compose(
        [
            v2.Lambda(lambda x: x/ (255 / (K - 1)) if K != 5 else x / 63),
            v2.ToDtype(torch.int64),
            # v2.Lambda(lambda x: x[None, ...]),
            v2.Lambda(lambda x: class2one_hot(x, K=K)[0]),

            # lambda img: np.array(img),
            # # The idea is that the classes are mapped to {0, 255} for binary cases
            # # {0, 85, 170, 255} for 4 classes
            # # {0, 51, 102, 153, 204, 255} for 6 classes
            # # Very sketchy but that works here and that simplifies visualization
            # lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # max <= 1
            # lambda nd: torch.from_numpy(nd).to(dtype=torch.int64)[
            #     None, ...
            # ],  # Add one dimension to simulate batch
            # lambda t: class2one_hot(t, K=K)[0],
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
        train_set, batch_size=batch_size, num_workers=args.num_workers, shuffle=True
    )

    val_set = SliceDataset(
        "val",
        root_dir,
        img_transform=img_transform,
        gt_transform=gt_transform,
        debug=args.debug,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, num_workers=args.num_workers, shuffle=False
    )

    model = MyModel(args, batch_size, K, train_loader, val_loader)

    trainer = pl.Trainer(
        accelerator="cpu" if args.cpu else "auto",
        max_epochs=args.epochs,
        precision=args.precision,
    )
    trainer.fit(model, train_loader, val_loader)


def get_args():
    # Dataset-specific parameters
    datasets_params = {
        # K = number of classes, B = batch size
        "TOY2": {"K": 2, "B": 2},
        "SEGTHOR": {"K": 5, "B": 8},
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument(
        "--dataset",
        default=next(iter(datasets_params)),
        choices=list(datasets_params),
        help="Which dataset to use for the training.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="data",
        help="The path to get the GT scan, in order to get the correct number of slices",
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
        "--model_name",
        type=str,
        default="shallowCNN",
        choices=["shallowCNN", "ENet", "UDBRNet"],
        help="Model to use for training",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducibility."
    )
    parser.add_argument(
        "--mode",
        default="full",
        choices=["partial", "full"],
        help="Whether to supervise all the classes ('full') or, "
        "only a subset of them ('partial').",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. "
        "Default 0 to avoid pickle lambda error.",
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

    # TODO: Check delta between
    parser.add_argument(
        "--lr", type=float, default=0.0005, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force the code to run on CPU, even if a GPU is available.",
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
    args.datasets_params = datasets_params
    return args


def main():
    args = get_args()
    print(args)
    if not args.wandb_project_name:
        setup_wandb(args)
    runTraining(args)


if __name__ == "__main__":
    main()
