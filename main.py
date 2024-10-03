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

# MPS issue: aten::max_unpool2d' not available for MPS devices
# Solution: set fallback to 1 before importing torch
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from lightning import seed_everything

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.transform import resize
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import wandb
from dataset import SliceDataset
from models import get_model
from utils.losses import get_loss
from utils.metrics import dice_batch, dice_coef
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from utils.tensor_utils import (
    probs2class,
    probs2one_hot,
    save_images,
    tqdm_,
    print_args,
    set_seed,
)

torch.set_float32_matmul_precision("medium")


class ReScale(v2.Transform):
    def __init__(self, K):
        self.scale = 1 / (255 / (K - 1)) if K != 5 else 1 / 63

    def __call__(self, img):
        return img * self.scale


class Class2OneHot(v2.Transform):
    def __init__(self, K):
        self.K = K

    def __call__(self, seg):
        b, *img_shape = seg.shape

        device = seg.device
        res = torch.zeros(
            (b, self.K, *img_shape), dtype=torch.int32, device=device
        ).scatter_(1, seg[:, None, ...], 1)
        return res[0]


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
    return target_arr


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


class MyModel(pl.LightningModule):
    def __init__(self, args, batch_size, K, train_loader, val_loader):
        super().__init__()

        # Model part
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.K = K
        self.net = args.model(1, self.K)
        self.net.init_weights()
        self.loss_fn = get_loss(
            args.loss, self.K, include_background=args.include_background
        )

        # Dataset part
        self.root_dir: Path = Path(args.data_dir) / str(args.dataset)
        self.gt_shape = self._get_gt_shape()

        # Logging part
        self.log_loss_tra = torch.zeros((args.epochs, len(self.train_loader)))
        self.log_dice_tra = torch.zeros(
            (args.epochs, len(self.train_loader.dataset), self.K)
        )
        self.log_loss_val = torch.zeros((args.epochs, len(self.val_loader)))
        self.log_dice_val = torch.zeros(
            (args.epochs, len(self.val_loader.dataset), self.K)
        )
        self.log_dice_3d_tra = torch.zeros(
            (args.epochs, len(self.gt_shape["train"].keys()), self.K)
        )
        self.log_dice_3d_val = torch.zeros(
            (args.epochs, len(self.gt_shape["val"].keys()), self.K)
        )

        self.best_dice = 0

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.net.parameters()), lr=self.args.lr
        )

    def _get_gt_shape(self):
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

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()

        self.gt_volumes = {
            p: np.zeros((Z, self.K, X, Y), dtype=np.uint8)
            for p, (X, Y, Z) in self.gt_shape["val"].items()
        }

        self.pred_volumes = {
            p: np.zeros((Z, self.K, X, Y), dtype=np.uint8)
            for p, (X, Y, Z) in self.gt_shape["val"].items()
        }

    def _prepare_3d_dice(self, batch_stems, gt, pred_seg):
        for i, seg_class in enumerate(pred_seg):
            stem = batch_stems[i]
            _, patient_n, z = stem.split("_")
            patient_id = f"Patient_{patient_n}"

            X, Y, _ = self.gt_shape["val"][patient_id]

            self.pred_volumes[patient_id] = resize_and_save_slice(
                seg_class, self.K, X, Y, z, self.pred_volumes[patient_id]
            )
            self.gt_volumes[patient_id] = resize_and_save_slice(
                gt[i], self.K, X, Y, z, self.gt_volumes[patient_id]
            )

    def forward(self, x):
        # Sanity tests to see we loaded and encoded the data correctly
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, gt = batch["images"], batch["gts"]
        pred_logits = self(img)
        pred_probs = F.softmax(1 * pred_logits, dim=1)
        pred_seg = probs2one_hot(pred_probs)
        loss = self.loss_fn(pred_probs, gt)
        self.log_loss_tra[self.current_epoch, batch_idx] = loss.detach()
        self.log_dice_tra[
            self.current_epoch, batch_idx : batch_idx + img.size(0), :
        ] = dice_coef(pred_seg, gt)

        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log_dict(
            {
                f"train/dice/{k}": self.log_dice_tra[
                    self.current_epoch, : batch_idx + img.size(0), k
                ].mean()
                for k in range(1, self.K)
            },
            prog_bar=True,
            logger=False,
            on_step=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        img, gt = batch["images"], batch["gts"]
        pred_logits = self(img)
        pred_probs = F.softmax(1 * pred_logits, dim=1)
        pred_seg = probs2one_hot(pred_probs)
        loss = self.loss_fn(pred_probs, gt)
        self.log_loss_val[self.current_epoch, batch_idx] = loss.detach()
        self.log_dice_val[
            self.current_epoch, batch_idx : batch_idx + img.size(0), :
        ] = dice_coef(pred_seg, gt)

        self._prepare_3d_dice(batch["stems"], gt, pred_seg)

    def on_validation_epoch_end(self):
        log_dict = {
            "val/loss": self.log_loss_val[self.current_epoch].mean().detach(),
            "val/dice/total": self.log_dice_val[self.current_epoch, :, 1:]
            .mean()
            .detach(),
        }
        for k, v in self.get_dice_per_class(
            self.log_dice_val, self.K, self.current_epoch
        ).items():
            log_dict[f"val/dice/{k}"] = v
        if self.args.dataset == "SEGTHOR":
            for i, (patient_id, pred_vol) in tqdm_(
                enumerate(self.pred_volumes.items()), total=len(self.pred_volumes)
            ):
                gt_vol = torch.from_numpy(self.gt_volumes[patient_id]).to(self.device)
                pred_vol = torch.from_numpy(pred_vol).to(self.device)

                dice_3d = dice_batch(gt_vol, pred_vol)
                self.log_dice_3d_val[self.current_epoch, i, :] = dice_3d

            log_dict["val/dice_3d/total"] = (
                self.log_dice_3d_val[self.current_epoch, :, 1:].mean().detach()
            )
            # log_dict["val/dice_3d_class"] = self.get_dice_per_class(self.log_dice_3d_val, self.K, self.current_epoch)
            for k, v in self.get_dice_per_class(
                self.log_dice_3d_val, self.K, self.current_epoch
            ).items():
                log_dict[f"val/dice_3d/{k}"] = v
        self.log_dict(log_dict)

        current_dice = self.log_dice_val[self.current_epoch, :, 1:].mean().detach()
        if current_dice > self.best_dice:
            self.best_dice = current_dice
            self.save_model()

        super().on_validation_epoch_end()

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
        if self.args.wandb_project_name:
            self.logger.save(str(self.args.dest / "bestweights.pt"))


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")

    K = args.datasets_params[args.dataset]["K"]
    root_dir = Path(args.data_dir) / args.dataset
    batch_size = args.datasets_params[args.dataset]["B"]
    args.dest.mkdir(parents=True, exist_ok=True)

    # Transforms
    img_transform = v2.Compose([v2.ToDtype(torch.float32, scale=True)])
    gt_transform = v2.Compose([ReScale(K), v2.ToDtype(torch.int64), Class2OneHot(K)])

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

    wandb_logger = (
        WandbLogger(project=args.wandb_project_name)
        if args.wandb_project_name
        else None
    )

    trainer = pl.Trainer(
        accelerator="cpu" if args.cpu else "auto",
        max_epochs=args.epochs,
        precision=args.precision,
        num_sanity_val_steps=0,  # Sanity check fails due to the 3D dice computation
        logger=wandb_logger,
    )
    trainer.fit(model, train_loader, val_loader)


def get_args():

    # Group 1: Dataset & Model configuration
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="shallowCNN",
        choices=["shallowCNN", "ENet"],
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
    parser.add_argument("--temperature", default=1, type=float)
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
        "TOY2": {"K": 2, "B": args.batch_size},
        "SEGTHOR": {"K": 5, "B": args.batch_size},
    }
    return args


def main():
    args = get_args()

    seed_everything(args.seed)
    if not args.wandb_project_name:
        setup_wandb(args)
    runTraining(args)


if __name__ == "__main__":
    main()
