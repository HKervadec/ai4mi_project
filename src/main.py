#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

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

import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from shutil import copytree, rmtree

import torch
from torch.optim.lr_scheduler import LRScheduler
import wandb
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader

from functools import partial
import autorootcwd  # noqa

from src.models import ShallowNet
from src.utils.args import Args, get_args
from src.utils.dataset import SliceDataset
from src.models.ShallowNet import shallowCNN
from src.models.ENet import ENet
from src.utils.utils import (
    Dcm,
    class2one_hot,
    get_root_dir,
    probs2one_hot,
    probs2class,
    seed_all,
    tqdm_,
    dice_coef,
    save_images,
)

from src.utils.losses import CrossEntropy

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {"K": 2, "net": shallowCNN, "B": 2, "kernels": 8, "factor": 2}
datasets_params["SEGTHOR"] = {"K": 5, "net": ENet, "B": 8, "kernels": 8, "factor": 2}
datasets_params["SEGTHOR_CLEAN"] = {
    "K": 5,
    "net": ENet,
    "B": 8,
    "kernels": 8,
    "factor": 2,
}


def img_transform(img):
    img = img.convert("L")
    img = np.array(img)[np.newaxis, ...]
    img = img / 255  # max <= 1
    img = torch.tensor(img, dtype=torch.float32)
    return img


def gt_transform(K, img):
    img = np.array(img)[...]
    # The idea is that the classes are mapped to {0, 255} for binary cases
    # {0, 85, 170, 255} for 4 classes
    # {0, 51, 102, 153, 204, 255} for 6 classes
    # Very sketchy but that works here and that simplifies visualization
    img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
    img = torch.tensor(img, dtype=torch.int64)[
        None, ...
    ]  # Add one dimension to simulate batch
    img = class2one_hot(img, K=K)
    return img[0]


def setup(
    args: Args,
) -> tuple[nn.Module, Any, LRScheduler, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]["K"]
    kernels: int = (
        datasets_params[args.dataset]["kernels"]
        if "kernels" in datasets_params[args.dataset]
        else 8
    )
    factor: int = (
        datasets_params[args.dataset]["factor"]
        if "factor" in datasets_params[args.dataset]
        else 2
    )

    # dropoutRate might not be in class, probably want more robust type checking here
    net: ENet | ShallowNet.shallowCNN = datasets_params[args.dataset]["net"](
        1, K, kernels=kernels, factor=factor, dropoutRate=args.dropout
    )
    net.init_weights()
    net.to(device)

    lr = args.lr
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=lr, weight_decay=args.weight_decay, betas=args.betas
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Dataset part
    batch_size: int = datasets_params[args.dataset]["B"]
    data_root_dir = autorootcwd.root / "data" / args.dataset

    train_set = SliceDataset(
        "train",
        data_root_dir,
        img_transform=img_transform,
        gt_transform=partial(gt_transform, K),
        debug=args.debug,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(args.seed),
        shuffle=True,
    )

    val_set = SliceDataset(
        "val",
        data_root_dir,
        img_transform=img_transform,
        gt_transform=partial(gt_transform, K),
        debug=args.debug,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False,
    )

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, scheduler, device, train_loader, val_loader, K)


def get_loss_func(args: Args, num_classes: int):
    if args.mode == "full":
        return CrossEntropy(
            idk=list(range(num_classes))
        )  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == "SEGTHOR":
        return CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)


def runTraining(args: Args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")

    net, optimizer, scheduler, device, train_loader, val_loader, num_classes = setup(
        args
    )

    wandb.init(
        entity="ai-for-medical-imaging",
        project=f"{args.dataset}",
        config=vars(args) | datasets_params[args.dataset],
        dir=get_root_dir() / "results" / "wandb",
    )

    # Adds histogram of the gradients and parameters
    # NOTE Does add a lot of info to our project, need to see if we want that
    if args.wandb_watch:
        wandb.watch(net, log="all", log_freq=100)

    loss_fn = get_loss_func(args, num_classes)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros(
        (args.epochs, len(train_loader.dataset), num_classes)  # type: ignore
    )
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros(
        (args.epochs, len(val_loader.dataset), num_classes)  # type: ignore
    )

    best_dice: float = 0

    # NOTE Just need a total rewrite of this, split it up into functions
    # Also not handy bc train and val are in this same loop
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
                case "val":
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val
                case _:
                    raise  # Should never be reached, but needed to silence ide warn

            with (
                cm()
            ):  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                total_correct = 0
                total_pixels = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data["images"].to(device)
                    gt = data["gts"].to(device)

                    if opt is not None:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    batch_size, _, W, H = img.shape

                    with torch.autocast(device_type="cuda" if args.gpu else "cpu"):
                        pred_logits = net(img)
                        pred_probs = F.softmax(
                            args.temperature * pred_logits, dim=1
                        )  # 1 is the temperature parameter

                        # Metrics computation, not used for training
                        pred_seg = probs2one_hot(pred_probs)
                        log_dice[e, j : j + batch_size, :] = dice_coef(
                            pred_seg, gt
                        )  # One DSC value per sample and per class

                        # Pixel-wise accuracy
                        predicted_classes = pred_probs.argmax(dim=1)  # (B, W, H)
                        gt_classes = gt.argmax(dim=1)  # (B, W, H)
                        total_correct += (predicted_classes == gt_classes).sum().item()
                        total_pixels += predicted_classes.numel()

                        loss = loss_fn(pred_probs, gt)
                        log_loss[e, i] = (
                            loss.item()
                        )  # One loss value per batch (averaged in the loss)

                    if opt is not None:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == "val":
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = (
                                63 if num_classes == 5 else int(255 / (num_classes - 1))
                            )
                            save_images(
                                predicted_class * mult,
                                data["stems"],
                                args.dest / f"iter{e:03d}" / m,
                            )

                    j += batch_size  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    epoch_acc = total_correct / total_pixels
                    postfix_dict: dict[str, str] = {
                        "Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                        "Loss": f"{log_loss[e, : i + 1].mean():5.2e}",
                        "Acc": f"{epoch_acc:05.3f}",
                    }
                    if num_classes > 2:
                        postfix_dict |= {
                            f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                            for k in range(1, num_classes)
                        }
                    tq_iter.set_postfix(postfix_dict)

                if m == "train":
                    acc_tra = epoch_acc
                else:
                    acc_val = epoch_acc

        metrics = {
            "epoch": e,
            "train/loss": log_loss_tra[e].mean().item(),
            "train/dice": log_dice_tra[e, :, 1:].mean().item(),
            "train/acc": acc_tra,
            "val/loss": log_loss_val[e].mean().item(),
            "val/dice": log_dice_val[e, :, 1:].mean().item(),
            "val/acc": acc_val,
        }
        if num_classes > 2:
            for k in range(1, num_classes):
                metrics[f"train/dice_{k}"] = log_dice_tra[e, :, k].mean().item()
                metrics[f"val/dice_{k}"] = log_dice_val[e, :, k].mean().item()
        wandb.log(metrics)

        # Scheduler at the end of each epoch
        scheduler.step()

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", "w") as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net.state_dict(), args.dest / "bestweights.pt")

    # Wait for the background logging thread to finish
    wandb.finish()


def main():
    args = get_args()

    # Seed everything right at the beginning
    seed_all(args.seed, args.gpu)

    pprint(args)

    runTraining(args)


if __name__ == "__main__":
    main()
