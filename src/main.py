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

import dataclasses
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

from src.utils.config import Config, get_config
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
    gated_dice,
    save_images,
)

from src.utils.losses import CrossEntropy


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
    config: Config,
) -> tuple[nn.Module, Any, LRScheduler, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    device = torch.device("cuda") if config.gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    num_classes: int = config.dataset.num_classes
    kernels: int = config.model.kernels
    factor: int = config.model.factor

    # NOTE Gonna rewrite this into a BaseModel which can load any subclass from str
    if config.model.name == "ENet":
        net = ENet(
            1, num_classes, kernels=kernels, factor=factor, dropoutRate=config.dropout
        )
    else:
        net = shallowCNN(
            1, num_classes, kernels=kernels, factor=factor, dropoutRate=config.dropout
        )

    net.init_weights()
    net.to(device)

    lr = config.lr
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=lr, weight_decay=config.weight_decay, betas=config.betas
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    # Dataset part
    batch_size: int = config.batch_size
    data_root_dir = autorootcwd.root / "data" / config.dataset.name

    train_set = SliceDataset(
        "train",
        data_root_dir,
        img_transform=img_transform,
        gt_transform=partial(gt_transform, num_classes),
        debug=config.debug,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(config.seed),
        shuffle=True,
    )

    val_set = SliceDataset(
        "val",
        data_root_dir,
        img_transform=img_transform,
        gt_transform=partial(gt_transform, num_classes),
        debug=config.debug,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False,
    )

    config.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, scheduler, device, train_loader, val_loader, num_classes)


def get_loss_func(config: Config, num_classes: int):
    if config.mode == "full":
        return CrossEntropy(
            idk=list(range(num_classes))
        )  # Supervise both background and foreground
    elif config.mode in ["partial"] and config.dataset.name == "SEGTHOR":
        return CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(config.mode, config.dataset)


def runTraining(config: Config):
    print(f">>> Setting up to train on {config.dataset} with {config.mode}")

    net, optimizer, scheduler, device, train_loader, val_loader, num_classes = setup(
        config
    )

    scaler = torch.amp.GradScaler("cuda", enabled=config.gpu)

    wandb.init(
        entity="ai-for-medical-imaging",
        project=f"{config.dataset}-baseline",
        config=dataclasses.asdict(config),
        dir=get_root_dir() / "results" / "wandb",
    )

    # Adds histogram of the gradients and parameters
    # NOTE Does add a lot of info to our project, need to see if we want that
    if config.wandb_watch:
        wandb.watch(net, log="all", log_freq=100)

    loss_fn = get_loss_func(config, num_classes)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((config.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros(
        (config.epochs, len(train_loader.dataset), num_classes)  # type: ignore
    )
    log_loss_val: Tensor = torch.zeros((config.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros(
        (config.epochs, len(val_loader.dataset), num_classes)  # type: ignore
    )
    log_presence_tra: Tensor = torch.zeros(
        (config.epochs, len(train_loader.dataset), num_classes),  # type: ignore
        dtype=torch.bool,
    )
    log_presence_val: Tensor = torch.zeros(
        (config.epochs, len(val_loader.dataset), num_classes),  # type: ignore
        dtype=torch.bool,
    )

    best_dice: float = 0

    # NOTE Just need a total rewrite of this, split it up into functions
    # Also not handy bc train and val are in this same loop
    for e in range(config.epochs):
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
                    log_presence = log_presence_tra
                case "val":
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val
                    log_presence = log_presence_val
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

                    with torch.autocast(device_type="cuda" if config.gpu else "cpu"):
                        pred_logits = net(img)
                        pred_probs = F.softmax(
                            config.temperature * pred_logits, dim=1
                        )  # 1 is the temperature parameter

                        # Metrics computation, not used for training
                        pred_seg = probs2one_hot(pred_probs)
                        log_dice[e, j : j + batch_size, :] = dice_coef(
                            pred_seg, gt
                        )  # One DSC value per sample and per class
                        log_presence[e, j : j + batch_size, :] = gt.sum(dim=(-2, -1)) > 0

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
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()

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
                                config.dest / f"iter{e:03d}" / m,
                            )

                    j += batch_size  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    epoch_acc = total_correct / total_pixels
                    d, p = log_dice[e, :j], log_presence[e, :j]
                    postfix_dict: dict[str, str] = {
                        "Dice": f"{gated_dice(d[:, 1:], p[:, 1:]):05.3f}",
                        "Loss": f"{log_loss[e, : i + 1].mean():5.2e}",
                        "Acc": f"{epoch_acc:05.3f}",
                    }
                    if num_classes > 2:
                        postfix_dict |= {
                            f"Dice-{k}": f"{gated_dice(d[:, k], p[:, k]):05.3f}"
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
            "train/dice": gated_dice(
                log_dice_tra[e, :, 1:], log_presence_tra[e, :, 1:]
            ).item(),
            "train/acc": acc_tra,
            "val/loss": log_loss_val[e].mean().item(),
            "val/dice": gated_dice(
                log_dice_val[e, :, 1:], log_presence_val[e, :, 1:]
            ).item(),
            "val/acc": acc_val,
        }
        if num_classes > 2:
            for k in range(1, num_classes):
                metrics[f"train/dice_{k}"] = gated_dice(
                    log_dice_tra[e, :, k], log_presence_tra[e, :, k]
                ).item()
                metrics[f"val/dice_{k}"] = gated_dice(
                    log_dice_val[e, :, k], log_presence_val[e, :, k]
                ).item()
        wandb.log(metrics)

        # Scheduler at the end of each epoch
        scheduler.step()

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(config.dest / "loss_tra.npy", log_loss_tra)
        np.save(config.dest / "dice_tra.npy", log_dice_tra)
        np.save(config.dest / "presence_tra.npy", log_presence_tra)
        np.save(config.dest / "loss_val.npy", log_loss_val)
        np.save(config.dest / "dice_val.npy", log_dice_val)
        np.save(config.dest / "presence_val.npy", log_presence_val)

        current_dice: float = gated_dice(
            log_dice_val[e, :, 1:], log_presence_val[e, :, 1:]
        ).item()
        if current_dice > best_dice:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            with open(config.dest / "best_epoch.txt", "w") as f:
                f.write(message)

            best_folder = config.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(config.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net.state_dict(), config.dest / "bestweights.pt")

    # Wait for the background logging thread to finish
    wandb.finish()


def main():
    args = get_config()

    # Seed everything right at the beginning
    seed_all(args.seed, args.gpu)

    pprint(args)

    runTraining(args)


if __name__ == "__main__":
    main()
