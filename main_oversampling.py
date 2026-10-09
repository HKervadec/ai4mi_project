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

import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree
from PIL import Image

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import BatchSampler, DataLoader, Sampler

from functools import partial 

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from ViT import ViT
from swin_model import build_swin_unet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   nsd_score,
                   cldice,
                   save_images)

from losses import (CrossEntropy, TverskyLoss)

METRIC_SPACING_MM = (500 / 256, 500 / 256)

index_pools = {
        "background": 0,                    # Background
        "large-organ foreground": 2,        # Heart
        "medium-organ foreground": 4,       # Aorta 
        "small-organ foreground": {1, 3}    # Esophagus, Trachea
}

models = {
    "ENet": ENet,
    "ViT": ViT,
    "SwinUnet": build_swin_unet
}

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN_BG"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_FULL"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_FULL_CROPPED"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_FULL_RM_BG"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}




def resize_if_needed(img, img_size, resampling):
    if img_size is not None:
        img = img.resize((img_size, img_size), resampling)
    return img


def img_transform(img, img_size=None):
        img = img.convert('L')

        img = resize_if_needed(img, img_size, Image.Resampling.BILINEAR)


        img = np.array(img)[np.newaxis, ...]
        img = img / 255  # max <= 1
        img = torch.tensor(img, dtype=torch.float32)
        return img

def gt_transform(K, img, img_size=None):
        img = img.convert('L')
        img = resize_if_needed(img, img_size, Image.Resampling.NEAREST)
        img = np.array(img)[...]
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
        img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
        img = class2one_hot(img, K=K)
        return img[0]

class ForegroundOversamplingBatchSampler(Sampler[list[int]]):
    """Yield batches with foreground and small-organ slices.
        1/6 of the slices in the batch contain small-organ slices,
        1/3 of the slices contain other foreground slices,
        1/2 of the slices contain background slices """

    def __init__(self, dataset: SliceDataset, batch_size: int,
                 oversample_foreground_percent: float = 1 / 3,
                 oversample_small_organ_percent: float = 0.0,
                 drop_last: bool = False):
        if not 0 <= oversample_foreground_percent <= 1:
            raise ValueError("oversample_foreground_percent must be in [0, 1]")
        if not 0 <= oversample_small_organ_percent <= oversample_foreground_percent:
            raise ValueError(
                "oversample_small_organ_percent must be in "
                "[0, oversample_foreground_percent]"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.oversample_foreground_percent = oversample_foreground_percent
        self.oversample_small_organ_percent = oversample_small_organ_percent
        self.foreground_indices: list[int] = []
        self.background_indices: list[int] = []
        self.small_organ_indices: list[int] = []

        for index, (_, gt_path) in enumerate(dataset.files):
            if gt_path is None:
                raise ValueError("Foreground oversampling requires ground-truth labels")
            with Image.open(gt_path) as gt:
                labels = np.asarray(gt)
                if np.any(labels > 0):
                    self.foreground_indices.append(index)
                    # SEGTHOR labels use 0, 63, 126, 189, and 252.
                    class_ids = labels // 63 if labels.max(initial=0) > 4 else labels
                    if np.isin(class_ids, list(index_pools["small-organ foreground"])).any():
                        self.small_organ_indices.append(index)
                else:
                    self.background_indices.append(index)

        if not self.foreground_indices:
            raise ValueError("The dataset contains no foreground slices")
        if oversample_small_organ_percent > 0 and not self.small_organ_indices:
            raise ValueError("The dataset contains no small-organ slices")

    def __len__(self) -> int:
        dataset_size = len(self.foreground_indices) + len(self.background_indices)
        if self.drop_last:
            return dataset_size // self.batch_size
        return (dataset_size + self.batch_size - 1) // self.batch_size

    @staticmethod
    def _draw(indices: list[int], count: int) -> Tensor:
        if count == 0:
            return torch.empty(0, dtype=torch.long)
        source = torch.tensor(indices, dtype=torch.long)
        if count <= len(source):
            return source[torch.randperm(len(source))[:count]]
        return source[torch.randint(len(source), (count,))]

    def __iter__(self):
        for _ in range(len(self)):
            foreground_count = min(
                round(self.batch_size * self.oversample_foreground_percent),
                self.batch_size,
            )
            small_organ_count = min(
                round(self.batch_size * self.oversample_small_organ_percent),
                foreground_count,
            )
            background_count = self.batch_size - foreground_count

            # If every slice is foreground, use it as the fallback pool.
            background_pool = self.background_indices or self.foreground_indices
            small_organs = self._draw(self.small_organ_indices, small_organ_count)
            foreground = self._draw(
                self.foreground_indices,
                foreground_count - small_organ_count,
            )
            background = self._draw(background_pool, background_count)
            batch = torch.cat((small_organs, foreground, background))
            batch = batch[torch.randperm(len(batch))]
            yield batch.tolist()


def set_batch_size_and_oversample(dataset: SliceDataset, batch_size: int,
                                  oversample_foreground_percent: float,
                                  oversample_small_organ_percent: float = 0.0) -> BatchSampler:
    return ForegroundOversamplingBatchSampler(
        dataset,
        batch_size=batch_size,
        oversample_foreground_percent=oversample_foreground_percent,
        oversample_small_organ_percent=oversample_small_organ_percent,
    )

def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
    factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2

    net_class = models[args.model]

    if args.model == "ViT":
        net = net_class(img_size=256, patch_size=8, out_dim=K, mlp_dim=2048)
    elif args.model == "SwinUnet":
        checkpoint = Path(args.swin_checkpoint)
        net = net_class(
            num_classes=K,
            checkpoint=checkpoint,
            img_size=224
        )
    else:
        net = net_class(1, K, kernels=kernels, factor=factor)
        net.init_weights()

    # net = datasets_params[args.dataset]['net'](1, K, kernels=kernels, factor=factor)
    # net.init_weights()
    net.to(device)

    lr = 0.0005
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    # Dataset part
    # B: int = datasets_params[args.dataset]['B']
    B: int = args.batch_size
    root_dir = Path("data") / args.dataset

    swin_img_size = 224 if args.model == "SwinUnet" else None


    train_set = SliceDataset('train',
                            root_dir,
                            img_transform=partial(img_transform, img_size=swin_img_size),
                            gt_transform=partial(gt_transform, K, img_size=swin_img_size),
                            debug=args.debug)
    train_batch_sampler = set_batch_size_and_oversample(
        train_set,
        B,
        args.oversample_foreground_percent,
        args.oversample_small_organ_percent,
    )
    train_loader = DataLoader(train_set,
                              batch_sampler=train_batch_sampler,
                              num_workers=5)

    val_set = SliceDataset('val',
                           root_dir,
                            img_transform=partial(img_transform, img_size=swin_img_size),
                            gt_transform=partial(gt_transform, K, img_size=swin_img_size),
                           debug=args.debug)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=5,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train {args.model} on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    if args.mode == "full":
        idk = list(range(K))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        idk = [0, 1, 3, 4]  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    match args.loss:
        case "ce":
            loss_fn = CrossEntropy(idk=idk)
        # After observing in 3D Slicer, model predicts foregroud very often,
        # resulting in very large organs
        # Tversky loss to explicitly weight the penalty for false positives versus fals negatives
        case "tversky":
            loss_fn = TverskyLoss(idk=idk, alpha=args.alpha, beta=1.0 - args.alpha)
        case _:
            raise ValueError(args.loss)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader) * args.batch_size, K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))
    log_nsd_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))
    log_cldice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0
    best_epoch: int = 0

    for e in range(args.epochs):
        # NSD and clDice are slow so we do it every N epochs.
        slow_metrics: bool = args.metric_every > 0 and (
            (e % args.metric_every == 0) or (e == args.epochs - 1))
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    # print(f"pred_seg size: {pred_seg.size()}")
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                    # only calculate NSD and clDice for validation
                    if m == 'val' and slow_metrics:
                        log_nsd_val[e, j:j + B, :] = nsd_score(pred_seg, gt, METRIC_SPACING_MM)
                        log_cldice_val[e, j:j + B, :] = cldice(pred_seg, gt)   

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if m == 'val' and slow_metrics:
                        postfix_dict |= {"NSD": f"{log_nsd_val[e, :j, 1:].mean():05.3f}",
                                         "clDice": f"{log_cldice_val[e, :j, 1:].mean():05.3f}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)
        np.save(args.dest / "nsd_val.npy", log_nsd_val)
        np.save(args.dest / "cldice_val.npy", log_cldice_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            best_epoch = e
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")

    # Ensure NSD and clDice are available for the best epoch even if it was not a slow_metrics epoch
    best_slow_computed: bool = args.metric_every > 0 and (
        (best_epoch % args.metric_every == 0) or (best_epoch == args.epochs - 1)
    )
    if args.metric_every > 0 and not best_slow_computed and (args.dest / "bestweights.pt").exists():
        print(f"\n>>> Computing NSD and clDice for best epoch ({best_epoch})...")
        net.load_state_dict(torch.load(args.dest / "bestweights.pt", map_location=device))
        net.eval()
        with torch.no_grad():
            j = 0
            for data in val_loader:
                img = data['images'].to(device)
                gt = data['gts'].to(device)
                B = img.shape[0]
                pred_logits = net(img)
                pred_probs = F.softmax(1 * pred_logits, dim=1)
                pred_seg = probs2one_hot(pred_probs)
                log_nsd_val[best_epoch, j:j + B, :] = nsd_score(pred_seg, gt, METRIC_SPACING_MM)
                log_cldice_val[best_epoch, j:j + B, :] = cldice(pred_seg, gt)
                j += B
        np.save(args.dest / "nsd_val.npy", log_nsd_val)
        np.save(args.dest / "cldice_val.npy", log_cldice_val)

    class_names = {
        0: "0 (Background)",
        1: "1 (Esophagus)",
        2: "2 (Heart)",
        3: "3 (Trachea)",
        4: "4 (Aorta)",
    } if K == 5 else {0: "0 (Background)", **{k: f"Class {k}" for k in range(1, K)}}

    sep = "=" * 78
    row_sep = "-" * 20 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12
    table_lines = [
        sep,
        f"{f'Best Epoch Metrics (Epoch {best_epoch})':^78}",
        sep,
        f"{'Class':<20}|{'Val Dice':>12}|{'Val NSD':>12}|{'Val clDice':>12}|{'Train Dice':>12}",
        row_sep,
    ]
    for k in range(K):
        c_name = class_names.get(k, f"Class {k}")
        v_dice = log_dice_val[best_epoch, :, k].mean().item()
        v_nsd = log_nsd_val[best_epoch, :, k].mean().item()
        v_cldice = log_cldice_val[best_epoch, :, k].mean().item()
        t_dice = log_dice_tra[best_epoch, :, k].mean().item()
        table_lines.append(
            f"{c_name:<20}|{v_dice:>12.3f}|{v_nsd:>12.3f}|{v_cldice:>12.3f}|{t_dice:>12.3f}"
        )

    table_lines.append(row_sep)
    fg_label = f"Mean (FG 1..{K - 1})" if K > 2 else "Mean (FG 1)"
    table_lines.append(
        f"{fg_label:<20}|"
        f"{log_dice_val[best_epoch, :, 1:].mean().item():>12.3f}|"
        f"{log_nsd_val[best_epoch, :, 1:].mean().item():>12.3f}|"
        f"{log_cldice_val[best_epoch, :, 1:].mean().item():>12.3f}|"
        f"{log_dice_tra[best_epoch, :, 1:].mean().item():>12.3f}"
    )
    table_lines.append(
        f"{f'Mean (All 0..{K - 1})':<20}|"
        f"{log_dice_val[best_epoch, :, :].mean().item():>12.3f}|"
        f"{log_nsd_val[best_epoch, :, :].mean().item():>12.3f}|"
        f"{log_cldice_val[best_epoch, :, :].mean().item():>12.3f}|"
        f"{log_dice_tra[best_epoch, :, :].mean().item():>12.3f}"
    )
    table_lines.append(row_sep)
    val_loss_best = log_loss_val[best_epoch].mean().item()
    tra_loss_best = log_loss_tra[best_epoch].mean().item()
    table_lines.append(
        f"{'Loss':<20}|  Val: {val_loss_best:<18.2e}|  Train: {tra_loss_best:<18.2e}"
    )
    table_lines.append(sep)
    table_str = "\n".join(table_lines)

    print("\n Training finished. ")
    print(f"\n Best dice: {best_dice}")
    print(f"\n{table_str}")

    with open(args.dest / "best_epoch.txt", 'a') as f:
        f.write(f"\n\n{table_str}\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--swin_checkpoint', type=Path,
                        help="Path to the pretrained Swin-Unet checkpoint")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")
    parser.add_argument('--seed', default=0, type=int)


    parser.add_argument('--metric_every', default=5, type=int,
                        help="Compute the slow metrics (NSD and clDice) every N epochs. ")
    parser.add_argument('--oversample_foreground_percent', default=1 / 3, type=float,
                        help="Fraction of each training batch forced to contain foreground "
                             "(nnU-Net default: 1/3).")
    parser.add_argument('--oversample_small_organ_percent', default=0.0, type=float,
                        help="Fraction of each batch reserved for slices containing "
                             "small-organ classes 1 or 3. This is part of, not in "
                             "addition to, the foreground fraction.")
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--loss', default='ce', choices=['ce', 'tversky'],
                        help="Loss function to use for training.")
    parser.add_argument('--alpha', default=0.3, type=float,
                        help="Weight for false positives in Tversky loss (beta = 1 - alpha).")
    args = parser.parse_args()

    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()
