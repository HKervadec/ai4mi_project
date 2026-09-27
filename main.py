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
from shutil import copytree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from functools import partial 

from dataset import SliceDataset, parse_segthor_slice_stem
from ShallowNet import shallowCNN
from ENet import ENet
from SwinUNet import SwinUNet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   volume_dice_from_slices,
                   save_images)

from losses import (CrossEntropy, DiceLoss, DiceCELoss)

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}

def img_transform(img):
        img = img.convert('L')
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
        img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
        img = class2one_hot(img, K=K)
        return img[0]


def record_volume_dice(
    log_dice: Tensor,
    epoch: int,
    patient_id: str,
    slices: list[tuple[int, Tensor, Tensor]],
    patient_indexes: dict[str, int],
    K: int,
) -> None:
    assert patient_id in patient_indexes, patient_id
    log_dice[epoch, patient_indexes[patient_id], :] = volume_dice_from_slices(slices, K)


def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int, Any | None]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
    factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2
    match args.architecture:
        case 'baseline':
            net = datasets_params[args.dataset]['net'](1, K, kernels=kernels, factor=factor)
        case 'enet':
            net = ENet(1, K, kernels=kernels, factor=factor)
        case 'swin_unet':
            net = SwinUNet(1, K)
        case _:
            raise ValueError(f"Unsupported architecture: {args.architecture}")
    net.init_weights()
    net.to(device)
    print(f">> Model has {sum(parameter.numel() for parameter in net.parameters()):,} trainable parameters")

    default_lrs = {'adam': 0.0005, 'adamw': 0.0005, 'sgd_nesterov': 0.01}
    lr = args.lr if args.lr is not None else default_lrs[args.optimizer]

    match args.optimizer:
        case 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999),
                                         weight_decay=args.weight_decay)
        case 'adamw':
            optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999),
                                          weight_decay=args.weight_decay)
        case 'sgd_nesterov':
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True,
                                        weight_decay=args.weight_decay)
        case _:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    scheduler = None
    if args.lr_scheduler == 'polynomial':
        def polynomial_decay(completed_epochs: int) -> float:
            return max(0.0, 1 - completed_epochs / args.epochs) ** 0.9

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=polynomial_decay)

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    root_dir = Path("data") / args.dataset



    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform= partial(gt_transform, K),
                             debug=args.debug)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=5,
                              shuffle=True)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=partial(gt_transform, K),
                           debug=args.debug)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=5,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K, scheduler)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K, scheduler = setup(args)
    losses = {'ce': CrossEntropy, 'dice': DiceLoss, 'dicece': DiceCELoss}

    if args.mode == "full": # Supervise both background and foreground
        idk = list(range(K))
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR': # Do not supervise the heart (class 2)
        idk = [0, 1, 3, 4]
    else:
        raise ValueError(args.mode, args.dataset)

    loss_fn = losses[args.loss](idk=idk)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    # SegTHOR validation slices are named Patient_XX_ZZZZ.  The validation
    # loader is not shuffled, so each patient's slices arrive consecutively
    # and can be accumulated one volume at a time.
    compute_3d_dice: bool = args.dataset in ['SEGTHOR', 'SEGTHOR_CLEAN']
    if args.selection_metric == 'dice3d' and not compute_3d_dice:
        raise ValueError('--selection-metric dice3d is only available for SegTHOR datasets')
    val_patient_indexes: dict[str, int] = {}
    log_dice3d_val: Tensor | None = None
    if compute_3d_dice:
        val_patient_ids = sorted({parse_segthor_slice_stem(img_path.stem)[0]
                                  for img_path, _ in val_loader.dataset.files})
        val_patient_indexes = {patient_id: i for i, patient_id in enumerate(val_patient_ids)}
        log_dice3d_val = torch.zeros((args.epochs, len(val_patient_ids), K))

    best_score: float = 0

    for e in range(args.epochs):
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

                    active_patient_id: str | None = None
                    active_patient_slices: list[tuple[int, Tensor, Tensor]] = []
                    completed_patients: set[str] = set()

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
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

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

                        if compute_3d_dice:
                            assert log_dice3d_val is not None
                            gt_class = probs2class(gt)
                            for stem, pred_slice, gt_slice in zip(data['stems'], predicted_class, gt_class):
                                patient_id, slice_id = parse_segthor_slice_stem(stem)
                                if active_patient_id is None:
                                    active_patient_id = patient_id
                                elif patient_id != active_patient_id:
                                    assert active_patient_id not in completed_patients
                                    record_volume_dice(log_dice3d_val, e, active_patient_id,
                                                       active_patient_slices, val_patient_indexes, K)
                                    completed_patients.add(active_patient_id)
                                    active_patient_id = patient_id
                                    active_patient_slices = []

                                active_patient_slices.append((slice_id,
                                                              pred_slice.detach().cpu(),
                                                              gt_slice.detach().cpu()))

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

            if m == 'val' and compute_3d_dice:
                assert log_dice3d_val is not None
                assert active_patient_id is not None
                assert active_patient_id not in completed_patients
                record_volume_dice(log_dice3d_val, e, active_patient_id,
                                   active_patient_slices, val_patient_indexes, K)
                completed_patients.add(active_patient_id)
                assert completed_patients == set(val_patient_indexes)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)
        if log_dice3d_val is not None:
            np.save(args.dest / "dice3d_val.npy", log_dice3d_val)

        dice2d: float = log_dice_val[e, :, 1:].mean().item()
        dice3d: float | None = None
        if log_dice3d_val is not None:
            dice3d = log_dice3d_val[e, :, 1:].mean().item()

        match args.selection_metric:
            case 'dice2d':
                current_score = dice2d
            case 'dice3d':
                if dice3d is None:
                    raise ValueError('--selection-metric dice3d is only available for SegTHOR datasets')
                current_score = dice3d
            case _:
                raise ValueError(f"Unsupported selection metric: {args.selection_metric}")

        if current_score > best_score:
            score_details = f"2D={dice2d:05.3f}"
            if dice3d is not None:
                score_details += f", 3D={dice3d:05.3f}"
            message = (f">>> Improved {args.selection_metric} at epoch {e}: "
                       f"{best_score:05.3f}->{current_score:05.3f} DSC "
                       f"({score_details})")
            print(message)
            best_score = current_score
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            copytree(args.dest / f"iter{e:03d}", Path(best_folder), dirs_exist_ok=True)

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")

        if scheduler is not None:
            scheduler.step()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--loss', default='ce', choices=['ce', 'dice', 'dicece'])
    parser.add_argument('--selection-metric', default='dice2d', choices=['dice2d', 'dice3d'],
                        help='Validation metric used to select and save the best model. '
                             '3D Dice is available for SegTHOR datasets.')
    parser.add_argument('--architecture', default='baseline',
                        choices=['baseline', 'enet', 'swin_unet'],
                        help='Network architecture. The default preserves the dataset-specific baseline network.')
    parser.add_argument('--optimizer', default='adam',
                        choices=['adam', 'adamw', 'sgd_nesterov'],
                        help='Optimizer to use. The default reproduces the original Adam baseline.')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate. Uses the optimizer-specific default when omitted.')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay coefficient. Defaults to 0.0, preserving the baseline.')
    parser.add_argument('--lr-scheduler', default='none', choices=['none', 'polynomial'],
                        help='Learning-rate schedule. The default keeps the learning rate fixed.')
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")

    args = parser.parse_args()

    if args.lr is not None and args.lr <= 0:
        parser.error('--lr must be positive')
    if args.weight_decay < 0:
        parser.error('--weight-decay must be non-negative')

    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()
