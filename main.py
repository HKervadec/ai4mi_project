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

import random
import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from functools import partial 

from dataset import SliceDataset
from experiments import EXPERIMENTS, get as get_experiment
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import (CrossEntropy)

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}

def seed_everything(seed: int) -> None:
    """Seed every RNG so a run is reproducible and A/B differences reflect the
    technique, not run-to-run noise (weight init, shuffle order, future
    augmentation). Call before building the network and the dataloaders."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism over speed: reproducible convolutions for fair comparisons.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic, distinct seed. Module-level
    (not a lambda/closure) so it stays picklable under the 'spawn' start method."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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

def build_loss(name: str, K: int, mode: str):
    """Loss factory -- the single place to add new losses later (dice, focal, ...).

    Keeping this here means a new loss is one branch, and every existing
    experiment keeps using exactly the loss it used before.
    """
    if name != "ce":
        raise NotImplementedError(f"loss '{name}' is not implemented yet; add it in build_loss")
    if mode == "full":
        return CrossEntropy(idk=list(range(K)))       # supervise all classes
    if mode == "partial":
        # SEGTHOR-specific (skip heart, class 2). Guard so it can't silently
        # produce a wrong loss on a non-5-class dataset (matches master, which
        # only allowed partial for SEGTHOR).
        if K != 5:
            raise ValueError(f"partial mode is SEGTHOR-specific (K=5), got K={K}")
        return CrossEntropy(idk=[0, 1, 3, 4])
    raise ValueError(f"unknown mode: {mode}")


def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("running on mps macbook")
    else:
        device = torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    # An experiment (if given) supplies the data directory and the network params
    # key; otherwise fall back to the plain --dataset path (unchanged behavior).
    exp = get_experiment(args.experiment) if args.experiment else None
    params_key: str = exp.base_dataset if exp else args.dataset
    params = datasets_params[params_key]

    K: int = params['K']
    kernels: int = params.get('kernels', 8)
    factor: int = params.get('factor', 2)
    net = params['net'](1, K, kernels=kernels, factor=factor)
    net.init_weights()
    net.to(device)

    lr = 0.0005
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    # Dataset part
    B: int = params['B']
    root_dir = Path(exp.data_dir) if exp else Path("data") / args.dataset
    augment: bool = exp.train.augment if exp else False
    print(f">> Using data from {root_dir} (augment={augment})")

    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform= partial(gt_transform, K),
                             augment=augment,
                             debug=args.debug)
    # Deterministic shuffling: a seeded generator fixes the batch order and
    # seed_worker makes each worker's RNG reproducible (matters once augmentation
    # is added). Both keep A/B comparisons fair across runs of the same seed.
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=5,
                              shuffle=True,
                              worker_init_fn=seed_worker,
                              generator=loader_gen)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=partial(gt_transform, K),
                           augment=False,
                           debug=args.debug)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=5,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    exp = get_experiment(args.experiment) if args.experiment else None
    mode: str = exp.train.mode if exp else args.mode
    loss_name: str = exp.train.loss if exp else "ce"
    label: str = args.experiment if exp else args.dataset
    print(f">>> Setting up to train on {label} with mode={mode}, loss={loss_name}, seed={args.seed}")

    seed_everything(args.seed)  # before net init + dataloader construction
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    loss_fn = build_loss(loss_name, K, mode)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0

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

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

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
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--experiment', default=None, choices=list(EXPERIMENTS),
                        help="Named config from experiments.py: selects the data directory, loss, "
                             "augmentation, etc. Takes precedence over --dataset/--mode.")
    parser.add_argument('--dest', type=Path, default=None,
                        help="Destination directory for results (predictions and weights). "
                             "Defaults to results/<experiment> when --experiment is given.")

    parser.add_argument('--seed', default=42, type=int,
                        help="RNG seed for reproducible runs. Vary it (e.g. 42, 43, 44) to "
                             "measure run-to-run variance and average results across seeds.")
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")

    args = parser.parse_args()

    if args.dest is None:
        if args.experiment:
            # Keep each experiment self-contained: results live next to the
            # sliced data, under data/experiments/<name>/results.
            args.dest = Path(get_experiment(args.experiment).data_dir) / "results"
        else:
            parser.error("--dest is required unless --experiment is given")

    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()
