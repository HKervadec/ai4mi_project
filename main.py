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

import argparse
import multiprocessing
import time
import warnings
from copy import deepcopy
from typing import Any
from pathlib import Path
from pprint import pprint
from shutil import copytree, rmtree
from os import environ

import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

import dataset
from models import *
import utils
from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (Dcm,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   jaccard_coef,
                   average_hausdorff_distance,
                   average_hausdorff_distance_per_class,
                   average_symmetric_surface_distance,
                   compute_precision,
                   compute_recall,
                   visualize_sample,
                   save_images)

from losses import create_loss_fn

import wandb #TODO: remove all wandb instances on final submission
from losses import (CrossEntropy)
import matplotlib.pyplot as plt


datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the clases with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8}


def setup(args) -> tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, Any, DataLoader, DataLoader, int]:
    # Device setup
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    # Initialize model
    K: int = datasets_params[args.dataset]['K']
    net = eval(args.model)(1, K, **vars(args))
    net.init_weights(args)
    net.to(device)

    # Setup of the root folder
    if args.scratch:
        tmpdir = environ["TMPDIR"]
        root_dir = Path(tmpdir+"/data") / args.dataset
    else:
        root_dir = Path("data") / args.dataset

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    base_img_transforms, base_gt_transforms, train_img_transforms, train_gt_transforms = dataset.get_transforms(K)
    # Used to seed dataloader workers, passed in `generator` param
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=train_img_transforms,
                             gt_transform=train_gt_transforms,
                             debug=args.debug,
                             remove_unannotated=args.remove_unannotated)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=args.num_workers,
                              shuffle=True,
                              worker_init_fn=utils.seed_worker,
                              generator=g)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=base_img_transforms,
                           gt_transform=base_gt_transforms,
                           debug=args.debug,
                           remove_unannotated=args.remove_unannotated)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=args.num_workers,
                            shuffle=False,
                            worker_init_fn=utils.seed_worker,
                            generator=g)

    args.dest.mkdir(parents=True, exist_ok=True)

    # Initialize optimizer and (possibly) scheduler
    lr = args.lr
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.lr_weight_decay)
    scheduler = None
    if args.enable_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    return net, optimizer, scheduler, device, train_loader, val_loader, K

#TODO Decide necessity of this?
def skip_empty_masks(gt: Tensor, pred_seg: Tensor) -> bool:
    """
    Returns True if both ground truth & predicted segmentation masks are empty,
    Useful for AHD metric - patients with multiple sliced images.
    """
    return gt.sum().item() == 0 and pred_seg.sum().item() == 0


def run_model(args):

    # This will load the saved model and just run the evaluation on the validation set
    if args.evaluation:
        args.epochs = 1
        modes = ['val']
    # This will run the entire training routine, along with the validation at every iter
    else:
        modes = ['train', 'val']

    utils.seed_everything(args)
    start = time.time()
    print(f">>> Setting up to train on {args.dataset} with {args.model}")
    net, optimizer, scheduler, device, train_loader, val_loader, K = setup(args)

    loss_fn = create_loss_fn(args, K)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    per_batch = lambda data_loader: torch.zeros((args.epochs, len(data_loader)))
    per_sample_and_class = lambda data_loader: torch.zeros((args.epochs, len(data_loader.dataset), K))

    log_loss_tra, log_loss_val = per_batch(train_loader), per_batch(val_loader)
    dice_tra, dice_val = per_sample_and_class(train_loader), per_sample_and_class(val_loader)
    # Extra metrics to monitor training
    precision_tra, precision_val = per_sample_and_class(train_loader), per_sample_and_class(val_loader)
    recall_tra, recall_val = per_sample_and_class(train_loader), per_sample_and_class(val_loader)
    jacc_tra, jacc_val = per_sample_and_class(train_loader), per_sample_and_class(val_loader)
    # Validation metrics (too expensive to apply on every sample during train)
    ahd_val: Tensor = per_sample_and_class(val_loader)
    assd_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset)))

    best_dice: float = 0
    best_metrics = {}

    for e in range(args.epochs):
        for m in modes:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    dice = dice_tra
                    precision = precision_tra
                    recall = recall_tra
                    jaccard = jacc_tra

                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    dice = dice_val
                    precision = precision_val
                    recall = recall_val
                    jaccard = jacc_val

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
                    # One metric value (DSC, Jaccard, Precision, Recall) per sample and per class
                    dice[e, j:j + B, :] = dice_coef(gt, pred_seg)
                    jaccard[e, j:j + B, :] = jaccard_coef(gt, pred_seg)
                    precision[e, j:j + B, :] = compute_precision(gt, pred_seg)
                    recall[e, j:j + B, :] = compute_recall(gt, pred_seg)

                    pred_seg = pred_seg.to(device)
                    gt = gt.to(device)

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()
                        # Apply LR scheduler
                        if scheduler:
                            scheduler.step()
                    if m == 'val':
                        # These metrics are calculated only on the last epoch, on validation set
                        if e == (args.epochs - 1):
                            for batch_idx in range(B):
                                # SanityCheck: Debugging statements
                                # if e == 0 and batch_idx < 3:  #Visualizing for the first 3 batches of  first epoch
                                #     visualize_sample(gt[batch_idx], pred_seg[batch_idx], e, batch_idx)
                                # print(f"Shape of gt[{batch_idx}]: {gt[batch_idx].shape}")
                                # print(f"Shape of pred_seg[{batch_idx}]: {pred_seg[batch_idx].shape}")

                                assert gt[batch_idx].shape == pred_seg[
                                    batch_idx].shape, "Shape mismatch between GT and prediction"  # Check that the shapes are identical

                                # Computing AHD per class for each batch
                                ahd_values_per_class = average_hausdorff_distance_per_class(gt[batch_idx],
                                                                                            pred_seg[batch_idx], K)
                                for k in range(K):
                                    if ahd_values_per_class[k] != float('inf'):  # Skipping 'inf' values
                                        ahd_val[e, j + batch_idx, k] = ahd_values_per_class[k]

                                # ahd_value = average_hausdorff_distance(gt[batch_idx], pred_seg[batch_idx]) #Old AHD metric
                                # log_ahd[e, j + batch_idx, :] = ahd_value
                                assd_sample_value = average_symmetric_surface_distance(gt[batch_idx], pred_seg[batch_idx])
                                assd_val[e, j + batch_idx] = assd_sample_value

                        # Save the predictions for the validation set
                        if not args.dry_run:
                            with warnings.catch_warnings():
                                warnings.filterwarnings('ignore', category=UserWarning)
                                predicted_class: Tensor = probs2class(pred_probs)
                                mult: int = 63 if K == 5 else (255 / (K - 1))
                                save_images(predicted_class * mult,
                                            data['stems'],
                                            args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}

                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        metrics = utils.save_loss_and_metrics(K, e, args.dest,
                                              loss=[log_loss_tra, log_loss_val],
                                              dice=[dice_tra, dice_val],
                                              jaccard=[jacc_tra, jacc_val],
                                              precision=[precision_tra, precision_val],
                                              recall=[recall_tra, recall_val],
                                              ahd_validation=ahd_val,
                                              assd_validation=assd_val)
        wandb.log(metrics)

        current_dice: float = dice_val[e, :, 1:].mean().item()
        if args.evaluation:
            print(f">>> Evaluation Dice avg on validation: {current_dice:05.3f}")
        elif current_dice > best_dice:
            print(f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC")
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                    f.write(str(e))

            if not args.dry_run:
                best_folder = args.dest / "best_epoch"
                if best_folder.exists():
                        rmtree(best_folder)
                copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            best_weights_path = args.dest / "bestweights.pt"
            torch.save(net.state_dict(), best_weights_path)
            utils.wandb_save_model(args.disable_wandb, best_weights_path, {'Epoch': e, 'Dice Validation Avg': best_dice})
            best_metrics = metrics

    for key, value in best_metrics.items():
        wandb.run.summary[key] = value
    end = time.time()
    print(f"[FINISHED] Duration: {(end - start):0.2f} s")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--dataset', default='SEGTHOR', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights). "
                             "If in evaluation mode, then this is the directory where the results are saved.")
    parser.add_argument('--seed', default=42, type=int, help='Random seed to use for reproducibility of the experiments')

    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logic around epochs and logging easily.")
    parser.add_argument('--evaluation', action='store_true',
                        help='Will load the model from the dest_results and evaluate it '
                             'on the validation set, with all the available metrics.')

    parser.add_argument('--dropoutRate', type=float, default=0.2, help="Dropout rate for the ENet model")
    parser.add_argument('--lr', type=float, default=0.0005, help="Learning rate")
    parser.add_argument('--lr_weight_decay', type=float, default=0.1, help="Weight decay factor for the AdamW optimizer")
    parser.add_argument('--enable_lr_scheduler', action='store_true')

    parser.add_argument('--alpha', type=float, default=0.5, help="Alpha parameter for loss functions")
    parser.add_argument('--beta', type=float, default=0.5, help="Beta parameter for loss functions")
    parser.add_argument('--focal_alpha', type=float, default=0.25, help="Alpha parameter for Focal Loss")
    parser.add_argument('--focal_gamma', type=float, default=2.0, help="Gamma parameter for Focal Loss")

    # Optimize snellius batch job
    parser.add_argument('--scratch', action='store_true', help="Use the scratch folder of snellius")
    parser.add_argument('--dry_run', action='store_true', help="Disable saving the image validation results on every epoch")
    parser.add_argument('--disable_wandb', action='store_true', help="Disable the WandB logging")
    parser.add_argument('--run_on_mac', action='store_true', help="If code runs on mac cpu, some extra configuration needs to be done")

    # Arguments for more flexibility of the run
    parser.add_argument('--remove_unannotated', action='store_true', help="Remove the unannotated images")
    parser.add_argument('--loss', default='CrossEntropy', choices=['CrossEntropy', 'Dice', 'FocalLoss', 'CombinedLoss', 'FocalDiceLoss', 'TverskyLoss'])
    parser.add_argument('--model', type=str, default='ENet', choices=['ENet', 'shallowCNN', 'UNet', 'UNetPlusPlus', 'DeepLabV3Plus'])
    parser.add_argument('--run_prefix', type=str, default='', help='Name to prepend to the run name')
    parser.add_argument('--run_group', type=str, default=None, help='Your name so that the run can be grouped by it')

    # Arguments for running with different backbones
    parser.add_argument('--encoder_name', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--unfreeze_enc_last_n_layers', type=int, default=1, help="Train the last n layers of the encoder")

    args = parser.parse_args()
    run_name = utils.get_run_name(args, parser)
    if not args.evaluation:
        args.dest = args.dest / run_name
    else:
        # Disable WandB by default if in evaluation mode
        args.disable_wandb = True

    if args.run_on_mac:
        # Added since for python 3.8+, OS X multiprocessing starts processes with spawn instead of fork
        # see https://github.com/pytest-dev/pytest-flask/issues/104
        multiprocessing.set_start_method("fork") #TODO remove on final submission

    utils.wandb_login(args.disable_wandb)
    wandb.init(
        entity="ai_4_mi",
        project="SegTHOR",
        name=run_name,
        config=vars(args),
        mode="disabled" if args.disable_wandb else "online",
        group=args.run_group
    )

    print(f">> {run_name} <<")
    pprint(args)
    run_model(args)


if __name__ == '__main__':
    main()
