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
import random
import os
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
import wandb

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print(">> Loaded environment variables from .env file")
except ImportError:
    print(">> python-dotenv not installed. Install with: pip install python-dotenv")
    print(">> Or set environment variables manually")

from functools import partial 

from dataset import SliceDataset
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


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across different libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f">> Random seed set to {seed} for reproducibility")


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

def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Initialize wandb
    wandb_mode = "offline" if args.wandb_offline else "online"
    
    # Check if wandb API key is available
    api_key = os.getenv('WANDB_API_KEY')
    if not api_key and wandb_mode == "online":
        print(">> Warning: WANDB_API_KEY not found in environment variables")
        print(">> Switching to offline mode. Set WANDB_API_KEY in .env file for online mode")
        wandb_mode = "offline"
    
    # Set experiment name
    if args.wandb_name:
        experiment_name = args.wandb_name
    else:
        experiment_name = f"{args.dataset}_{args.mode}_{args.epochs}epochs"
    
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=experiment_name,
        mode=wandb_mode,
        config={
            "dataset": args.dataset,
            "mode": args.mode,
            "epochs": args.epochs,
            "gpu": args.gpu,
            "debug": args.debug,
            "dest": str(args.dest),
            "seed": args.seed,
            "experiment_name": experiment_name
        }
    )
    
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
    factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2
    net = datasets_params[args.dataset]['net'](1, K, kernels=kernels, factor=factor)
    net.init_weights()
    net.to(device)

    lr = 0.0005
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Log model architecture and hyperparameters to wandb
    wandb.config.update({
        "learning_rate": lr,
        "optimizer": "Adam",
        "betas": (0.9, 0.999),
        "num_classes": K,
        "kernels": kernels,
        "factor": factor,
        "batch_size": datasets_params[args.dataset]['B'],
        "num_workers": 5,
        "seed": args.seed
    })
    
    # Log model architecture (commented out due to pickle issues with wandb.watch)
    # wandb.watch(net, log="all", log_freq=10)

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

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    if args.mode == "full":
        loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

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

        # Log metrics to wandb
        train_loss_epoch = log_loss_tra[e, :].mean().item()
        val_loss_epoch = log_loss_val[e, :].mean().item()
        train_dice_epoch = log_dice_tra[e, :, 1:].mean().item()  # Exclude background class
        val_dice_epoch = log_dice_val[e, :, 1:].mean().item()   # Exclude background class
        
        # Log per-class dice scores
        wandb_log = {
            "epoch": e,
            "train_loss": train_loss_epoch,
            "val_loss": val_loss_epoch,
            "train_dice": train_dice_epoch,
            "val_dice": val_dice_epoch
        }
        
        # Add per-class dice scores
        for k in range(1, K):  # Skip background class
            wandb_log[f"train_dice_class_{k}"] = log_dice_tra[e, :, k].mean().item()
            wandb_log[f"val_dice_class_{k}"] = log_dice_val[e, :, k].mean().item()
        
        wandb.log(wandb_log, step=e)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            
            # Log best dice improvement to wandb
            wandb.log({"best_dice": current_dice, "best_epoch": e}, step=e)
            
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")
            
            # Save model artifacts to wandb
            wandb.save(str(args.dest / "bestmodel.pkl"))
            wandb.save(str(args.dest / "bestweights.pt"))
    
    # Final logging and artifact saving
    print(f">>> Training completed. Best dice: {best_dice:.3f}")
    wandb.log({"final_best_dice": best_dice}, step=args.epochs-1)
    
    # Create wandb artifacts for the complete experiment
    artifact = wandb.Artifact(
        name=f"model_{args.dataset}_{args.mode}",
        type="model",
        description=f"Best model for {args.dataset} dataset in {args.mode} mode"
    )
    artifact.add_file(str(args.dest / "bestweights.pt"))
    artifact.add_file(str(args.dest / "bestmodel.pkl"))
    artifact.add_file(str(args.dest / "best_epoch.txt"))
    wandb.log_artifact(artifact)
    
    # Create metrics artifact
    metrics_artifact = wandb.Artifact(
        name=f"metrics_{args.dataset}_{args.mode}",
        type="metrics",
        description=f"Training metrics for {args.dataset} dataset in {args.mode} mode"
    )
    metrics_artifact.add_file(str(args.dest / "loss_tra.npy"))
    metrics_artifact.add_file(str(args.dest / "dice_tra.npy"))
    metrics_artifact.add_file(str(args.dest / "loss_val.npy"))
    metrics_artifact.add_file(str(args.dest / "dice_val.npy"))
    wandb.log_artifact(metrics_artifact)
    
    # Finish wandb run
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")
    parser.add_argument('--wandb_project', type=str, default='ai4mi-segthor',
                        help="Wandb project name")
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help="Wandb entity name (optional)")
    parser.add_argument('--wandb_offline', action='store_true',
                        help="Run wandb in offline mode")
    parser.add_argument('--wandb_name', type=str, default=None,
                        help="Custom name for wandb experiment run")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_random_seed(args.seed)

    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()
