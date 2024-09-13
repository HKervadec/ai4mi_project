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
import os
import nibabel as nib
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
from skimage.transform import resize

from dataset import SliceDataset
from models.ShallowNet import shallowCNN
from models.ENet import ENet

from utils.losses import CrossEntropy
from utils.metrics import dice_coef, dice_batch
from utils.tensor_utils import (
    Dcm,
    class2one_hot,
    probs2one_hot,
    probs2class,
    tqdm_,
    save_images,
)
import wandb

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
        },
    )

def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    if args.gpu:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(">> Picked MPS (Apple Silicon GPU) to run experiments")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(">> Picked CUDA to run experiments")
        else:
            device = torch.device("cpu")
            print(">> CUDA/MPS not available, falling back to CPU")
    else:
        device = torch.device("cpu")
        print(f">> Picked CPU to run experiments")
    
    K: int = args.datasets_params[args.dataset]["K"]
    net = args.datasets_params[args.dataset]["net"](1, K)
    net.init_weights()
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Dataset part
    B: int = args.datasets_params[args.dataset]["B"] # Batch size
    root_dir = Path("data") / args.dataset

    img_transform = transforms.Compose(
        [
            lambda img: img.convert("L"),
            lambda img: np.array(img)[np.newaxis, ...],
            lambda nd: nd / 255,  # max <= 1
            lambda nd: torch.tensor(nd, dtype=torch.float32),
        ]
    )

    gt_transform = transforms.Compose(
        [
            lambda img: np.array(img)[...],
            # The idea is that the classes are mapped to {0, 255} for binary cases
            # {0, 85, 170, 255} for 4 classes
            # {0, 51, 102, 153, 204, 255} for 6 classes
            # Very sketchy but that works here and that simplifies visualization
            lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # max <= 1
            lambda nd: torch.tensor(nd, dtype=torch.int64)[
                None, ...
            ],  # Add one dimension to simulate batch
            lambda t: class2one_hot(t, K=K),
            itemgetter(0),
        ]
    )

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

    args.dest.mkdir(parents=True, exist_ok=True)

    # For each patient in dataset, get the ground truth volume shape
    gt_shape = {'train': {}, 'val': {}}
    for split, dataset in [('train', train_set), ('val', val_set)]:
        if args.debug:
            split_patient_ids = set([x['stems'].split('_')[1] for x in dataset])
        else:
            split_patient_ids = set([x.split('_')[1] for x in os.listdir(f'{root_dir}/{split}/gt')])
        
        for patient_number in split_patient_ids:
            patient_id = f'Patient_{patient_number}'

            orig_nib = nib.load(f'{args.data_source}/{patient_id}/GT.nii.gz')    
            orig_vol = np.asarray(orig_nib.dataobj)
            gt_shape[split][patient_id] = orig_vol.shape

    return (net, optimizer, device, train_loader, val_loader, K, gt_shape)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K, gt_shape = setup(args)

    if args.mode == "full":
        loss_fn = CrossEntropy(
            idk=list(range(K))
        )  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset in ["SEGTHOR", "SEGTHOR_STUDENTS"]:
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    # The 3d dice log works per patient
    log_dice_3d_tra : Tensor = torch.zeros(((args.epochs, len(gt_shape['train'].keys()), K)))
    log_dice_3d_val : Tensor = torch.zeros(((args.epochs, len(gt_shape['val'].keys()), K)))

    best_dice: float = 0

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

            # Prep for 3D dice computation
            gt_volumes = {p : np.zeros((Z, K, X, Y), dtype=np.int16) 
                for p, (X,Y,Z) in gt_shape[m].items()}
            pred_volumes = {p : np.zeros((Z, K, X, Y), dtype=np.int16) 
                for p, (X,Y,Z) in gt_shape[m].items()}

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data["images"].to(device)
                    gt = data["gts"].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(
                        1 * pred_logits, dim=1
                    )  # 1 is the temperature parameter

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
                        loss.backward()
                        opt.step()

                    # Save predictions and gt slice for 3D Dice computation
                    for i, seg_class in enumerate(pred_seg):
                        stem = data['stems'][i]
                        _, patient_n, z = stem.split('_')
                        patient_id = f'Patient_{patient_n}'

                        X, Y, _ = gt_shape[m][patient_id]
                        
                        resize_and_save_slice(seg_class, K, X, Y, z, pred_volumes[patient_id])
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

                log_dict = {m : {
                        "loss": log_loss[e].mean().item(),
                        "dice": log_dice[e, :, 1:].mean().item(),
                        "dice_class": get_dice_per_class(args, log_dice, K, e),
                    }}

                # Compute 3D Dice   
                if m == 'val':  
                    print('Computing 3D dice...')
                    for i, (patient_id, pred_vol) in tqdm_(enumerate(pred_volumes.items()), total=len(pred_volumes)):
                        gt_vol = torch.from_numpy(gt_volumes[patient_id]).to(device)
                        pred_vol = torch.from_numpy(pred_vol).to(device)

                        dice_3d = dice_batch(gt_vol, pred_vol)
                        log_dice_3d[e, i, :] = dice_3d
                    log_dict["dice_3d"] = log_dice_3d[e, :, 1:].mean().item(), 
                    log_dict["dice_3d_class"] = get_dice_per_class(args, log_dice_3d, K, e)       

                # Log the metrics after each 'e' epoch 
                if not args.disable_wandb:
                    wandb.log(log_dict)  

                # TODO testing, of both 3D Dice (sanity test somehow) and the current wandb logging.
                # try a test run, logging only like 2 epochs in your own testing environment

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
            if not args.disable_wandb:
                wandb.save(str(args.dest / "bestmodel.pkl"))
                wandb.save(str(args.dest / "bestweights.pt"))

def get_dice_per_class(args, log, K, e):
    if args.dataset == 'SEGTHOR': 
        class_names = [(1,'background'), (2,'esophagus'), (3,'heart'), (4,'trachea'), (5,'aorta')]
        dice_per_class = {f"dice_{k}_{n}": log[e, :, k-1].mean().item() for k,n in class_names}
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
    target_arr[int(z),:,:,:] = resized_arr[...]

def get_args():

    # K: Number of classes
    # Avoids the clases with C (often used for the number of Channel)
    datasets_params: dict[str, dict[str, Any]] = {}
    datasets_params["TOY2"] = {"K": 2, "net": shallowCNN, "B": 2}
    datasets_params["SEGTHOR"] = {"K": 5, "net": ENet, "B": 8}

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", 
        default=25, 
        type=int
    )
    parser.add_argument(
        "--dataset", 
        default="TOY2", 
        choices=datasets_params.keys(),
        help="Which dataset to use for the training."
    )
    parser.add_argument(
        "--mode", 
        default="full", 
        choices=["partial", "full"],
        help="Whether to supervise all the classes ('full') or, "
        "only a subset of them ('partial')."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Destination directory to save the results (predictions and weights).",
    )
    parser.add_argument(
        "--data_source",
        type=Path,
        default='./data/segthor_train/train',
        help="The path to get the GT scan, in order to get the correct number of slices"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=0, 
        help="Number of subprocesses to use for data loading. "
        "Default 0 to avoid pickle lambda error"
    )  
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.0005,
        help="Learning rate for the optimizer."
    ) 
    parser.add_argument(
        "--gpu", 
        action="store_true",
        help="Use the GPU if available, otherwise fall back to the CPU."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep only a fraction (10 samples) of the datasets, "
        "to test the logic around epochs and logging easily.",
    )
    parser.add_argument(
        '--disable_wandb',
        action='store_true',
        help='Use flag to disable wandb logging, i.e. for debugging.'
    )
    parser.add_argument(
        '--wandb_project_name',
        type=str,
        help='Project wandb will be logging run to'
    )
    args = parser.parse_args()
    pprint(args)  
    args.datasets_params = datasets_params

    return args

def main():
    args = get_args()
    if not args.disable_wandb:
        setup_wandb(args) 
    runTraining(args)

if __name__ == "__main__":
    main()
