#!/usr/bin/env python3

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
# from torchvision import transforms
from torch.utils.data import DataLoader

from functools import partial 

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from networks3d import Test3DSegmenter
# from ENet3d import ENet3d
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import (CrossEntropy3D)

# 3D imports
import os
from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ToTensord,
    AsDiscreted,
)

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}

datasets_params["segthor_part1"] = {'K': 5, 'net': Test3DSegmenter, 'B': 1, 'kernels': 8, 'factor': 2}

# Configuration
DATA_DIR = "data/segthor_part1/train"
MODEL_SAVE_DIR = "saved_models"
LEARNING_RATE = 1e-4

# Construct 3D image/ground-truth paths
def make_data_list(patient_names):
    data = []
    for patient in patient_names:
        patient_dir = os.path.join(DATA_DIR, patient)

        image_path = os.path.join(
            patient_dir,
            f"{patient}.nii.gz"
        )
        label_path = os.path.join(
            patient_dir,
            "GT.nii.gz"
        )

        # Check that files actually exist
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        if not os.path.exists(label_path):
            raise FileNotFoundError(label_path)
        
        data.append({
            "image": image_path,
            "label": label_path,
        })
    return data

def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
    factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2
    net = datasets_params[args.dataset]['net'](K)#(1, K, kernels=kernels, factor=factor)
    net.init_weights()
    net.to(device)

    lr = LEARNING_RATE
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    # root_dir = Path("data") / args.dataset

    # Define patient split
    train_data = make_data_list([f"Patient_{i:02d}" for i in range(1, 2)])  # Patients 1-16
    val_data = make_data_list([f"Patient_{i:02d}" for i in range(20, 21)])   # Patients 17-20

    print("Training patients:")
    for item in train_data:
        print("  ", item["image"])
    print("\nValidation patients:")
    for item in val_data:
        print("  ", item["image"])

    # Define preprocessing using MONAI
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),           # Load .nii.gz files
        EnsureChannelFirstd(keys=["image", "label"]),  # [D, H, W] -> [1, D, H, W]
        AsDiscreted(keys=["label"], to_onehot=5),      # Class labels to one-hot
        ScaleIntensityd(keys=["image"]),               # Normalize image intensity
        ToTensord(keys=["image", "label"]),            # Convert to PyTorch tensors
    ])

    # Create MONAI datasets
    train_dataset = Dataset(
        data=train_data,
        transform=transforms,
    )
    val_dataset = Dataset(
        data=val_data,
        transform=transforms,
    )

    # Create PyTorch DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=B,
        shuffle=False,
    )

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)

# Training
def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    if args.mode == "full":
        loss_fn = CrossEntropy3D(idk=list(range(K)))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        print("ERROR: mode==partial currently doesn't work for 3D.")
        raise ValueError(args.mode, args.dataset)
        # loss_fn = CrossEntropy3D(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    # log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))  # REPLACE WITH 3D METRICS
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    # log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))  # REPLACE WITH 3D METRICS

    # best_dice: float = 0  # REPLACE WITH 3D METRICS
    best_loss: float = float("inf")

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
                    # log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    # log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['image'].float().to(device)
                    gt = data['label'].long().to(device)
                    # gt = gt.squeeze(1)  # [B, 1, D, H, W] -> [B, D, H, W]

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, D, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # REPLACE WITH 3D METRICS:
                    # Metrics computation, not used for training
                    # pred_seg = probs2one_hot(pred_probs)
                    # log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    # REPLACE WITH 3D VERSION:
                    # if m == 'val':
                    #     with warnings.catch_warnings():
                    #         warnings.filterwarnings('ignore', category=UserWarning)
                    #         predicted_class: Tensor = probs2class(pred_probs)
                    #         mult: int = 63 if K == 5 else (255 / (K - 1))
                    #         save_images(predicted_class * mult,
                    #                     data['stems'],
                    #                     args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    # postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                    #                                 "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    # if K > 2:
                    #     postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                    #                      for k in range(1, K)}
                    postfix_dict: dict[str, str] = {"Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    tq_iter.set_postfix(postfix_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        # np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        # np.save(args.dest / "dice_val.npy", log_dice_val)

        # Replaced "dice" with "loss" in the below code: replace with proper 3D metric
        current_loss: float = log_loss_val[e, :, 1:].mean().item()
        if current_loss > best_loss:
            message = f">>> Improved loss at epoch {e}: {best_loss:05.3f}->{current_loss:05.3f} DSC"
            print(message)
            best_loss = current_loss
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")

    # Save model
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    torch.save(
        net.state_dict(),
        f"{MODEL_SAVE_DIR}/model.pt",
    )

    print("\nTraining finished.")
    print(f"Model saved to {MODEL_SAVE_DIR}/model.pt")

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

    args = parser.parse_args()

    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()
