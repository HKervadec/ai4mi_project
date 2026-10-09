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

import torch
import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter

from functools import partial 

from dataset import SliceDataset, n_input_channels
from ShallowNet import shallowCNN
from ENet import ENet
from ViT import ViT
try:
    from swin_model import build_swin_unet
except ImportError:  # swin_model.py not committed yet
    build_swin_unet = None
from ViT_Sil import ViT_Sil
from UNet import UNet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   nsd_score,
                   cldice,
                   save_images)

from losses import CrossEntropy, GeneralizedDiceLoss, CrossEntropyDice

METRIC_SPACING_MM = (500 / 256, 500 / 256)

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["TOTALSEG"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}

# Architectures, decoupled from the dataset: --model overrides the dataset default. The per-dataset
# 'net' above stays the default, so existing job scripts keep training the ENet baseline unchanged.
models: dict[str, Any] = {'enet': ENet, 'unet': UNet, 'shallow': shallowCNN, 'vit': ViT, 'vit_sil': ViT_Sil}
if build_swin_unet is not None:
    models['swin'] = build_swin_unet

def img_transform_original(img, pixel_spacing_mm= None):
        img = img.convert('L')
        img = np.array(img)[np.newaxis, ...]
        img = img / 255  # max <= 1
        img = torch.tensor(img, dtype=torch.float32)
        return img

def img_transform(img, pixel_spacing_mm=None):
        ## Default preprocessing
        # img = img.convert('L')
        # img = np.array(img)[np.newaxis, ...]
        # img = img / 255  # max <= 1

        img = np.array(img.convert('L'), dtype=np.uint8)

        ## Preprocessing
        # Gaussian filtering
        #img = gaussian_filter(img, sigma=0.5)
        #img = np.clip(img, 0, 255).astype(np.uint8)
        # CLAHE
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        # Denoising
        img = cv.fastNlMeansDenoising(img, None, 50, 7, 21)
        img = cv.fastNlMeansDenoising(img, None, 20, 7, 21)
        # Edge sharpening
        sharpen_kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]], dtype=np.float32)
        img = cv.filter2D(img, -1, sharpen_kernel)
        # Opening and closing
        morph_kernel = np.ones((3, 3), dtype=np.uint8)
        img = cv.morphologyEx(img, cv.MORPH_OPEN, morph_kernel)
        #img = cv.morphologyEx(img, cv.MORPH_CLOSE, morph_kernel)

        # Pixel space normalization
        

        # Normalize and add the model's channel dimension.
        img = img.astype(np.float32) / 255
        img = img[None, ...]

        img = torch.tensor(img, dtype=torch.float32)
        return img

img_transforms: dict[str, Any] = {'clean': img_transform,
                                  'original': img_transform_original}

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
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    params: dict[str, Any] = datasets_params[args.dataset]
    K: int = params['K']
    # --model/--kernels/--factor override the per-dataset defaults, so the same dataset can be
    # trained with either backbone without editing datasets_params.
    net_class = models[args.model] if args.model else params['net']
    kernels: int = args.kernels if args.kernels is not None else params.get('kernels', 8)
    factor: int = args.factor if args.factor is not None else params.get('factor', 2)
    in_channels: int = n_input_channels(args.neighbours, args.coords, args.fourier_freqs)

    if args.model == 'swin':
        # Factory function with its own signature; ignores in_channels/kernels/factor
        # and loads its own pretrained checkpoint.
        assert in_channels == 1, "Swin-Unet expects a single input channel"
        net = build_swin_unet(num_classes=K, checkpoint=Path(args.swin_checkpoint), img_size=256)
    else:
        net = net_class(in_channels, K, kernels=kernels, factor=factor)
        net.init_weights()
    print(f">> Network input channels: {in_channels}")
    if args.load_weights:
        # Fine-tuning: start from a previous run (e.g. the TOTALSEG pretraining) instead of the random
        # init above. Loading is strict, so a mismatch in K/kernels/factor fails here rather than silently.
        net.load_state_dict(torch.load(args.load_weights, map_location='cpu'))
        print(f">> Loaded weights from {args.load_weights}")
    net.to(device)

    lr = 0.0005
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    root_dir = Path("data") / args.dataset

    transform_fn = img_transforms[args.img_transform]
    print(f">> Using '{args.img_transform}' image preprocessing")

    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=transform_fn,
                             gt_transform= partial(gt_transform, K),
                             debug=args.debug,
                             neighbours=args.neighbours,
                             coords=args.coords,
                             fourier_freqs=args.fourier_freqs)

    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=5,
                              shuffle=True)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=transform_fn,
                           gt_transform=partial(gt_transform, K),
                           debug=args.debug,
                           neighbours=args.neighbours,
                           coords=args.coords,
                           fourier_freqs=args.fourier_freqs)

    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=5,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
   
    print(f">>> Setting up to train {args.model} on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    if args.mode == "full":
        if args.loss == "ce":
            loss_fn = CrossEntropy(
                idk=list(range(K))
            )

        elif args.loss == "generalized_dice":
            loss_fn = GeneralizedDiceLoss(
                idk=list(range(K))
            )

        elif args.loss == "ce_dice":
            loss_fn = CrossEntropyDice(
                ce_idk=list(range(K)),
                dice_idk=list(range(1, K))
            )

    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        loss_fn = CrossEntropy(
            idk=[0, 1, 3, 4]
        )

    else:
        raise ValueError(args.mode, args.dataset)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))
    log_nsd_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))
    log_cldice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0

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
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")

    print("\n Training finished. ")
    print(f"\n Best dice: {best_dice}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--swin_checkpoint', type=Path,
                        help="Path to the pretrained Swin-Unet checkpoint")
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--neighbours', default=0, type=int,
                        help="2.5D: use 2n+1 slices as input channels.")
    parser.add_argument('--coords', default='none',
                        choices=['none', 'z', 'xy', 'xyz'],
                        help="Add coordinate channels (CoordConv).")
    parser.add_argument('--fourier_freqs', default=0, type=int,
                        help="Replace each coordinate ramp with sin/cos pairs.")
    parser.add_argument('--model', default=None, choices=list(models.keys()),
                        help="Override the dataset's default architecture.")
    parser.add_argument('--kernels', type=int, default=None,
                        help="Base channel count, overriding the per-dataset default. "
                             "UNet wants 64 for the widths of the paper; 8 is the ENet baseline.")
    parser.add_argument('--factor', type=int, default=None,
                        help="Channel growth per level for UNet, projection factor for ENet.")
    parser.add_argument('--load_weights', type=Path, default=None,
                        help="bestweights.pt to initialize the network with, instead of a random init")
    parser.add_argument('--img_transform', default='clean', choices=list(img_transforms.keys()),
                        help="Added option between preprocessed image transform and the old original one for testing.")
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")
    parser.add_argument('--metric_every', default=5, type=int,
                        help="Compute the slow metrics (NSD and clDice) every N epochs. ")
    parser.add_argument('--seed', default=0, type=int) 

    parser.add_argument(
     '--loss',
     default='ce',
     choices=['ce', 'generalized_dice', 'ce_dice'],
     help='Loss function for full supervision.'
 )

    args = parser.parse_args()

    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()
