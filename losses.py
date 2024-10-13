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

import torch
from torch import einsum
from torch import Tensor
from utils import simplex, sset


def create_loss_fn(args, K: int):
    # Select the classes we want to supervise.
    if args.mode == 'full':
        idk = list(range(K))        # Supervise both background and foreground
    elif args.mode == 'partial':
        idk = [0, 1, 3, 4]          # Do not supervise the heart (class 2)
    else:
        raise ValueError(f"{args.mode} is not supported as a mode")

    match args.loss:
        case 'CrossEntropy':
            loss_fn = CrossEntropy(idk=idk)
        case 'Dice':
            loss_fn = DiceLoss()
        case 'FocalLoss':
            loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, idk=idk)  # Preference for alpha=0.25 and gamma=2
        case 'CombinedLoss':
            loss_fn = CombinedLoss(alpha=args.alpha, beta=args.beta, idk=idk)  # Pass idk parameter
        case 'FocalDiceLoss':
            # Preference for alpha=0.3, beta=0.5, focal_alpha=0.25 and focal_gamma=2
            # This focuses more on the foreground classes
            loss_fn = FocalDiceLoss(alpha=args.alpha, beta=args.beta, focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma, idk=idk)
        case 'TverskyLoss':
            loss_fn = TverskyLoss(alpha=args.alpha, beta=args.beta)
        case _:
            raise ValueError(f"{args.loss} is not supported as a loss")
    return loss_fn


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class DiceLoss():
    def __init__(self, smooth=1e-5):
        self.smooth = smooth

    def __call__(self, pred_softmax: Tensor, target: Tensor) -> Tensor:
        assert pred_softmax.shape == target.shape
        assert simplex(pred_softmax)
        assert sset(target, [0, 1])

        intersection = torch.sum(pred_softmax * target, dim=(2, 3))
        union = torch.sum(pred_softmax, dim=(2, 3)) + torch.sum(target, dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss():
    def __init__(self, alpha=.25, gamma=2, reduction='mean', **kwargs):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = CrossEntropy(**kwargs)

    def __call__(self, pred_softmax: Tensor, target: Tensor) -> Tensor:
        ce = self.ce_loss(pred_softmax, target)
        pt = torch.exp(-ce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss:
    def __init__(self, alpha=0.5, beta=0.5, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = CrossEntropy(**kwargs)
        self.dice_loss = DiceLoss()

    def __call__(self, pred_softmax: Tensor, target: Tensor) -> Tensor:
        ce = self.ce_loss(pred_softmax, target)
        dice = self.dice_loss(pred_softmax, target)
        return self.alpha * ce + self.beta * dice
    

class FocalDiceLoss:
    def __init__(self, alpha=.3, beta=.7, focal_alpha=.25, focal_gamma=2, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, **kwargs)
        self.dice_loss = DiceLoss()

    def __call__(self, pred_softmax: Tensor, target: Tensor) -> Tensor:
        focal = self.focal_loss(pred_softmax, target)
        dice = self.dice_loss(pred_softmax, target)
        return self.alpha * focal + self.beta * dice
    

class TverskyLoss:
    def __init__(self, alpha=.5, beta=.5, smooth=1e-5):
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def __call__(self, pred_softmax: Tensor, target: Tensor) -> Tensor:
        assert pred_softmax.shape == target.shape
        assert simplex(pred_softmax)
        assert sset(target, [0, 1])

        true_pos = torch.sum(pred_softmax * target, dim=(2, 3))
        false_neg = torch.sum(target * (1 - pred_softmax), dim=(2, 3))
        false_pos = torch.sum(pred_softmax * (1 - target), dim=(2, 3))

        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        return 1 - tversky.mean()