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
from torch.nn import functional as F
from utils import simplex, sset

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
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, pred_softmax: Tensor, targets: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(pred_softmax, targets.argmax(dim=1), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

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
    

class CombinedLossWithFocal:
    def __init__(self, alpha=0.5, beta=0.5, gamma=2, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = FocalLoss(gamma=gamma)
        self.dice_loss = DiceLoss()

    def __call__(self, pred_softmax: Tensor, target: Tensor) -> Tensor:
        ce = self.ce_loss(pred_softmax, target)
        dice = self.dice_loss(pred_softmax, target)
        return self.alpha * ce + self.beta * dice
    

class TverskyLoss:
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5):
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