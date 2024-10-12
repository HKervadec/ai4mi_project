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
import torch
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

class FocalLoss():
    def __init__(self, **kwargs):
        self.gamma = kwargs["gamma"]
        self.idk = kwargs["idk"]
        self.weights = kwargs["focal_loss_weights"] # [1.0, 22.3814, 1.3688, 29.9430, 5.2261] (for inv class frequency experiment)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")
    
    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape

        b, _, h, w = pred_softmax.shape

        alpha = torch.tensor(self.weights).view(1, -1, 1, 1).repeat(b, 1, h, w)
        p = pred_softmax[:, self.idk, ...]

        log_p = (p[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, alpha * (1 - p) ** self.gamma * (log_p))
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)

class JaccardLoss():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        # Default smoothing 1 for stability and avoiding division by zero
        self.smooth = kwargs.get("smooth",1)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        pred = pred_softmax[:, self.idk, ...]
        target = weak_target[:, self.idk, ...].float()

        intersection = einsum("bkwh,bkwh->bk", target, pred)
        pred_sum = pred.sum(dim=(2,3))
        target_sum = target.sum(dim=(2,3))

        union = pred_sum + target_sum - intersection - self.smooth
        iou = (intersection + self.smooth) / union

        loss = 1 - iou.mean()
        return loss
class DiceLoss():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        # Default smoothing 1 for stability and avoiding division by zero
        self.smooth = kwargs.get("smooth",1)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        pred = pred_softmax[:, self.idk, ...]
        target = weak_target[:, self.idk, ...].float()

        intersection = einsum("bkwh,bkwh->bk", target, pred)
        pred_sum = pred.sum(dim=(2,3))
        target_sum = target.sum(dim=(2,3))

        if intersection.sum() == 0.0:
            print("test")
            self.smooth = 1e-5
        else:
            self.smooth = 0
        dice_score = 2 * (intersection) / (pred_sum + target_sum)

        loss = 1 - dice_score.mean()
        return loss



class LovaszSoftmaxLoss():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def lovasz_grad(self, gt_sorted):
        """
        Computes the gradient of the Lovasz extension with respect to sorted errors.
        """
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1.0 - intersection / union
        if len(gt_sorted) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    def __call__(self, pred_softmax, target):
        """
        Computes the Lovasz-Softmax loss for one-hot encoded multi-class segmentation.
        :param pred_softmax: softmax probabilities of shape [Batch, Classes, Height, Width]
        :param target: one-hot encoded ground truth of shape [Batch, Classes, Height, Width]
        :return: scalar loss value
        """
        assert pred_softmax.shape == target.shape, "Predictions and targets must have the same shape"

        # Select only the relevant classes using fancy indexing
        pred = pred_softmax[:, self.idk, ...]  # [Batch, Relevant Classes, Height, Width]
        target = target[:, self.idk, ...].float()  # One-hot encoded target

        # Flatten predictions and target for easier manipulation
        B, C, H, W = pred.shape  # Batch, Classes, Height, Width
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)  # Flatten [B*H*W, C]
        target_flat = target.permute(0, 2, 3, 1).reshape(-1, C)  # Flatten [B*H*W, C]

        losses = []
        for c in range(C):  # Iterate over each class in the relevant set (self.idk)
            fg = target_flat[:, c]  # Binary mask for the current class (flattened)
            if fg.sum() == 0:
                continue  # Skip if no ground truth for this class

            # Compute errors (absolute difference between prediction and ground truth mask)
            errors = (fg - pred_flat[:, c]).abs()

            # Sort errors in descending order
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]

            # Compute Lovasz gradient
            jaccard = self.lovasz_grad(fg_sorted)

            # Compute the Lovasz loss for this class
            loss = torch.dot(jaccard, errors_sorted)
            losses.append(loss)

        # Return the mean Lovasz loss over all valid classes
        loss = sum(losses) / len(losses) if losses else torch.tensor(0.0, device=pred.device)
        return loss

class CustomLoss():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        # Default smoothing 1 for stability and avoiding division by zero
        self.smooth = kwargs.get("smooth", 1)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")



    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        pred = pred_softmax[:, self.idk, ...]
        target = weak_target[:, self.idk, ...].float()

        log_p = (pred + 1e-10).log()

        loss = - einsum("bkwh,bkwh->", target, log_p)
        loss /= target.sum() + 1e-10


        intersection = einsum("bkwh,bkwh->bk", target, pred)
        pred_sum = pred.sum(dim=(2, 3))
        target_sum = target.sum(dim=(2, 3))

        if intersection.sum() == 0.0:
            print("test")
            self.smooth = 1e-5
        else:
            self.smooth = 0
        dice_score = 2 * (intersection) / (pred_sum + target_sum)
        dice_loss = 1 - dice_score

        loss = loss * dice_loss

        return loss.mean()