#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

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


from torch import einsum

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

def foreground(idk, include_background: bool) -> list[int]:
    """Positions in ``idk`` to average an overlap loss over: all, or all but class 0 (background)."""
    return [i for i, k in enumerate(idk) if include_background or k != 0]


class DiceLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.eps = kwargs.get('eps', 1e-6)
        # False: average over the organs only; background Dice is ~1 and only dilutes the loss (as nnU-Net)
        self.include_background = kwargs.get('include_background', True)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        # Keep only the classes selected for supervision
        probs = pred_softmax[:, self.idk, ...]
        target = weak_target[:, self.idk, ...].float()

        # Ignore pixels belonging to classes not included in self.idk
        # When all classes are selected, this mask is 1 everywhere
        supervised = target.sum(dim=1, keepdim=True) > 0
        probs = probs * supervised
        target = target * supervised

        # One dice score per selected class, aggregated across batch and pixels.
        intersection = einsum("bkwh,bkwh->k", probs, target)
        denominator = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))

        dice_per_class = (2 * intersection + self.eps) / (denominator + self.eps)
        dice_per_class = dice_per_class[foreground(self.idk, self.include_background)]

        return 1 - dice_per_class.mean()

class TverskyLoss():
    """Multi-class soft Tversky loss over the classes in ``idk``.

    alpha weights false positives; beta weights false negatives.
    """
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.alpha = kwargs.get('alpha', 0.3)
        self.beta = kwargs.get('beta', 0.7)
        self.eps = kwargs.get('eps', 1e-6)
        self.include_background = kwargs.get('include_background', True)

        if self.alpha < 0 or self.beta < 0:
            raise ValueError('alpha and beta must be non-negative')

        print(
            f"Initialized {self.__class__.__name__} with "
            f"idk={self.idk}, alpha={self.alpha}, beta={self.beta}, eps={self.eps}, "
            f"include_background={self.include_background}"
        )

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        # Keep only the selected supervised classes.
        probs = pred_softmax[:, self.idk, ...]
        target = weak_target[:, self.idk, ...].float()

        # Match DiceLoss behaviour exactly for partially supervised masks.
        supervised = target.sum(dim=1, keepdim=True) > 0
        probs = probs * supervised
        target = target * supervised

        # Aggregate each class across the current batch and all spatial pixels.
        true_positive = einsum("bkwh,bkwh->k", probs, target)
        false_positive = einsum("bkwh,bkwh->k", probs, 1 - target)
        false_negative = einsum("bkwh,bkwh->k", 1 - probs, target)

        tversky_per_class = (true_positive + self.eps) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.eps
        )
        tversky_per_class = tversky_per_class[foreground(self.idk, self.include_background)]

        return 1 - tversky_per_class.mean()


class CrossEntropyTversky():
    """CE plus an asymmetric Tversky overlap loss."""
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.alpha = kwargs.get('alpha', 0.3)
        self.beta = kwargs.get('beta', 0.7)
        self.tversky_weight = kwargs.get('tversky_weight', 1.0)
        self.include_background = kwargs.get('include_background', True)

        self.cross_entropy = CrossEntropy(idk=self.idk)
        self.tversky = TverskyLoss(
            idk=self.idk,
            alpha=self.alpha,
            beta=self.beta,
            include_background=self.include_background,
        )

        print(
            f"Initialized {self.__class__.__name__} with "
            f"idk={self.idk}, alpha={self.alpha}, beta={self.beta}, "
            f"tversky_weight={self.tversky_weight}"
        )

    def __call__(self, pred_softmax, weak_target):
        cross_entropy_loss = self.cross_entropy(pred_softmax, weak_target)
        tversky_loss = self.tversky(pred_softmax, weak_target)
        return cross_entropy_loss + self.tversky_weight * tversky_loss

class CrossEntropyDice():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.dice_weight = kwargs.get('dice_weight', 1.0)
        self.include_background = kwargs.get('include_background', True)

        self.cross_entropy = CrossEntropy(idk=self.idk)
        self.dice = DiceLoss(idk=self.idk, include_background=self.include_background)

        print(
            f"Initialized {self.__class__.__name__} "
            f"with idk={self.idk}, dice_weight={self.dice_weight}, include_background={self.include_background}"
        )

    def __call__(self, pred_softmax, weak_target):
        cross_entropy_loss = self.cross_entropy(pred_softmax, weak_target)
        dice_loss = self.dice(pred_softmax, weak_target)
        return cross_entropy_loss + self.dice_weight * dice_loss

class FocalLoss():
    """Focal loss: down-weights easy examples via (1 - p_t)^gamma.
    Lin et al., Focal Loss for Dense Object Detection, ICCV 2017.
    gamma=0 reduces to standard cross-entropy.
    """
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.gamma = kwargs.get('gamma', 2.0)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        probs = pred_softmax[:, self.idk, ...]
        mask = weak_target[:, self.idk, ...].float()

        log_p = (probs + 1e-10).log()
        focal_weight = (1.0 - probs) ** self.gamma

        loss = -einsum("bkwh,bkwh->", mask * focal_weight, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class FocalDice():
    """Focal loss + Dice loss — pixel-level hard-example mining combined with overlap supervision."""
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.gamma = kwargs.get('gamma', 2.0)
        self.dice_weight = kwargs.get('dice_weight', 1.0)
        self.include_background = kwargs.get('include_background', True)

        self.focal = FocalLoss(idk=self.idk, gamma=self.gamma)
        self.dice = DiceLoss(idk=self.idk, include_background=self.include_background)

        print(
            f"Initialized {self.__class__.__name__} with "
            f"idk={self.idk}, gamma={self.gamma}, dice_weight={self.dice_weight}, "
            f"include_background={self.include_background}"
        )

    def __call__(self, pred_softmax, weak_target):
        focal_loss = self.focal(pred_softmax, weak_target)
        dice_loss = self.dice(pred_softmax, weak_target)
        return focal_loss + self.dice_weight * dice_loss


