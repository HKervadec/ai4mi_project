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


class DiceLoss():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        self.smooth: float = kwargs['smooth'] if 'smooth' in kwargs else 1e-8
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        pred = pred_softmax[:, self.idk, ...]
        mask = weak_target[:, self.idk, ...].float()

        # Summed over the batch and not per image: most slices contain no
        # esophagus, and a per-image Dice scores an absent class as a perfect 1
        # with no gradient, so the small classes end up supervised on almost
        # nothing.
        inter = einsum("bkwh,bkwh->k", pred, mask)
        sizes = einsum("bkwh->k", pred) + einsum("bkwh->k", mask)

        dice = (2 * inter + self.smooth) / (sizes + self.smooth)

        return 1 - dice.mean()


class CEDice():
    def __init__(self, **kwargs):
        self.alpha: float = kwargs['alpha'] if 'alpha' in kwargs else 0.5
        self.ce = CrossEntropy(idk=kwargs['idk'])
        # Background stays supervised by the cross-entropy, but is kept out of
        # the Dice: at ~99% of the pixels it would drown the organs it is there
        # to rebalance.
        self.dice = DiceLoss(idk=[k for k in kwargs['idk'] if k != 0])

    def __call__(self, pred_softmax, weak_target):
        return (self.alpha * self.ce(pred_softmax, weak_target)
                + (1 - self.alpha) * self.dice(pred_softmax, weak_target))


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)

class WeightedCrossEntropy():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.weights = kwargs['weights']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        weights = self.weights[self.idk].to(pred_softmax.device)
        weights = weights.view(1, -1, 1, 1)

        weighted_mask = mask * weights

        loss = -einsum("bkwh,bkwh->", weighted_mask, log_p)
        loss /= weighted_mask.sum() + 1e-10

        return loss
