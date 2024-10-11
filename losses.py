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
        self.weighted = kwargs["weighted"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")
    
    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape

        b, _, h, w = pred_softmax.shape

        if self.weighted:
            alpha = 1
        else:
            # # 1.0 for background and precomputed inverse class frequencies for non-background classes (total_pixels/class_pixels,
            # # where total_pixels is the sum of all pixels that have class 1, 2, 3 or 4, but not 0)
            # alpha = torch.tensor([1.0, 22.3814, 1.3688, 29.9430, 5.2261]).view(1, -1, 1, 1).repeat(b, 1, h, w)

            # empirically test if this helps k=1 (esophagus) perform better
            alpha = torch.tensor([1.0, 5.0, 1.0, 1.0, 1.0]).view(1, -1, 1, 1).repeat(b, 1, h, w)
        p = pred_softmax[:, self.idk, ...]

        log_p = (p[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, alpha * (1 - p) ** self.gamma * (log_p))
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)
