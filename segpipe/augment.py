"""Data augmentation for the training slices."""

import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T


def per_slice(op, img, factor):
    """Apply an intensity op to every channel on its own.

    torchvision's intensity ops only accept 1- or 3-channel images, and a 2.5D stack
    (input.context_slices > 0) is neither a grayscale slice nor an RGB image. Treating each
    channel as its own grayscale slice keeps them usable, and for the plain 2D case (1 channel)
    it is exactly what they did before.
    """
    return torch.cat([op(channel[None], factor) for channel in img])


class Combined:
    # affine, roll, elastic, brightness, contrast (from branch Testing-data-augmentation).
    # Draws all randomness from the per-worker `rng` seeded in run.py, so augmentation
    # is reproducible for a given train.seed.
    def __call__(self, img, gt, rng):
        # img is now [Batch, Channels, Height, Width]
        # gt is now [Batch, Classes, Height, Width]
        
        # spatial transforms are applied to both image and ground truth
        if rng.random() > 0.5:
            # rotating and scaling
            angle = float(rng.uniform(-10, 10))
            scale = float(rng.uniform(0.9, 1.1))

            img = TF.affine(img, angle=angle, translate=[0, 0], scale=scale, shear=0, interpolation=TF.InterpolationMode.BILINEAR)
            gt = TF.affine(gt, angle=angle, translate=[0, 0], scale=scale, shear=0, interpolation=TF.InterpolationMode.NEAREST)

            # make sure that the ground truth is not empty after the transformation
            empty_pixels = gt.sum(dim=1) == 0
            gt[:, 0][empty_pixels] = 1

        # adding random roll
        if rng.random() > 0.5:
            # get image dimensions
            B, C, H, W = img.shape

            # pick a random shift amount (up to 25% of the image size)
            shift_w = int(rng.integers(-W // 4, W // 4 + 1))
            shift_h = int(rng.integers(-H // 4, H // 4 + 1))

            # roll the image and ground truth (using -2 and -1 to always target H and W)
            img = torch.roll(img, shifts=(shift_h, shift_w), dims=(-2, -1))
            gt = torch.roll(gt, shifts=(shift_h, shift_w), dims=(-2, -1))

        # adding elastic deformation
        if rng.random() > 0.5:
            seed = int(rng.integers(0, 2 ** 31))

            # apply to image
            torch.manual_seed(seed)
            elastic_transform = T.ElasticTransform(alpha=35.0, sigma=5.0, interpolation=TF.InterpolationMode.BILINEAR)
            img = elastic_transform(img)

            # apply to ground truth
            torch.manual_seed(seed)
            elastic_transform_gt = T.ElasticTransform(alpha=35.0, sigma=5.0, interpolation=TF.InterpolationMode.NEAREST)
            gt = elastic_transform_gt(gt)

            # make sure that the ground truth is not empty after the transformation
            empty_pixels = gt.sum(dim=1) == 0
            gt[:, 0][empty_pixels] = 1

        # intensity transforms are applied only to the image
        if rng.random() > 0.5:
            # adjust brightness
            brightness_factor = float(rng.uniform(0.8, 1.2))
            img = per_slice(TF.adjust_brightness, img, brightness_factor)

        if rng.random() > 0.5:
            # adjust contrast
            contrast_factor = float(rng.uniform(0.8, 1.2))
            img = per_slice(TF.adjust_contrast, img, contrast_factor)

        return img, gt


# name -> class with __call__(img, gt, rng) -> (img, gt)
AUGMENTS: dict = {
    "combined": Combined,
}


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, gt, rng):
        for t in self.transforms:
            img, gt = t(img, gt, rng)
        return img, gt


def build_augment(items):
    transforms = []
    for item in items or []:
        params = {"name": item} if isinstance(item, str) else dict(item)
        name = params.pop("name")
        if name not in AUGMENTS:
            raise KeyError(f"unknown augmentation '{name}'. Known: {sorted(AUGMENTS)}")
        transforms.append(AUGMENTS[name](**params))
    return Compose(transforms) if transforms else None
