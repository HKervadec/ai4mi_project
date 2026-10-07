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

    Works on a batch [B, C, H, W]: each channel is passed as a [B, 1, H, W] batch of grayscale
    slices, so the per-image statistics (e.g. contrast mean) are unchanged.
    """
    return torch.cat([op(img[:, c:c + 1], factor) for c in range(img.shape[1])], dim=1)


def fill_background(gt):
    """Mark GT pixels without any class (moved in from outside the image) as background.

    torch.where instead of a boolean-mask assignment (gt[:, 0][empty] = 1), which crashes at
    random on MPS with older torch versions.
    """
    empty_pixels = gt.sum(dim=1) == 0
    background = torch.where(empty_pixels, torch.ones_like(gt[:, 0]), gt[:, 0])
    return torch.cat([background[:, None], gt[:, 1:]], dim=1)


class Combined:
    # affine, roll, elastic, brightness, contrast (from branch Testing-data-augmentation).
    # Draws all randomness from `rng` (seeded from train.seed in train.py), so augmentation
    # is reproducible for a given train.seed.
    # Default: one random draw per batch, so every slice in a batch gets the same transform.
    # per_sample=True: an independent draw for every slice in the batch (separate experiment).
    def __init__(self, per_sample: bool = False):
        self.per_sample = per_sample

    def __call__(self, img, gt, rng):
        if not self.per_sample:
            return self._apply(img, gt, rng)
        pairs = [self._apply(img[b:b + 1], gt[b:b + 1], rng) for b in range(img.shape[0])]
        return torch.cat([i for i, _ in pairs]), torch.cat([g for _, g in pairs])

    def _apply(self, img, gt, rng):
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
            gt = fill_background(gt)

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

            # apply the same displacement (same seed) to the ground truth, with fill=None: torchvision's
            # fill does a boolean-mask assignment that crashes at random on MPS with older torch
            # versions ("shape mismatch: value tensor of shape [...]"). Pixels from outside the
            # image become 0 either way, and fill_background below makes them background.
            torch.manual_seed(seed)
            displacement = T.ElasticTransform.get_params([35.0, 35.0], [5.0, 5.0], list(gt.shape[-2:]))
            gt = TF.elastic_transform(gt, displacement, TF.InterpolationMode.NEAREST, fill=None)

            # make sure that the ground truth is not empty after the transformation
            gt = fill_background(gt)

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
