"""Paired augmentations for 2D CT slices and their label maps, from torchvision.

Pass the image as a float ``tv_tensors.Image`` in [0, 1] and the labels as a
``tv_tensors.Mask``. Geometric transforms then move the labels along with the
image (nearest-neighbour for the mask, so no new class values appear), and
intensity transforms leave the labels untouched.
"""

import zlib
from collections.abc import Sequence

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


def random_gamma(
    img: torch.Tensor, gamma_range: tuple[float, float] = (0.7, 1.5)
) -> torch.Tensor:
    # torchvision only offers a fixed gamma, so draw one here
    return F.adjust_gamma(img, torch.empty(1).uniform_(*gamma_range).item())


AUGMENTATIONS: dict[str, v2.Transform] = {
    "affine": v2.RandomAffine(
        degrees=10,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
        interpolation=v2.InterpolationMode.BILINEAR,
    ),
    # alpha: strength, sigma: smoothness; ~3px mean / ~10px max shift on 256x256
    "elastic": v2.ElasticTransform(alpha=250.0, sigma=10.0),
    "gamma": v2.Lambda(random_gamma, tv_tensors.Image),
    "brightness_contrast": v2.ColorJitter(brightness=0.15, contrast=0.15),
    "noise": v2.GaussianNoise(sigma=0.03),
}

# Scanner noise differs from one slice to the next, while the patient's position,
# anatomy and the windowing are shared by the whole scan.
# (but since we are still doing only 2d it doesn't matter that much)
PER_SLICE: set[str] = {"noise"}


def augment(
    img: tv_tensors.Image,
    gt: tv_tensors.Mask,
    stem: str,
    names: Sequence[str] = tuple(AUGMENTATIONS),
    seed: int = 0,
) -> tuple[tv_tensors.Image, tv_tensors.Mask]:
    """Augment one slice the same way as the other slices of its patient.

    stem is the slice name given by slice_segthor.py, e.g. "Patient_03_0042".
    The random draws are seeded by the patient part of it, so every slice of a
    patient gets the same warp and intensity change; the ones in PER_SLICE are
    seeded by the full stem instead, and applied last. Change seed (e.g. every
    epoch) to get new draws.
    """
    patient: str = stem.rsplit("_", 1)[0]
    shared: list[str] = [name for name in names if name not in PER_SLICE]
    per_slice: list[str] = [name for name in names if name in PER_SLICE]

    for key, group in [(patient, shared), (stem, per_slice)]:
        torch.manual_seed(zlib.crc32(f"{seed}/{key}".encode()))
        for name in group:
            img, gt = AUGMENTATIONS[name](img, gt)

    return img, gt
