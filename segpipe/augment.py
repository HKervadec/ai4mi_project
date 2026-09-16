"""Data augmentation for the training slices."""

from dataset import augment_pair


class Combined:
    # affine, roll, elastic, brightness, contrast (from branch Testing-data-augmentation);
    # uses the python/torch RNGs seeded per worker instead of rng
    def __call__(self, img, gt, rng):
        return augment_pair(img, gt)


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
