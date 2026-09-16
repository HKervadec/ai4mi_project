"""Data augmentation for the training slices."""

# name -> class with __call__(img, gt, rng) -> (img, gt)
AUGMENTS: dict = {}


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
