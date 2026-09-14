"""Data loaders built from the `data` config section, reusing the original SliceDataset.

Extension point — online augmentation: register a factory under kind "augment" returning a callable
(image (1,H,W) float, gt (K,H,W) one-hot) -> (image, gt), then list it in `data.augment`:
    augment: [{name: random_flip, kwargs: {p: 0.5}}]
Augmentations run on the train split only. Offline preprocessing (HU windowing, resampling, ...)
belongs in the slicing step: write a new sliced dataset and point `data.root` at it.
"""
import random
import re
from functools import partial

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset import SliceDataset
from src.config import REPO
from src.registry import build
from utils import class2one_hot


def img_transform(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("L"))[np.newaxis, ...] / 255  # (1, H, W) in [0, 1]
    return torch.tensor(arr, dtype=torch.float32)


def gt_transform(num_classes: int, label_scale: int, img: Image.Image) -> torch.Tensor:
    labels = torch.tensor(np.array(img) // label_scale, dtype=torch.int64)[None, ...]
    return class2one_hot(labels, K=num_classes)[0]  # (K, H, W)


class Augmented(Dataset):
    def __init__(self, base: SliceDataset, augments: list):
        self.base, self.augments = base, augments

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        for aug in self.augments:
            item["images"], item["gts"] = aug(item["images"], item["gts"])
        return item


def seed_worker(worker_id: int) -> None:
    # torch seeds each worker deterministically; propagate that to numpy/random for augmentations
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_dataset(cfg: dict, split: str) -> Dataset:
    d = cfg["data"]
    dataset = SliceDataset(split, REPO / d["root"], img_transform=img_transform,
                           gt_transform=partial(gt_transform, d["num_classes"], d["label_scale"]))
    if cfg["train"]["debug_samples"]:
        dataset.files = dataset.files[:cfg["train"]["debug_samples"]]
    if split == "train" and d["augment"]:
        dataset = Augmented(dataset, [build("augment", a["name"], **a.get("kwargs", {}))
                                      for a in d["augment"]])
    return dataset


def build_loader(cfg: dict, split: str, device: torch.device) -> DataLoader:
    d = cfg["data"]
    return DataLoader(build_dataset(cfg, split), batch_size=d["batch_size"],
                      num_workers=d["num_workers"], shuffle=split == "train",
                      worker_init_fn=seed_worker, pin_memory=device.type == "cuda")


def patient_of(stem: str, regex: str) -> str:
    match = re.fullmatch(regex, stem)
    if match is None:
        raise ValueError(f"slice '{stem}' does not match data.patient_regex '{regex}'")
    return match.group(1)
