"""Splits, the slice cache and the slice dataset."""

import json
import shutil
import argparse
from pathlib import Path
from functools import partial

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info

from main import img_transform, gt_transform
from slice_segthor import slice_patient
from utils import tqdm_

CLASS_NAMES: list[str] = ["background", "esophagus", "heart", "trachea", "aorta"]
K: int = len(CLASS_NAMES)
SPLITS_DIR = Path("splits")


def load_split(name: str, fold: int) -> tuple[list[str], list[str]]:
    data = json.loads((SPLITS_DIR / f"{name}.json").read_text())
    if not 0 <= fold < len(data["folds"]):
        raise ValueError(f"split '{name}' has {len(data['folds'])} fold(s), got fold={fold}")
    train, val = data["folds"][fold]["train"], data["folds"][fold]["val"]
    assert not set(train) & set(val), f"train/val overlap in split {name} fold {fold}"
    return train, val


# All patients sliced once with slice_segthor.slice_patient into <cache>/{img,gt}/Patient_XX_zzzz.png,
# so a split only selects patients and never requires re-slicing.
def cache_dir(cfg) -> Path:
    h, w = cfg.data.shape
    window = cfg.data.get("window")
    intensity = "minmax" if window is None else f"window{window[0]:g}_{window[1]:g}"
    # target_spacing changes the pixels (resample + crop/pad instead of resize), so it
    # must be part of the cache key -- otherwise a resampled and a non-resampled run
    # with the same gt/window/shape would share a folder and reuse the wrong slices.
    spacing = cfg.data.get("target_spacing")
    spatial = f"{h}x{w}" if spacing is None else f"sp{spacing[0]:g}_{spacing[1]:g}_{spacing[2]:g}_{h}x{w}"
    # Extra windows add img1/, img2/, ... folders, so they also get their own cache folder.
    for level, width in extra_windows(cfg):
        intensity += f"+window{level:g}_{width:g}"
    return Path(cfg.data.get("cache_root", "data/cache")) / f"{Path(cfg.data.gt).name}_{intensity}_{spatial}"


def extra_windows(cfg) -> tuple[tuple[float, float], ...]:
    # data.extra_windows: [[level, width], ...] -> one extra input channel per window
    return tuple(tuple(w) for w in cfg.data.get("extra_windows") or [])


def build_cache(cfg) -> Path:
    dest = cache_dir(cfg)
    if (dest / "done.json").exists():
        return dest

    src = Path(cfg.data.gt)
    assert (src / "train").exists(), f"{src}/train not found. Build the GT first, e.g.: make {cfg.data.gt}"
    window = tuple(cfg.data.window) if cfg.data.get("window") is not None else None
    target_spacing = tuple(cfg.data.target_spacing) if cfg.data.get("target_spacing") is not None else None
    tmp = dest.with_name(dest.name + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"> Building cache {dest} from {src}")

    patients = sorted(p.name for p in (src / "train").iterdir() if p.is_dir())
    for pid in tqdm_(patients):
        # slice_patient returns the voxel spacing; evaluate_3d re-reads it from each
        # GT header when scoring, so there's nothing to persist here.
        slice_patient(pid, dest_path=tmp, source_path=src, shape=tuple(cfg.data.shape),
                      window=window, target_spacing=target_spacing, extra_windows=extra_windows(cfg))
    (tmp / "done.json").write_text(json.dumps({"gt": str(cfg.data.gt), "window": window,
                                                "extra_windows": extra_windows(cfg),
                                                "target_spacing": target_spacing,
                                                "shape": list(cfg.data.shape), "patients": patients}, indent=2))
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return dest


def context_slices(cfg) -> int:
    """2.5D: how many neighbouring slices to stack on *each* side of the center slice (0 = plain 2D)."""
    return int((cfg.get("input") or {}).get("context_slices", 0) or 0)


def n_channels(cfg) -> int:
    # (2*context+1) slices, each with 1 + len(extra_windows) HU windows
    return (2 * context_slices(cfg) + 1) * (1 + len(extra_windows(cfg)))


def parse_stem(stem: str) -> tuple[str, int]:
    """'Patient_01_0123' -> ('Patient_01', 123). The cache writes this name in slice_patient."""
    pid, _, idz = stem.rpartition("_")
    return pid, int(idz)


class SliceDataset(Dataset):
    def __init__(self, cfg, patient_ids: list[str], augment=None, debug: bool = False):
        root = cache_dir(cfg)
        assert (root / "done.json").exists(), f"cache {root} missing: python -m segpipe.data --config <config>"

        wanted = set(patient_ids)
        images = sorted(p for p in (root / "img").glob("*.png") if p.stem.rsplit("_", 1)[0] in wanted)
        self.files = [(p, root / "gt" / p.name) for p in images]
        # Channel order per slice: img/ (data.window), then img1/, img2/, ... (data.extra_windows)
        self.channel_dirs = [root / "img"] + [root / f"img{i}" for i in range(1, len(extra_windows(cfg)) + 1)]

        # 2.5D: a sample is the center slice plus `context` neighbours on each side, stacked as
        # input channels (the GT stays the center slice only, so the task is unchanged). The z range
        # is taken from the *full* volume, before --debug truncates the sample list.
        self.context = context_slices(cfg)
        self.z_range: dict[str, tuple[int, int]] = {}
        for path in images:
            pid, idz = parse_stem(path.stem)
            lo, hi = self.z_range.get(pid, (idz, idz))
            self.z_range[pid] = (min(lo, idz), max(hi, idz))

        if debug:
            self.files = self.files[:10]

        self.gt_transform = partial(gt_transform, K)
        self.augment = augment
        self._rng = None
        print(f">> Dataset: {len(patient_ids)} patients, {len(self.files)} slices, "
              f"{n_channels(cfg)} input channel(s)")

    def __len__(self) -> int:
        return len(self.files)

    def _get_rng(self) -> np.random.Generator:
        # One generator per worker, seeded from torch (seeded in run.py), so augmentation is reproducible
        if self._rng is None:
            info = get_worker_info()
            seed = info.seed if info is not None else int(torch.randint(0, 2 ** 31, ()).item())
            self._rng = np.random.default_rng(seed % 2 ** 32)
        return self._rng

    def _load_slice(self, name: str) -> torch.Tensor:
        """One slice with all its HU windows: (1 + len(extra_windows)) x H x W."""
        return torch.cat([img_transform(Image.open(d / name)) for d in self.channel_dirs])

    def _load_stack(self, img_path: Path) -> torch.Tensor:
        """The slice itself, or the 2.5D stack of 2*context+1 slices (center slice in the middle),
        each slice contributing its window channels -> n_channels(cfg) x H x W.

        At the top/bottom of a volume there is no neighbour, so the edge slice is repeated
        (clamping) -- every sample keeps the same number of channels as the network expects.
        """
        if not self.context:
            return self._load_slice(img_path.name)

        pid, idz = parse_stem(img_path.stem)
        lo, hi = self.z_range[pid]
        neighbours = [min(max(idz + offset, lo), hi) for offset in range(-self.context, self.context + 1)]
        return torch.cat([self._load_slice(f"{pid}_{z:04d}.png") for z in neighbours])

    def __getitem__(self, index: int) -> dict:
        img_path, gt_path = self.files[index]
        img = self._load_stack(img_path)
        gt = self.gt_transform(Image.open(gt_path))

        if self.augment is not None:
            img, gt = self.augment(img, gt, self._get_rng())

        return {"images": img, "gts": gt, "stems": img_path.stem}


def main() -> None:
    from segpipe.config import load_config

    parser = argparse.ArgumentParser(description="Build the slice cache for a config")
    parser.add_argument("--config", default="configs/current.yaml")
    print(f"> Cache ready: {build_cache(load_config(parser.parse_args().config))}")


if __name__ == "__main__":
    main()
