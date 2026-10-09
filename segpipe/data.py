"""Splits, the slice cache and the slice dataset."""

import json
import shutil
import argparse
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info

from main import img_transform
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
    """Slice every patient once into cache_dir(cfg); safe when several jobs start at the same time.

    Each call builds into its own temporary folder and moves it into place with one atomic
    rename, so a finished cache is never deleted or overwritten while another job reads it.
    Jobs that start together may each build a copy; the first to finish wins, the rest discard
    theirs. (Before, all jobs shared one _tmp folder and deleted each other's files.)
    """
    dest = cache_dir(cfg)
    if (dest / "done.json").exists():
        return dest

    src = Path(cfg.data.gt)
    assert (src / "train").exists(), f"{src}/train not found. Build the GT first, e.g.: make {cfg.data.gt}"
    window = tuple(cfg.data.window) if cfg.data.get("window") is not None else None
    target_spacing = tuple(cfg.data.target_spacing) if cfg.data.get("target_spacing") is not None else None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=dest.name + "_tmp_", dir=dest.parent))
    print(f"> Building cache {dest} from {src} (in {tmp.name})")

    try:
        patients = sorted(p.name for p in (src / "train").iterdir() if p.is_dir())
        for pid in tqdm_(patients):
            # slice_patient returns the voxel spacing; evaluate_3d re-reads it from each
            # GT header when scoring, so there's nothing to persist here.
            slice_patient(pid, dest_path=tmp, source_path=src, shape=tuple(cfg.data.shape),
                          window=window, target_spacing=target_spacing, extra_windows=extra_windows(cfg))
        # Slices per patient, so SliceDataset can refuse a cache that lost files.
        counts = Counter(parse_stem(p.stem)[0] for p in (tmp / "img").glob("*.png"))
        (tmp / "done.json").write_text(json.dumps({"gt": str(cfg.data.gt), "window": window,
                                                    "extra_windows": extra_windows(cfg),
                                                    "target_spacing": target_spacing,
                                                    "shape": list(cfg.data.shape), "patients": patients,
                                                    "slices": {pid: counts[pid] for pid in patients}}, indent=2))
        if (dest / "done.json").exists():  # another job finished first
            print(f"> Cache {dest} was built by another job meanwhile, using that one")
            return dest
        try:
            tmp.rename(dest)
        except OSError:
            if (dest / "done.json").exists():  # another job renamed its copy into place just before us
                return dest
            # Never delete it here: it may be another job's cache appearing at this moment.
            raise SystemExit(f"{dest} exists without done.json (left by a crashed or older build): "
                             f"delete it and run again")
        return dest
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def context_slices(cfg) -> int:
    """2.5D: how many neighbouring slices to stack on *each* side of the center slice (0 = plain 2D)."""
    return int((cfg.get("input") or {}).get("context_slices", 0) or 0)


def n_channels(cfg) -> int:
    # (2*context+1) slices, each with 1 + len(extra_windows) HU windows
    return (2 * context_slices(cfg) + 1) * (1 + len(extra_windows(cfg)))


def one_hot(labels: torch.Tensor, K: int) -> torch.Tensor:
    """(B, H, W) class map -> (B, K, H, W) int32 one-hot, the same output as utils.class2one_hot
    without its torch.unique sanity checks (the label range is checked when the slice is loaded)."""
    return torch.zeros((labels.shape[0], K, *labels.shape[1:]), dtype=torch.int32,
                       device=labels.device).scatter_(1, labels[:, None].long(), 1)


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
        self._check_complete(root, patient_ids, images)
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

        self.augment = augment
        self._rng = None
        print(f">> Dataset: {len(patient_ids)} patients, {len(self.files)} slices, "
              f"{n_channels(cfg)} input channel(s)")

    @staticmethod
    def _check_complete(root: Path, patient_ids: list[str], images: list[Path]) -> None:
        """Fail fast on a damaged cache instead of silently training on part of the data."""
        found = Counter(parse_stem(p.stem)[0] for p in images)
        missing = sorted(set(patient_ids) - set(found))
        assert not missing, f"cache {root} has no slices for {missing}: delete it and rebuild"
        expected = json.loads((root / "done.json").read_text()).get("slices")
        if expected is None:  # cache built before slice counts were recorded
            return
        wrong = {pid: (found[pid], expected[pid]) for pid in patient_ids if found[pid] != expected[pid]}
        assert not wrong, f"cache {root} is incomplete, (found, expected) slices: {wrong}: delete it and rebuild"
        no_gt = [p.name for p in images if not (root / "gt" / p.name).exists()]
        assert not no_gt, f"cache {root} is missing {len(no_gt)} GT slices, e.g. {no_gt[:3]}: delete it and rebuild"

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
        # The GT is returned as a uint8 class map (H x W, values 0..K-1); train() one-hot encodes it
        # on the device. Encoding here (main.gt_transform) cost ~100 ms per slice in the workers and
        # sent a 20x larger int32 tensor to the main process.
        labels = np.array(Image.open(gt_path)) // 63  # the cache stores class k as k * 63
        assert labels.max() < K, (gt_path, labels.max())
        gt = torch.from_numpy(labels)

        if self.augment is not None:
            img, gt = self.augment(img, one_hot(gt[None], K)[0], self._get_rng())
            gt = gt.argmax(dim=0).to(torch.uint8)

        return {"images": img, "gts": gt, "stems": img_path.stem}


def slice_weights(dataset: SliceDataset, empty_weight: float = 1.0, class_weights=None) -> torch.Tensor:
    """Sampling weight per slice for train.sampler, from the organs in each slice's GT.

    A slice without any organ gets `empty_weight`; a slice with organs gets the largest
    `class_weights` entry among the organs it contains (1.0 for organs not listed).
    `class_weights` keys are class indices or names, e.g. {esophagus: 2.0}.
    """
    by_class = {(CLASS_NAMES.index(k) if isinstance(k, str) else int(k)): float(v)
                for k, v in (class_weights or {}).items()}
    present = np.zeros((len(dataset.files), K), dtype=bool)  # slice -> which classes its GT contains
    for i, (_, gt_path) in enumerate(tqdm_(dataset.files, desc=">> Sampler weights")):
        present[i] = np.bincount(np.asarray(Image.open(gt_path)).ravel() // 63, minlength=K)[:K] > 0

    empty = ~present[:, 1:].any(axis=1)
    weights = np.ones(len(present))
    for k, w in by_class.items():
        weights[present[:, k]] = np.maximum(weights[present[:, k]], w)
    weights[empty] = empty_weight

    # Share of the slices vs. share of the draws, to check the setting does what was intended
    drawn = lambda mask: weights[mask].sum() / weights.sum()
    print(f">> Sampler: empty slices {empty.mean():.1%} of the data -> {drawn(empty):.1%} of the draws; "
          + ", ".join(f"{CLASS_NAMES[k]} {present[:, k].mean():.1%} -> {drawn(present[:, k]):.1%}"
                      for k in range(1, K)))
    return torch.from_numpy(weights)


def main() -> None:
    from segpipe.config import load_config

    parser = argparse.ArgumentParser(description="Build the slice cache for a config")
    parser.add_argument("--config", default="configs/current.yaml")
    print(f"> Cache ready: {build_cache(load_config(parser.parse_args().config))}")


if __name__ == "__main__":
    main()
