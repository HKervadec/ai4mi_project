#!/usr/bin/env python3

"""Side-by-side viewer for the augmentations in src/utils/augmentations.py.

Shows a slice and its labels next to augmented versions of it, to check by eye
that the labels still follow the anatomy. Slices of the same patient get the same
augmentation (except for the noise), see `augment` in augmentations.py.

    augmentation_viewer.py [augs] [data_dir] [n_augs] [seed] [n_save]

    # all augmentations, on the training slices
    uv run python src/scripts/augmentation_viewer.py

    # affine then noise, three random versions per slice, on the validation set
    uv run python src/scripts/augmentation_viewer.py affine,noise data/SEGTHOR/val 3

    # no display (Snellius): write 10 figures to results/augmentations and exit
    uv run python src/scripts/augmentation_viewer.py all data/SEGTHOR/train 1 0 10

Keys: right/left next/previous slice, r new random draw, s save figure, q quit.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms.v2.functional import pil_to_tensor

# So that `utils` resolves when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.augmentations import AUGMENTATIONS, augment

CLASS_NAMES: tuple[str, ...] = ("esophagus", "heart", "trachea", "aorta")
CLASS_COLORS: tuple[str, ...] = ("gold", "crimson", "deepskyblue", "limegreen")
OUT_DIR = Path("results/augmentations")


def load_gt(img_path: Path) -> tv_tensors.Mask:
    # slice_segthor.py stores the class index times 63, to make the PNGs visible
    gt = pil_to_tensor(Image.open(img_path.parents[1] / "gt" / img_path.name))
    return tv_tensors.Mask(gt // 63)


def next_labelled(img_paths: list[Path], position: int, direction: int) -> int:
    """Step to the next slice that is not pure background (most of them are)."""
    for _ in range(len(img_paths)):
        position = (position + direction) % len(img_paths)
        if load_gt(img_paths[position]).any():
            return position
    raise SystemExit("No slice has any labels")


def show(ax, img: torch.Tensor, gt: torch.Tensor | None = None) -> None:
    ax.imshow(img[0], cmap="gray", vmin=0, vmax=1, interpolation="none")
    if gt is not None:
        ax.imshow(
            np.ma.masked_equal(gt[0].numpy(), 0),  # background stays transparent
            cmap=ListedColormap(CLASS_COLORS),
            vmin=0.5,
            vmax=len(CLASS_NAMES) + 0.5,
            alpha=0.45,
            interpolation="none",
        )
    ax.set_xticks([])
    ax.set_yticks([])


def draw(
    fig: Figure, img_path: Path, names: tuple[str, ...], n_augs: int, seed: int
) -> None:
    img = tv_tensors.Image(pil_to_tensor(Image.open(img_path)) / 255)
    gt: tv_tensors.Mask = load_gt(img_path)

    # Row i uses seed + i, so the whole figure is reproducible from its title
    rows = [("original", img, gt)] + [
        (f"augmented #{i}", *augment(img, gt, img_path.stem, names, seed + i))
        for i in range(1, n_augs + 1)
    ]

    fig.clear()
    axes = fig.subplots(len(rows), 2, squeeze=False)
    for (label, row_img, row_gt), (ax_img, ax_gt) in zip(rows, axes):
        show(ax_img, row_img)
        show(ax_gt, row_img, row_gt)
        ax_img.set_ylabel(label)

    axes[0, 0].set_title("image")
    axes[0, 1].set_title("image + labels")
    fig.suptitle(f"{img_path.stem}    [{' + '.join(names)}]    seed={seed}")
    fig.legend(
        handles=[Patch(color=c, label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)],
        loc="lower center",
        ncol=len(CLASS_NAMES),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))


def save(fig: Figure, img_path: Path, names: tuple[str, ...], seed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: Path = OUT_DIR / f"{img_path.stem}_{'+'.join(names)}_{seed}.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


def view_augmentations(
    augs: str = "all",
    data_dir: str | Path = "data/SEGTHOR/train",
    n_augs: int | str = 1,
    seed: int | str = 0,
    n_save: int | str = 0,
) -> None:
    """Browse the slices in data_dir, each next to n_augs augmented versions.

    augs is "all" or a comma-separated list applied in order, e.g. "affine,noise".
    With n_save > 0, write that many figures to results/augmentations and exit
    instead of opening a window.
    """
    names: tuple[str, ...] = (
        tuple(AUGMENTATIONS) if augs == "all" else tuple(augs.split(","))
    )
    unknown: list[str] = [name for name in names if name not in AUGMENTATIONS]
    if unknown:
        raise SystemExit(
            f"Unknown augmentation(s) {unknown}, pick from {list(AUGMENTATIONS)}"
        )
    n_augs, seed, n_save = int(n_augs), int(seed), int(n_save)

    img_paths: list[Path] = sorted(Path(data_dir, "img").glob("*.png"))
    if not img_paths:
        raise SystemExit(
            f"No slices in {Path(data_dir, 'img')}, run slice_segthor.py first"
        )
    # Shuffled, as neighbouring slices look almost the same
    img_paths = [
        img_paths[i] for i in np.random.default_rng(seed).permutation(len(img_paths))
    ]
    print(f">> Found {len(img_paths)} slices in {data_dir}")

    if n_save:
        plt.switch_backend("Agg")
    fig: Figure = plt.figure(figsize=(8, 4 * (1 + n_augs) + 0.6))
    position: int = next_labelled(img_paths, -1, 1)

    if n_save:
        for _ in range(n_save):
            draw(fig, img_paths[position], names, n_augs, seed)
            save(fig, img_paths[position], names, seed)
            position = next_labelled(img_paths, position, 1)
        return

    def on_key(event) -> None:
        nonlocal position, seed
        match event.key:
            case "right":
                position = next_labelled(img_paths, position, 1)
            case "left":
                position = next_labelled(img_paths, position, -1)
            case "r":
                seed = int(np.random.default_rng().integers(2**31))
            case "s":
                save(fig, img_paths[position], names, seed)
                return
            case _:
                return
        draw(fig, img_paths[position], names, n_augs, seed)
        fig.canvas.draw_idle()

    # Free the keys matplotlib binds by default, so they do not also pan or save
    for keymap in ("keymap.back", "keymap.forward", "keymap.home", "keymap.save"):
        plt.rcParams[keymap] = []
    fig.canvas.mpl_connect("key_press_event", on_key)

    print(
        "right/left: next/previous slice | r: new random draw | s: save figure | q: quit"
    )
    draw(fig, img_paths[position], names, n_augs, seed)
    plt.show()


def main() -> None:
    view_augmentations(*sys.argv[1:])


if __name__ == "__main__":
    main()
