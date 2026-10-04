#!/usr/bin/env python3
"""Plot training vs validation loss (and Dice) curves for one or more run dirs.

Usage:
    python plot_curves.py results/current/holdout-f0-s44
    python plot_curves.py runA runB --dest curves.png

Each run dir must contain loss_tra.npy / loss_val.npy (and optionally
dice_tra.npy / dice_val.npy) as saved by main.py, with shape (epochs, batches[, K]).
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def per_epoch(path: Path) -> np.ndarray:
    a = np.load(path)
    if a.ndim == 3:  # (E, N, K): average over classes, skipping background (k=0)
        a = a[:, :, 1:].mean(axis=2) if a.shape[2] > 1 else a[:, :, 0]
    return a.mean(axis=1)


def run(args: argparse.Namespace) -> None:
    metrics = [m for m in ("loss", "dice") if any((d / f"{m}_val.npy").exists() for d in args.runs)]
    if not metrics:
        raise SystemExit(f"No loss_val.npy / dice_val.npy found in: {', '.join(map(str, args.runs))} "
                         f"(check the path, it is relative to the current directory)")
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)

    for ax, m in zip(axes[0], metrics):
        for d in args.runs:
            for split, style in (("tra", "--"), ("val", "-")):
                f = d / f"{m}_{split}.npy"
                if not f.exists():
                    continue
                y = per_epoch(f)
                ax.plot(np.arange(len(y)), y, style, label=f"{d.name} {'train' if split == 'tra' else 'val'}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(m)
        ax.set_title(m)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    if args.dest:
        fig.savefig(args.dest, dpi=150)
        print(f"Saved {args.dest}")
    else:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+", help="Run dir(s) containing the .npy logs")
    p.add_argument("--dest", type=Path, default=None, help="Save figure here instead of showing it")
    run(p.parse_args())
