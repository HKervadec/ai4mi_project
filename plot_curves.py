#!/usr/bin/env python3
"""Plot loss and patient-level Dice curves; the Dice curve peaks at the best epoch.

Dice here is the patient-pooled 2D Dice that training uses for model selection
(the `val_dice_2d` in summary.json), not the inflated per-slice `dice_val.npy`.

Each argument is a *group*: a run dir, or an experiment dir whose subdirs are seed
runs (holdout-f0-s42, -s43, ...), drawn as mean +/- std. Per run the data comes from
`log.csv` if present, else from the loss_*.npy / dice_*_patient.npy logs.

Usage:
    python plot_curves.py results/aug_gitta
    python plot_curves.py results/enet_tf_stage2 results/current --dest curves.png
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

CLASSES = ["esophagus", "heart", "trachea", "aorta"]


def find_runs(group: Path) -> list[Path]:
    def has_logs(d: Path) -> bool:
        return (d / "log.csv").exists() or (d / "dice_val_patient.npy").exists() or (d / "loss_val.npy").exists()
    if has_logs(group):
        return [group]
    return sorted(d for d in group.iterdir() if d.is_dir() and not d.name.endswith("-debug") and has_logs(d))


def npy_mean(path: Path, skip_bg: bool = False) -> np.ndarray | None:
    if not path.exists():
        return None
    a = np.load(path)
    if a.ndim == 3 and skip_bg:
        a = a[:, :, 1:]
    return a.reshape(a.shape[0], -1).mean(axis=1) if skip_bg or a.ndim < 3 else a.mean(axis=1)


def load_run(run: Path) -> dict[str, np.ndarray]:
    """Per-epoch curves: train_loss, val_loss, val_dice, train_dice, val_dice_<class>."""
    out: dict[str, np.ndarray] = {}
    log = run / "log.csv"
    if log.exists():
        rows = list(csv.DictReader(open(log)))
        for k in rows[0]:
            if k not in ("epoch", "lr", "seconds"):
                out[k] = np.array([float(r[k]) for r in rows])
    for name, f in (("train_loss", "loss_tra"), ("val_loss", "loss_val")):
        if name not in out and (run / f"{f}.npy").exists():
            out[name] = np.load(run / f"{f}.npy").mean(axis=1)
    for name, f in (("val_dice", "dice_val_patient"), ("train_dice", "dice_tra_patient")):
        if (name not in out or name == "train_dice") and (run / f"{f}.npy").exists():
            a = np.load(run / f"{f}.npy")  # (E, patients, K), class 0 = background
            out[name] = a[:, :, 1:].mean(axis=(1, 2))
            if name == "val_dice":
                for k, c in enumerate(CLASSES):
                    out.setdefault(f"val_dice_{c}", a[:, :, k + 1].mean(axis=1))
    return out


def stack(runs: list[dict], key: str) -> np.ndarray | None:
    ys = [r[key] for r in runs if key in r]
    if not ys:
        return None
    E = min(len(y) for y in ys)
    return np.stack([y[:E] for y in ys])  # (seeds, E)


def draw(ax, ys: np.ndarray, color: str, style: str, label: str) -> None:
    x, mean = np.arange(ys.shape[1]), ys.mean(axis=0)
    ax.plot(x, mean, style, color=color, label=label + (f" (n={len(ys)})" if len(ys) > 1 else ""))
    if len(ys) > 1:
        std = ys.std(axis=0)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)


def run(args: argparse.Namespace) -> None:
    groups = {g.name: [load_run(r) for r in find_runs(g)] for g in args.groups}
    groups = {n: r for n, r in groups.items() if r}
    if not groups:
        raise SystemExit("No logs found (need log.csv or .npy files; check paths)")

    per_class = len(groups) == 1 and any(f"val_dice_{CLASSES[0]}" in r for r in next(iter(groups.values())))
    has_loss = any("val_loss" in r or "train_loss" in r for runs in groups.values() for r in runs)
    n = int(has_loss) + 1 + int(per_class)  # drop the loss panel when the run has no loss logs
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes = list(axes[0])
    if not has_loss:
        axes.insert(0, plt.figure().gca())  # throwaway axes so the indices below stay fixed
    titles = ["loss", "val Dice (patient-pooled, classes 1-4)", "val Dice per class"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, runs) in enumerate(groups.items()):
        c = colors[i % len(colors)]
        for key, style, tag, ax in (("train_loss", "--", "train", axes[0]), ("val_loss", "-", "val", axes[0]),
                                    ("train_dice", "--", "train", axes[1]), ("val_dice", "-", "val", axes[1])):
            ys = stack(runs, key)
            if ys is not None:
                draw(ax, ys, c, style, f"{name} {tag}")
        val = stack(runs, "val_dice")
        if val is not None:  # best epoch = argmax of the mean val curve (single run: exactly the saved best epoch)
            best = int(val.mean(axis=0).argmax())
            axes[1].plot(best, val.mean(axis=0)[best], "*", color=c, markersize=12, label=f"{name} best (ep {best}: {val.mean(axis=0)[best]:.3f})")
        if per_class:
            for k, cls in enumerate(CLASSES):
                ys = stack(runs, f"val_dice_{cls}")
                if ys is not None:
                    draw(axes[2], ys, colors[k % len(colors)], "-", cls)

    for ax, t in zip(axes, titles):
        if not has_loss and t == "loss":
            plt.close(ax.figure)
            continue
        ax.set_xlabel("epoch")
        ax.set_title(t)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    if args.dest:
        fig.savefig(args.dest, dpi=150)
        print(f"Saved {args.dest}")
    else:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("groups", type=Path, nargs="+", help="Run dir(s) or experiment dir(s) containing seed runs")
    p.add_argument("--dest", type=Path, default=None, help="Save figure here instead of showing it")
    run(p.parse_args())
