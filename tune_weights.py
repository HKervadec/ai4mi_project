#!/usr/bin/env python3
"""
Per class decision weights.

idea explained:
At the end, the network gives each pixel a score for every class, and we pick
the highest one. Background wins most close calls, because the network has
seen background way more times than the esophagus, so it is
always a little under confident about small organs.

Fix: multiply each class score by a weight before picking a winner.

    winner = argmax( weight * score )

We try to search for good weights on the validation set. No retraining needed.

caveat:
We tune on HALF of the validation patients and report on the OTHER half, so the
numbers we quote aren't just fitted to the data we tuned on.

how to use:
    python tune_weights.py --model results/segthor/base_s0/bestmodel.pkl \
                           --data_dir data/SEGTHOR --K 5 --gpu
"""

import json
import argparse
from pathlib import Path
from functools import partial

import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import SliceDataset
from main import img_transform, gt_transform   # reuse, so inputs always match


def dice_per_class(probs, gts, patients, weights, K):
    """Dice for each class, computed per patient and then averaged."""
    pred = (probs.float() * weights[None, :, None, None]).argmax(dim=1)

    scores = []
    for p in np.unique(patients):
        sel = patients == p
        row = []
        for k in range(K):
            pk, tk = (pred[sel] == k), (gts[sel] == k)
            overlap = (pk & tk).sum().item()
            total = pk.sum().item() + tk.sum().item()
            # organ not in this scan at all -> call it good and move on
            row.append(1.0 if total == 0 else 2 * overlap / total)
        scores.append(row)

    return np.array(scores).mean(axis=0)


def search(probs, gts, patients, K):
    """Try a few weights for one class at a time, keep the one that helps."""
    weights = torch.ones(K)
    best = dice_per_class(probs, gts, patients, weights, K)[1:].mean()
    print(f"starting point (all weights 1): {best:.4f}")

    for _ in range(2):                                    # two passes
        for k in range(1, K):                             # skip background
            for c in [0.8, 1.2, 1.5, 2.0, 3.0, 4.0]:
                trial = weights.clone()
                trial[k] = c
                score = dice_per_class(probs, gts, patients, trial, K)[1:].mean()
                if score > best:
                    best, weights = score, trial
            print(f"  class {k}: weight {weights[k]:.1f} -> {best:.4f}")

    return weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data_dir', type=Path, default=Path("data/SEGTHOR"))
    p.add_argument('--K', type=int, default=5)
    p.add_argument('--gpu', action='store_true')
    args = p.parse_args()

    device = torch.device("cuda") if args.gpu and torch.cuda.is_available() else \
             torch.device("mps") if args.gpu and torch.backends.mps.is_available() else \
             torch.device("cpu")

    net = torch.load(args.model, map_location=device, weights_only=False).to(device)
    net.eval()

    val = SliceDataset('val', args.data_dir, img_transform=img_transform,
                       gt_transform=partial(gt_transform, args.K))
    loader = DataLoader(val, batch_size=8, num_workers=0, shuffle=False)

    # --- run the model once and keep its scores -------------------------- #
    probs, gts, stems = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = net(batch['images'].to(device))
            probs.append(torch.softmax(out, dim=1).cpu().half())
            gts.append(batch['gts'].argmax(dim=1).cpu())
            stems += batch['stems']

    probs, gts = torch.cat(probs), torch.cat(gts)
    patients = np.array(["_".join(s.split("_")[:-1]) or s for s in stems])

    # --- split the patients in half -------------------------------------- #
    names = np.unique(patients)
    half = max(1, len(names) // 2)
    tune, report = np.isin(patients, names[:half]), np.isin(patients, names[half:])
    print(f"tuning on {list(names[:half])}, reporting on {list(names[half:])}\n")

    weights = search(probs[tune], gts[tune], patients[tune], args.K)

    out_file = args.model.parent / "weights.json"
    out_file.write_text(json.dumps(weights.tolist()))
    print(f"\nweights saved to {out_file}")

    # --- honest before/after on the patients we did NOT tune on ----------- #
    if report.any():
        before = dice_per_class(probs[report], gts[report], patients[report],
                                torch.ones(args.K), args.K)
        after = dice_per_class(probs[report], gts[report], patients[report],
                               weights, args.K)

        labels = ['background', 'esophagus', 'heart', 'trachea', 'aorta'] \
            if args.K == 5 else [f"class {k}" for k in range(args.K)]

        print("\n----------- held out patients:")
        for k in range(args.K):
            print(f"{labels[k]:<12}{before[k]:7.4f} -> {after[k]:7.4f}"
                  f"  ({after[k] - before[k]:+.4f})")
        print(f"{'mean (organs)':<12}{before[1:].mean():7.4f} -> "
              f"{after[1:].mean():7.4f}  ({after[1:].mean() - before[1:].mean():+.4f})")


if __name__ == '__main__':
    main()
