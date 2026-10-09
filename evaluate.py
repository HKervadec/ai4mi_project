# Om te zorgen dat dit bestand gelijk het juiste experiment pakt
# Kan je het aanroepen met python evaluate.py --experiment <experiment name>
#
# This summarises the per-epoch 2D overlap metrics (Dice, IoU) logged by the old
# main.py path. The 3D metrics (Dice / HD / HD95 / ASSD / NSD, in mm) live in the
# segpipe pipeline: run.py -> segpipe/evaluate.py.

import numpy as np
import time
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True)
args = parser.parse_args()

BASE = os.path.join("data", "experiments", args.experiment, "results")

dice_val = np.load(BASE + "/dice_val.npy")
iou_val  = np.load(BASE + "/iou_val.npy")

# Background class = 0 → skip
dice_no_bg = dice_val[:, :, 1:]
iou_no_bg  = iou_val[:, :, 1:]

print("=== Validation Metrics (per-epoch 2D overlap) ===")
print(f"Mean Dice (all classes): {dice_no_bg.mean():.4f}")
print(f"Mean IoU  (all classes): {iou_no_bg.mean():.4f}")

# Per class
num_classes = dice_no_bg.shape[-1]

for c in range(num_classes):
    print(f"Class {c+1} Dice: {dice_no_bg[:, :, c].mean():.4f}")
    print(f"Class {c+1} IoU : {iou_no_bg[:, :, c].mean():.4f}")

# Per epoch
print("\n=== Per Epoch ===")
for e in range(dice_no_bg.shape[0]):
    print(f"Epoch {e}: Dice={dice_no_bg[e].mean():.4f}, IoU={iou_no_bg[e].mean():.4f}")

# Save the metrics
timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = os.path.join(BASE, f"metrics_summary_{timestamp}.txt")

with open(filename, "w") as f:
    f.write(f"Mean Dice: {dice_no_bg.mean():.4f}\n")
    f.write(f"Mean IoU : {iou_no_bg.mean():.4f}\n\n")

    for c in range(num_classes):
        f.write(f"Class {c+1} Dice: {dice_no_bg[:, :, c].mean():.4f}\n")
        f.write(f"Class {c+1} IoU : {iou_no_bg[:, :, c].mean():.4f}\n")

    f.write("\nPer Epoch:\n")
    for e in range(dice_no_bg.shape[0]):
        f.write(f"Epoch {e}: Dice={dice_no_bg[e].mean():.4f}, IoU={iou_no_bg[e].mean():.4f}\n")

print(f"\nThe evaluation has been saved as '{os.path.basename(filename)}'")
print(f"Location: {os.path.abspath(filename)}")
