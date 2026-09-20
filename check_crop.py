from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.v2 as T

def check_crop(paths, size):
    crop = T.CenterCrop((size, size))
    lost = sum(
        (m := torch.from_numpy(np.array(Image.open(p))) > 0).sum().item() - 
        crop(m.unsqueeze(0)).sum().item()
        for p in paths
    )
    return lost

train_gt = sorted(Path("data/SEGTHOR/train/gt").glob("*.png"))
val_gt = sorted(Path("data/SEGTHOR/val/gt").glob("*.png"))

for sz in [216, 196, 160]:
    print(f"Crop {sz}x{sz} -> Train lost: {check_crop(train_gt, sz)} px ; Val lost: {check_crop(val_gt, sz)} px")