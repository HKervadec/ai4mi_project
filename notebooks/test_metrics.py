import autoroot  # noqa

import torch
from src.utils.utils import (
    dice_coef,
    ahd_coef,
    hausdorff_coef,
    hd95_coef,
    nsd_coef,
    biou_coef,
    class2one_hot,
)


def make_disk(shape, center, radius):
    yy, xx = torch.meshgrid(
        torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij"
    )
    return ((yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2).long()


# --- Test 1: identical circles -> should get DSC=1, all distances=0, NSD=1, BIoU=1
gt = make_disk((64, 64), (32, 32), 15)
pred_perfect = gt.clone()

# --- Test 2: circle shifted by 3 pixels -> partial overlap, small distances
pred_shifted = make_disk((64, 64), (35, 32), 15)

# --- Test 3: circle with wrong radius (smaller) -> DSC drops more than NSD would with tolerance
pred_small = make_disk((64, 64), (32, 32), 10)

K = 2  # background + 1 foreground class
for name, pred in [
    ("perfect", pred_perfect),
    ("shifted_3px", pred_shifted),
    ("smaller_radius", pred_small),
]:
    label_oh = class2one_hot(gt.unsqueeze(0), K)  # (1, K, H, W)
    pred_oh = class2one_hot(pred.unsqueeze(0), K)

    label_b = label_oh.bool()
    pred_b = pred_oh.bool()

    print(f"\n--- {name} ---")
    print("DSC:  ", dice_coef(label_oh, pred_oh)[0, 1].item())
    print("AHD:  ", ahd_coef(label_b, pred_b, spacing_mm=(1, 1))[0, 1].item())
    print("HD:   ", hausdorff_coef(label_b, pred_b, spacing_mm=(1, 1))[0, 1].item())
    print("HD95: ", hd95_coef(label_b, pred_b, spacing_mm=(1, 1))[0, 1].item())
    print("NSD:  ", nsd_coef(label_b, pred_b, spacing_mm=(1, 1))[0, 1].item())
    print("BIoU: ", biou_coef(label_b, pred_b)[0, 1].item())
