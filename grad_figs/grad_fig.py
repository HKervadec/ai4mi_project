#!/usr/bin/env python3.7


import matplotlib
import matplotlib.pyplot as plt

# import torch
import numpy as np
from matplotlib.colors import SymLogNorm


def class2one_hot(seg: np.ndarray, K: int) -> np.ndarray:
    b, *img_shape = seg.shape
    res = np.zeros((b, K, *img_shape), dtype=np.int64)
    np.put_along_axis(res, seg[:, None, ...], 1, axis=1)

    return res


def noise(arr: np.ndarray) -> np.ndarray:
    # noise_level: int = np.random.randint(30)
    to_add = np.random.normal(-0.5, 0.5, arr.shape)
    # print(to_add.min(), to_add.max())

    return (arr + to_add).clip(0, 1)


if __name__ == "__main__":
    fontsize = 20
    DPI = 200
    matplotlib.rc("font", **{"size": fontsize})
    matplotlib.rc("text", usetex=True)

    gt = np.load("gt.npy")
    img = np.load("img.npy")
    softmax = np.load("softmax.npy")

    W, H = img.shape
    assert img.shape == gt.shape == softmax.shape

    noisy_softmax = noise(softmax)
    del softmax

    fg_pred = noisy_softmax
    bg_pred = 1 - noisy_softmax

    assert fg_pred.shape == gt.shape
    inter = (fg_pred * gt).sum()
    union = fg_pred.sum() + gt.sum()

    n_inter = ((1 - fg_pred) * (1 - gt)).sum()
    n_union = (1 - fg_pred).sum() + (1 - gt).sum()

    dice_grad = -2 * (gt * union - inter) / (union**2) - 2 * (
        (1 - gt) * n_union - n_inter
    ) / (n_union**2)
    dicedb = np.log10(np.abs(dice_grad).max() / np.abs(dice_grad).min())
    print(f"{dicedb=}")

    l2_grad = 2 * (noisy_softmax - gt)

    ce_grad = -(1 - gt) / ((1 - noisy_softmax) + 1e-10) - (gt) / (
        (noisy_softmax) + 1e-10
    )
    cedb = np.log10(np.abs(ce_grad).max() / np.abs(ce_grad).min())
    print(f"{cedb=}")
    print(f"{ce_grad.min()=}, {ce_grad.max()=}")

    # Labels figure with img
    # fig = plt.figure(figsize=(12, 9))
    fig = plt.figure()
    # plt.imshow(img, cmap="gray")
    im = plt.imshow(gt, cmap="jet", interpolation="nearest", alpha=1)
    plt.axis("off")
    plt.colorbar(im, ticks=[0, 1])
    plt.savefig("labels.png", dpi=DPI)
    fig.tight_layout()
    del fig

    # Softmax predictions
    fig = plt.figure()
    im = plt.imshow(fg_pred, cmap="jet", vmin=0, vmax=1, interpolation="nearest")
    plt.axis("off")
    plt.colorbar(im, ticks=[0, 1])
    plt.savefig("softmax.png", dpi=DPI)
    fig.tight_layout()
    del fig

    # DSC grad
    fig = plt.figure()
    im = plt.imshow(
        dice_grad,
        cmap="jet",
        interpolation="nearest",
        vmin=dice_grad.min(),
        vmax=dice_grad.max(),
    )
    plt.axis("off")
    plt.colorbar(im)
    plt.savefig("dsc_grad.png", dpi=DPI)
    fig.tight_layout()
    del fig

    # CE grad
    fig = plt.figure()
    # im = plt.imshow(ce_grad, cmap='jet', interpolation='nearest', vmin=ce_grad.min(), vmax=0)
    im = plt.imshow(
        ce_grad,
        cmap="jet",
        interpolation="nearest",
        norm=SymLogNorm(
            linthresh=0.03,
            vmin=ce_grad.min(),
            # vmax=ce_grad.max(
            vmax=0,
        ),
    )
    plt.axis("off")
    plt.colorbar(im, ticks=[-(10**10), -(10**5), 0, 10**5, 10**10])
    plt.savefig("ce_grad.png", dpi=DPI)
    fig.tight_layout()
    del fig

    # Trimming (only if https://imagemagick.org/index.php is available)
    # call(["mogrify", "-trim", "labels.png",
    #                           "softmax.png",
    #                           "dsc_grad.png",
    #                           "ce_grad.png"])
