"""Training loop (from runTraining in main.py)."""

import csv
import time
import random
import warnings
from pathlib import Path
from shutil import copytree, rmtree

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from segpipe.data import CLASS_NAMES, K
from utils import dice_coef, probs2class, probs2one_hot, save_images, tqdm_


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(model, loss_fn, optimizer, scheduler, train_set, val_set, cfg, run_dir: Path, device) -> dict:
    tc = cfg.train
    E = tc.epochs
    generator = torch.Generator()
    generator.manual_seed(tc.seed)
    loaders = {
        "train": DataLoader(train_set, batch_size=tc.batch_size, num_workers=tc.num_workers, shuffle=True,
                            worker_init_fn=seed_worker, generator=generator),
        "val": DataLoader(val_set, batch_size=tc.batch_size, num_workers=tc.num_workers, shuffle=False),
    }
    logs = {
        "train": (torch.zeros((E, len(loaders["train"]))), torch.zeros((E, len(train_set), K))),
        "val": (torch.zeros((E, len(loaders["val"]))), torch.zeros((E, len(val_set), K))),
    }

    csv_path = run_dir / "log.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_dice"]
                               + [f"val_dice_{n}" for n in CLASS_NAMES[1:]] + ["lr", "seconds"])

    best_dice: float = 0
    best_epoch: int = -1
    start = time.time()
    for e in range(E):
        epoch_start = time.time()
        for m in ["train", "val"]:
            is_train = m == "train"
            model.train(is_train)
            log_loss, log_dice = logs[m]
            loader = loaders[m]
            desc = f">> Training   ({e: 4d})" if is_train else f">> Validation ({e: 4d})"

            with torch.set_grad_enabled(is_train):
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data["images"].to(device)
                    gt = data["gts"].to(device)

                    if is_train:
                        optimizer.zero_grad()

                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = model(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)

                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()

                    if is_train:
                        loss.backward()
                        optimizer.step()

                    if not is_train:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            predicted_class = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult, data["stems"], run_dir / f"iter{e:03d}" / m)

                    j += B
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}" for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        lr = optimizer.param_groups[-1]["lr"]
        if scheduler is not None:
            scheduler.step()

        np.save(run_dir / "loss_tra.npy", logs["train"][0])
        np.save(run_dir / "dice_tra.npy", logs["train"][1])
        np.save(run_dir / "loss_val.npy", logs["val"][0])
        np.save(run_dir / "dice_val.npy", logs["val"][1])

        val_dice_per_class = logs["val"][1][e, :, 1:].mean(dim=0)
        current_dice: float = logs["val"][1][e, :, 1:].mean().item()
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([e, f"{logs['train'][0][e].mean():.5f}", f"{logs['val'][0][e].mean():.5f}",
                                    f"{current_dice:.4f}"] + [f"{v:.4f}" for v in val_dice_per_class.tolist()]
                                   + [f"{lr:.3e}", f"{time.time() - epoch_start:.0f}"])

        if current_dice > best_dice:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice, best_epoch = current_dice, e
            with open(run_dir / "best_epoch.txt", "w") as f:
                f.write(message)

            best_folder = run_dir / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(run_dir / f"iter{e:03d}", best_folder)

            torch.save(model, run_dir / "bestmodel.pkl")
            torch.save(model.state_dict(), run_dir / "bestweights.pt")

    best_per_class = logs["val"][1][best_epoch, :, 1:].mean(dim=0).tolist() if best_epoch >= 0 else [None] * (K - 1)
    return {"best_epoch": best_epoch,
            "val_dice_2d": round(best_dice, 4),
            "val_dice_2d_per_class": {n: (round(v, 4) if v is not None else None)
                                      for n, v in zip(CLASS_NAMES[1:], best_per_class)},
            "train_minutes": round((time.time() - start) / 60, 1)}
