"""Training loop. Mirrors main.py's runTraining (same forward, loss, metric and RNG order, so a seeded run reproduces it exactly) and adds resume, CSV/W&B logging and fault tolerance."""
import csv
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import src.losses  # noqa: F401  registers losses
import src.models  # noqa: F401  registers models
import src.optim  # noqa: F401  registers optimizers / schedulers
from src.checkpoint import load_checkpoint, rng_state, save_checkpoint, set_rng_state
from src.data import build_loader
from src.metrics import epoch_metrics, slice_counts
from src.registry import build
from utils import tqdm_

LOG = logging.getLogger("ai4mi")


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    return build("model", cfg["model"]["name"], in_channels=cfg["data"]["in_channels"],
                 num_classes=cfg["data"]["num_classes"], **cfg["model"]["kwargs"]).to(device)


def run_epoch(split: str, net: nn.Module, loader, loss_fn, optimizer, device, cfg: dict,
              desc: str) -> tuple[float, np.ndarray, list[str]]:
    training = optimizer is not None
    net.train(training)
    losses, counts, stems, failed = [], [], [], 0
    with torch.set_grad_enabled(training):
        for batch in tqdm_(loader, desc=desc, disable=not sys.stdout.isatty()):
            try:
                img, gt = batch["images"].to(device), batch["gts"].to(device)
                if training:
                    optimizer.zero_grad()
                probs = F.softmax(net(img), dim=1)
                loss = loss_fn(probs, gt)
                if training:
                    loss.backward()
                    optimizer.step()
            except RuntimeError as exc:  # CUDA OOM, a malformed slice, ...: skip the batch, bounded
                failed += 1
                LOG.exception("%s batch failed (%d/%d allowed), stems %s", split, failed,
                              cfg["train"]["max_failed_batches"], batch["stems"][:4])
                if training:
                    optimizer.zero_grad(set_to_none=True)
                if failed > cfg["train"]["max_failed_batches"]:
                    raise RuntimeError(f"too many failed {split} batches") from exc
                continue
            losses.append(loss.item())
            counts.append(slice_counts(probs.argmax(dim=1), gt))
            stems.extend(batch["stems"])
    return float(np.mean(losses)), np.concatenate(counts), stems


def write_row(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def truncate_rows(path: Path, before_epoch: int) -> None:
    """Drop CSV rows from epochs that will be re-run after a resume."""
    if not path.exists():
        return
    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["epoch"]) < before_epoch]
    path.unlink()
    for r in rows:
        write_row(path, r)


def overlay_samples(net: nn.Module, dataset, device, n: int = 4):
    """Fixed, evenly spaced val slices so W&B shows the same examples every epoch."""
    idx = np.linspace(0, len(dataset) - 1, n).astype(int)
    items = [dataset[i] for i in idx]
    net.eval()
    with torch.no_grad():
        img = torch.stack([it["images"] for it in items]).to(device)
        pred = net(img).argmax(dim=1).cpu().numpy()
    gts = torch.stack([it["gts"] for it in items]).argmax(dim=1).numpy()
    return img[:, 0].cpu().numpy(), gts, pred, [it["stems"] for it in items]


def fit(cfg: dict, run_dir: Path, device: torch.device, resume: bool, wb) -> int:
    """Train to cfg.train.epochs, resuming from checkpoints/last.pt if asked. Returns the best epoch."""
    net = build_model(cfg, device)
    optimizer = build("optim", cfg["optim"]["name"], params=net.parameters(), **cfg["optim"]["kwargs"])
    scheduler = build("scheduler", cfg["scheduler"]["name"], optimizer=optimizer,
                      epochs=cfg["train"]["epochs"], **cfg["scheduler"]["kwargs"])
    loaders = {split: build_loader(cfg, split, device) for split in ("train", "val")}
    loss_fn = build("loss", cfg["loss"]["name"], num_classes=cfg["data"]["num_classes"],
                    **cfg["loss"]["kwargs"])
    LOG.info("model %s: %.2fM parameters | train %d slices, val %d slices", cfg["model"]["name"],
             sum(p.numel() for p in net.parameters()) / 1e6,
             len(loaders["train"].dataset), len(loaders["val"].dataset))

    ckpt_dir, csv_path = run_dir / "checkpoints", run_dir / "epochs.csv"
    select = cfg["train"]["select_metric"]
    start, best, best_epoch = 0, -math.inf, -1
    if resume:
        state = load_checkpoint(ckpt_dir / "last.pt")
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler:
            scheduler.load_state_dict(state["scheduler"])
        set_rng_state(state["rng"])
        start, best, best_epoch = state["epoch"] + 1, state["best"], state["best_epoch"]
        truncate_rows(csv_path, start)
        LOG.info("resumed after epoch %d (best %s=%.4f at epoch %d)", state["epoch"], select, best, best_epoch)
    elif csv_path.exists():
        csv_path.unlink()  # leftovers from a run that died before its first checkpoint

    for epoch in range(start, cfg["train"]["epochs"]):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch}
        for split, opt in (("train", optimizer), ("val", None)):
            loss, counts, stems = run_epoch(split, net, loaders[split], loss_fn, opt, device, cfg,
                                            f">> {split:5s} ({epoch:4d})")
            row[f"{split}_loss"] = loss
            row |= epoch_metrics(split, counts, stems, cfg)
            if split == "val":
                val_counts, val_stems = counts, stems
        if scheduler:
            scheduler.step()

        improved = row[select] > best or best_epoch < 0  # NaN never improves; epoch 0 always saved
        if improved:
            best, best_epoch = row[select], epoch
            save_checkpoint(ckpt_dir / "best.pt", {"epoch": epoch, "model": net.state_dict(), "config": cfg})
            torch.save(net, ckpt_dir / "best_model.pkl")  # whole pickled net, as the submission asks
            np.savez_compressed(run_dir / "val_counts_best.npz", counts=val_counts, stems=np.array(val_stems))
        row |= {"lr": lr, "seconds": round(time.time() - t0, 1), "is_best": improved}
        write_row(csv_path, row)
        save_checkpoint(ckpt_dir / "last.pt", {
            "epoch": epoch, "model": net.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "best": best, "best_epoch": best_epoch, "rng": rng_state()})

        wb.log(row, step=epoch)
        every = cfg["wandb"]["image_every"]
        if wb.run and every and (epoch % every == 0 or epoch == cfg["train"]["epochs"] - 1):
            wb.log_overlays(*overlay_samples(net, loaders["val"].dataset, device),
                            cfg["data"]["class_names"], step=epoch)
        LOG.info("epoch %3d | train loss %.4f | val loss %.4f | val dice fg %.4f (legacy %.4f)%s | %.0fs",
                 epoch, row["train_loss"], row["val_loss"], row["val_dice_fg"],
                 row["val_dice_legacy_fg"], "  *best*" if improved else "", row["seconds"])

    return best_epoch
