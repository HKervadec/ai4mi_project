#!/usr/bin/env python3
"""Train one experiment and evaluate it (2D val Dice, 3D metrics).

    python run.py --config configs/experiments/<name>.yaml [--set key=value ...]
"""

import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

from segpipe.config import load_config, save_config
from segpipe.data import K, SliceDataset, build_cache, load_split, n_channels
from segpipe.augment import build_augment
from segpipe.models import build_model
from segpipe.losses import build_loss
from segpipe.optim import build_optimizer, build_scheduler
from segpipe.train import pick_device, seed_everything, train
from segpipe.evaluate import evaluate_3d


def git_info() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "--short", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--debug", action="store_true", help="keep only 10 slices of each dataset")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run folder")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    device = pick_device(args.device)
    run_name = f"{cfg.data.split}-f{cfg.data.fold}-s{cfg.train.seed}" + ("-debug" if args.debug else "")
    run_dir = Path("results") / cfg.experiment / run_name
    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{run_dir} already exists (use --overwrite, or change seed/fold)")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    run_info = {"config_file": args.config, "overrides": args.set, "device": str(device), "debug": args.debug,
                "started": datetime.now().isoformat(timespec="seconds"), "git": git_info()}
    save_config({**cfg.to_dict(), "_run": run_info}, run_dir / "config.yaml")
    print(f">>> Experiment '{cfg.experiment}' -> {run_dir} on {device}")

    build_cache(cfg)
    train_ids, val_ids = load_split(cfg.data.split, cfg.data.fold)
    seed_everything(cfg.train.seed)  # before net init + dataloader construction

    model = build_model(cfg.model, n_channels(cfg), K).to(device)
    loss_fn = build_loss(cfg, K)
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"), cfg.train.epochs)

    train_set = SliceDataset(cfg, train_ids, augment=build_augment(cfg.get("augment")), debug=args.debug)
    val_set = SliceDataset(cfg, val_ids, debug=args.debug)

    train_summary = train(model, loss_fn, optimizer, scheduler, train_set, val_set, cfg, run_dir, device)

    summary = {
        "experiment": cfg.experiment, "run": run_name,
        "owner": cfg.get("owner"), "idea": cfg.get("idea"),
        "split": cfg.data.split, "fold": cfg.data.fold, "seed": cfg.train.seed, "epochs": cfg.train.epochs,
        "debug": args.debug, "device": str(device), "git": run_info["git"],
        **train_summary,
    }

    metrics_3d = cfg.get("eval", {}).get("metrics_3d") or []
    if args.debug or train_summary["best_epoch"] < 0 or not metrics_3d:
        print(">> Skipping 3D evaluation (debug run, no best epoch, or no eval.metrics_3d)")
    else:
        print(f">> 3D evaluation ({', '.join(metrics_3d)}) of best_epoch/val")
        summary["metrics_3d"] = evaluate_3d(run_dir, cfg, val_ids, metrics_3d)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f">>> Best 2D val Dice {summary['val_dice_2d']} at epoch {summary['best_epoch']}")
    for name, m in summary.get("metrics_3d", {}).items():
        print(f">>> 3D {name}: {m['mean']}")
    print(f">>> Done: {run_dir / 'summary.json'}  (then: python compare.py)")


if __name__ == "__main__":
    main()
