"""Train one configured run.

    python -O -m src.train --config configs/segthor_enet_ce.yaml
    python -O -m src.train --config configs/segthor_enet_ce.yaml --set seed=1 optim.kwargs.lr=0.001
    python -m src.train --config configs/segthor_enet_ce.yaml --smoke     # 2 epochs, 16 slices

Re-running the same command resumes an interrupted run, or exits if it already finished.
"""
import argparse
import csv
import traceback
from pathlib import Path

import torch

from src.checkpoint import seed_everything
from src.config import config_hash, load_config
from src.engine import fit
from src.plots import plot_curves
from src.run import (copy_back, environment, now, prepare_run_dir, read_json, setup_logging,
                     update_manifest, write_json)
from src.wandb_logger import WandbLogger


def summarize(run_dir: Path, cfg: dict, best_epoch: int) -> dict:
    with (run_dir / "epochs.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    best = next(r for r in rows if int(r["epoch"]) == best_epoch)
    manifest = read_json(run_dir / "manifest.json")
    return {"experiment": cfg["experiment"], "seed": cfg["seed"], "run": run_dir.name,
            "config_hash": manifest["config_hash"], "model": cfg["model"]["name"],
            "loss": cfg["loss"]["name"], "optim": cfg["optim"]["name"],
            "lr": cfg["optim"]["kwargs"].get("lr"), "data_root": cfg["data"]["root"],
            "epochs": len(rows), "best_epoch": best_epoch,
            "best": {k: float(v) for k, v in best.items() if k.startswith(("val_", "train_"))},
            "user": manifest["user"], "git_commit": manifest["git_commit"],
            "started": manifest["started"], "finished": now(), "wandb_url": manifest.get("wandb_url")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    parser.add_argument("--smoke", action="store_true", help="tiny run to catch errors before using GPU time")
    parser.add_argument("--force", action="store_true", help="move an existing run aside and start over")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.set, smoke=args.smoke)
    prepared = prepare_run_dir(cfg, force=args.force, smoke=args.smoke)
    if prepared is None:
        print(f"{cfg['experiment']} seed {cfg['seed']} already finished; use --force to re-run")
        return
    run_dir, resume = prepared
    log = setup_logging(run_dir / "train.log")
    device = torch.device("cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")

    manifest = read_json(run_dir / "manifest.json")
    if resume:
        manifest = update_manifest(run_dir, status="running",
                                   resumes=manifest.get("resumes", []) + [environment(device) | {"at": now()}])
    else:
        manifest = update_manifest(run_dir, status="running", started=now(), config_hash=config_hash(cfg),
                                   smoke=args.smoke, **environment(device))
    log.info("%s %s -> %s on %s", "resuming" if resume else "starting", cfg["experiment"], run_dir, device)

    wb = WandbLogger(cfg, run_dir, run_name=f"{cfg['experiment']}/seed{cfg['seed']}",
                     run_id=manifest.get("wandb_id"))
    if wb.id:
        update_manifest(run_dir, wandb_id=wb.id, wandb_url=wb.url)
    seed_everything(cfg["seed"])  # after W&B init, which uses `random` itself; resume restores RNG later
    try:
        best_epoch = fit(cfg, run_dir, device, resume, wb)
    except BaseException as exc:
        log.error("run failed:\n%s", traceback.format_exc())
        update_manifest(run_dir, status="failed", failed_at=now(), error=repr(exc))
        wb.finish(exit_code=1)
        raise

    summary = summarize(run_dir, cfg, best_epoch)
    write_json(run_dir / "summary.json", summary)
    dice_cols = [f"val_dice_{cfg['data']['class_names'][k]}" for k in cfg["eval"]["classes"]] + ["val_dice_fg"]
    plot_curves(run_dir / "epochs.csv", run_dir / "plots" / "curves.png", dice_cols)
    update_manifest(run_dir, status="completed", finished=summary["finished"])
    (run_dir / ".done").touch()
    wb.summary({f"best/{k}": v for k, v in summary["best"].items()} | {"best_epoch": best_epoch})
    wb.finish()
    dest = copy_back(run_dir, cfg)
    log.info("done: best epoch %d, val dice fg %.4f; results in %s%s", best_epoch,
             summary["best"]["val_dice_fg"], run_dir, f", copied to {dest}" if dest else "")


if __name__ == "__main__":
    main()
