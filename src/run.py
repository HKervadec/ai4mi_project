"""Run directory contract: runs/<experiment>/seed<seed>/ with config, manifest, log and results.

Same config + finished (.done)      -> skip, nothing to do
Same config + unfinished            -> resume automatically from checkpoints/last.pt
Different config in an existing dir -> refuse; rename `experiment` or pass --force
--force                             -> move the old dir aside (never deleted) and start fresh
--smoke                             -> runs/_smoke/...: replaced on every smoke run
"""
import getpass
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

from src.config import REPO, config_hash

LOG = logging.getLogger("ai4mi")
COPY_BACK = ["config.yaml", "manifest.json", "summary.json", "epochs.csv", "eval/metrics_3d.csv"]


class RunConflict(RuntimeError):
    pass


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_dir_for(cfg: dict) -> Path:
    return REPO / cfg["paths"]["run_root"] / cfg["experiment"] / f"seed{cfg['seed']}"


def prepare_run_dir(cfg: dict, force: bool, smoke: bool = False) -> tuple[Path, bool] | None:
    """Returns (run_dir, resume), or None when this exact run already finished."""
    run_dir = run_dir_for(cfg)
    manifest = read_json(run_dir / "manifest.json")
    if run_dir.exists() and smoke:  # throwaway: replace, don't keep old copies
        shutil.rmtree(run_dir)
        manifest = {}
    elif run_dir.exists() and force:
        aside = run_dir.with_name(f"{run_dir.name}.old-{datetime.now():%Y%m%d-%H%M%S}")
        run_dir.rename(aside)
        print(f"--force: moved previous run to {aside}")
        manifest = {}
    elif manifest and manifest["config_hash"] != config_hash(cfg):
        raise RunConflict(f"{run_dir} holds a run with a different config "
                          f"({manifest['config_hash']} != {config_hash(cfg)}). "
                          f"Rename `experiment`, or pass --force to move the old run aside.")
    elif (run_dir / ".done").exists():
        return None
    resume = bool(manifest) and (run_dir / "checkpoints" / "last.pt").exists()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return run_dir, resume


def setup_logging(log_file: Path) -> logging.Logger:
    LOG.setLevel(logging.INFO)
    for handler in LOG.handlers:
        handler.close()
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)):
        handler.setFormatter(fmt)
        LOG.addHandler(handler)
    return LOG


def git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment(device: torch.device) -> dict:
    return {"user": getpass.getuser(), "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "git_commit": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "command": " ".join(sys.argv)}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def update_manifest(run_dir: Path, **fields) -> dict:
    manifest = read_json(run_dir / "manifest.json") | fields
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def copy_back(run_dir: Path, cfg: dict) -> Path | None:
    """Copy the small result files into the git-tracked metrics dir so they survive scratch purges."""
    if read_json(run_dir / "manifest.json").get("smoke"):
        return None
    dest = REPO / cfg["paths"]["metrics_dir"] / cfg["experiment"] / f"seed{cfg['seed']}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in COPY_BACK:
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, dest / Path(name).name)
    return dest


if __name__ == "__main__":  # print where a config's run lives: python -m src.run --config X [--set ..]
    import argparse
    from src.config import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(run_dir_for(load_config(args.config, args.set, smoke=args.smoke)))
