"""Optional Weights & Biases logging. Does nothing unless a WANDB_API_KEY is available.

The key is read from the environment or from a gitignored `.env` file in the repo root
(see `.env.example`). Entity/project live in configs/current.yaml under `wandb:`.
"""

import os
from pathlib import Path

_run = None


def _load_env_file() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def init(cfg, run_dir: Path, run_name: str, run_info: dict, debug: bool = False) -> None:
    global _run
    wcfg = cfg.get("wandb") or {}
    _load_env_file()
    if not wcfg.get("enabled", True) or not os.environ.get("WANDB_API_KEY"):
        print("> W&B logging off (no WANDB_API_KEY in environment or .env)")
        return
    try:
        import wandb
    except ImportError:
        print("> W&B logging off (pip install wandb)")
        return
    tags = [str(t) for t in (cfg.get("owner"), cfg.data.split, f"fold{cfg.data.fold}", "debug" if debug else None) if t]
    try:
        _run = wandb.init(
            entity=wcfg.get("entity"), project=wcfg.get("project", "ai4mi-segthor"),
            group=cfg.experiment, name=f"{cfg.experiment}/{run_name}", tags=tags,
            notes=cfg.get("idea"), dir=str(run_dir), config={**cfg.to_dict(), "_run": run_info},
            settings=wandb.Settings(silent=True),
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
    except Exception as exc:  # never let logging break a training run
        print(f"> W&B init failed, continuing without it: {exc}")
        _run = None


def log_epoch(epoch: int, **metrics) -> None:
    if _run is not None:
        _run.log({"epoch": epoch, **metrics})


def finish(summary: dict, run_dir: Path) -> None:
    global _run
    if _run is None:
        return
    for key, value in summary.items():
        _run.summary[key] = value
    # Not config.yaml: W&B already has the config (wandb.init), and run.save symlinks the file into
    # W&B's files/ folder, where W&B then writes its own config.yaml through the link, overwriting ours.
    for name in ("summary.json", "log.csv"):
        if (run_dir / name).exists():
            _run.save(str(run_dir / name), base_path=str(run_dir), policy="now")
    _run.finish()
    _run = None
