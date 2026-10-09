#!/usr/bin/env python3
"""Post-processing experiment: rescore another experiment's saved predictions, no training.

    python postprocess_run.py --config configs/experiments/<name>.yaml [--set key=value ...]

The config names the experiment whose predictions to reuse (`source: current`) and the
post-processing (`eval.postprocess`). The source run is picked like run.py names runs
(data.split, data.fold, train.seed), so `--set train.seed=43` rescores the seed-43 run.
Results go to results/<experiment>/<run>/ like any other run, so compare.py lists them
with Δ against `current`; the 2D val Dice and training fields are the source run's.
"""

import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

from run import git_info
from segpipe.config import load_config, save_config
from segpipe.data import load_split
from segpipe.evaluate import evaluate_3d
from segpipe.postprocess import build_postprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore a run's predictions with post-processing")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run folder")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    source = cfg.get("source")
    postprocess = cfg.get("eval", {}).get("postprocess") or []
    if not source or not postprocess:
        raise SystemExit(f"{args.config} needs `source: <experiment>` and `eval.postprocess`")

    run_name = f"{cfg.data.split}-f{cfg.data.fold}-s{cfg.train.seed}"
    source_dir = Path("results") / source / run_name
    for path in (source_dir / "summary.json", source_dir / "config.yaml", source_dir / "best_epoch" / "val"):
        if not path.exists():
            raise SystemExit(f"missing source artifact: {path}")
    run_dir = Path("results") / cfg.experiment / run_name
    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{run_dir} already exists (use --overwrite)")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    # Stitching must undo exactly the preprocessing the source run was trained with, so use
    # its saved config for that, not current.yaml (which may have changed since).
    source_cfg = load_config(source_dir / "config.yaml")
    run_info = {"config_file": args.config, "overrides": args.set, "source_run": str(source_dir),
                "started": datetime.now().isoformat(timespec="seconds"), "git": git_info()}
    save_config({**cfg.to_dict(), "_run": run_info}, run_dir / "config.yaml")
    shutil.copytree(source_dir / "best_epoch" / "val", run_dir / "best_epoch" / "val")
    print(f">>> Experiment '{cfg.experiment}': {', '.join(map(str, postprocess))} on {source_dir} -> {run_dir}")

    _, val_ids = load_split(source_cfg.data.split, source_cfg.data.fold)
    metrics_3d = cfg.get("eval", {}).get("metrics_3d") or []
    results = evaluate_3d(run_dir, source_cfg, val_ids, metrics_3d, build_postprocess(postprocess), raw=False)

    summary = json.loads((source_dir / "summary.json").read_text())
    summary.pop("metrics_3d_post", None)
    summary.update({"experiment": cfg.experiment, "owner": cfg.get("owner"), "idea": cfg.get("idea"),
                    "source_run": str(source_dir), "postprocess": postprocess, "git": run_info["git"], **results})
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    for name, m in results["metrics_3d"].items():
        print(f">>> 3D {name}: {m['mean']}")
    print(f">>> Done: {run_dir / 'summary.json'}  (then: python compare.py)")


if __name__ == "__main__":
    main()
