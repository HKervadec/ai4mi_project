#!/usr/bin/env python3
"""Evaluate an existing run without retraining.

    python eval_run.py results/<experiment>/<run>
    python eval_run.py results/<experiment>/<run> --metrics dice hd95
"""

import argparse
import json
from pathlib import Path

from segpipe.config import load_config
from segpipe.data import load_split
from segpipe.evaluate import METRICS, evaluate_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an existing run")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--metrics", nargs="+", choices=sorted(METRICS), default=None,
                        help="metrics to compute (default: all current 3D metrics)")
    args = parser.parse_args()

    run_dir = args.run_dir
    config_path = run_dir / "config.yaml"
    predictions = run_dir / "best_epoch" / "val"
    summary_path = run_dir / "summary.json"
    for path in (config_path, predictions, summary_path):
        if not path.exists():
            raise SystemExit(f"missing required run artifact: {path}")

    cfg = load_config(config_path)
    _, val_ids = load_split(cfg.data.split, cfg.data.fold)
    metrics = args.metrics or list(METRICS)
    print(f">> 3D evaluation ({', '.join(metrics)}) of {run_dir}")
    metrics_3d = evaluate_3d(run_dir, cfg, val_ids, metrics)

    summary = json.loads(summary_path.read_text())
    summary["metrics_3d"] = metrics_3d
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    for name, values in metrics_3d.items():
        print(f">>> 3D {name}: {values['mean']}")
    print(f">>> Updated: {summary_path}")


if __name__ == "__main__":
    main()