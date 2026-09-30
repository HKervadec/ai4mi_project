#!/usr/bin/env python3
"""Evaluate an existing run without retraining.

    python eval_run.py results/<experiment>/<run>
    python eval_run.py results/<experiment>/<run> --metrics dice hd95
    python eval_run.py results/<experiment>/<run> --postprocess largest_cc
"""

import argparse
import json
from pathlib import Path

from segpipe.config import load_config
from segpipe.data import load_split
from segpipe.evaluate import METRICS, evaluate_3d
from segpipe.postprocess import POSTPROCESS, build_postprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an existing run")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--metrics", nargs="+", choices=sorted(METRICS), default=None,
                        help="metrics to compute (default: all current 3D metrics)")
    parser.add_argument("--postprocess", nargs="*", choices=sorted(POSTPROCESS), default=None,
                        help="also score post-processed volumes (default: the run's eval.postprocess; "
                             "pass the flag without names to turn it off)")
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
    postprocess = args.postprocess if args.postprocess is not None else (cfg.get("eval", {}).get("postprocess") or [])
    print(f">> 3D evaluation ({', '.join(metrics)}) of {run_dir}"
          + (f", post-processing: {', '.join(map(str, postprocess))}" if postprocess else ""))
    results = evaluate_3d(run_dir, cfg, val_ids, metrics, build_postprocess(postprocess))

    summary = json.loads(summary_path.read_text())
    # Drop post-processed scores from an earlier evaluation so they never go stale.
    summary.pop("metrics_3d_post", None)
    summary.pop("postprocess", None)
    summary.update(results)
    if postprocess:
        summary["postprocess"] = postprocess
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    for key, metrics_3d in results.items():
        for name, values in metrics_3d.items():
            print(f">>> 3D {name}{' (post)' if key.endswith('post') else ''}: {values['mean']}")
    print(f">>> Updated: {summary_path}")


if __name__ == "__main__":
    main()