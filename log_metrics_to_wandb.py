#!/usr/bin/env python3
import argparse, sys
import numpy as np
from pathlib import Path
import wandb

def log_metrics_to_wandb(results_dir: str, epoch: int = 0):
    r = Path(results_dir)
    run_id_file = r / "wandb_run_id.txt"
    if not run_id_file.exists():
        print(f"No wandb run ID at {run_id_file}")
        sys.exit(1)

    run_id = run_id_file.read_text().strip()
    wandb.init(id=run_id, resume="must")

    metrics_to_log = {}
    for metric_name in ["3d_dice", "3d_hd95", "3d_assd", "3d_jaccard"]:
        f = r / "metrics" / f"{metric_name}.npy"
        if f.exists():
            data = np.load(f)
        
            # Log mean values per class
            mean_values = data.mean(axis=0)
            for class_idx, value in enumerate(mean_values):
                metrics_to_log[f'{metric_name}_class_{class_idx}'] = float(value)
            
            # Log overall mean
            metrics_to_log[f'{metric_name}_mean'] = float(data.mean())
            
            print(f'Logged {metric_name}: {mean_values}')

    if metrics_to_log:
        wandb.log(metrics_to_log, step=epoch)
        print(f"Logged: {metrics_to_log}")
    wandb.finish()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--epoch", type=int, default=0, help="Epoch to log (final epoch)")
    args = ap.parse_args()
    log_metrics_to_wandb(args.results_dir, args.epoch)

if __name__ == "__main__":
    main()
