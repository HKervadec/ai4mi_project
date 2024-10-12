#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def run(args: argparse.Namespace) -> None:
    baseline_metrics: np.ndarray = np.load(args.baseline_metric_file)
    experiment_metrics: np.ndarray = np.load(args.experiment_metric_file)

    match baseline_metrics.ndim:
        case 2:
            E, N = baseline_metrics.shape
            K = 1
        case 3:
            E, N, K = baseline_metrics.shape

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title("Baseline vs Experiment")

    epcs = np.arange(E)

    class_names = ["background", "esophagus", "heart", "trachea", "aorta"]

    # Plot baseline metrics with faint color
    for k in range(1, K):
        y = baseline_metrics[:, :, k].mean(axis=1)
        ax.plot(epcs, y, label=f"Baseline {class_names[k]}", linewidth=1.5, linestyle='--', alpha=0.3)

    # Plot experiment metrics with prominent color
    for k in range(1, K):
        y = experiment_metrics[:, :, k].mean(axis=1)
        ax.plot(epcs, y, label=f"Experiment {class_names[k]}", linewidth=1.5)

    if K > 2:
        ax.plot(epcs, baseline_metrics.mean(axis=1).mean(axis=1), label="Baseline All classes", linewidth=3, linestyle='--', alpha=0.3)
        ax.plot(epcs, experiment_metrics.mean(axis=1).mean(axis=1), label="Experiment All classes", linewidth=3)
        ax.legend()
    else:
        ax.plot(epcs, baseline_metrics.mean(axis=1), linewidth=3, linestyle='--', alpha=0.3)
        ax.plot(epcs, experiment_metrics.mean(axis=1), linewidth=3)

    fig.tight_layout()
    if args.dest:
        fig.savefig(args.dest)

    if not args.headless:
        plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot baseline vs experiment data over time')
    parser.add_argument('--baseline_metric_file', type=Path, required=True, metavar="BASELINE_METRIC.npy",
                        help="The baseline metric file to plot.")
    parser.add_argument('--experiment_metric_file', type=Path, required=True, metavar="EXPERIMENT_METRIC.npy",
                        help="The experiment metric file to plot.")
    parser.add_argument('--dest', type=Path, metavar="METRIC_MODE.png",
                        help="Optional: save the plot to a .png file")
    parser.add_argument("--headless", action="store_true",
                        help="Does not display the plot and save it directly (implies --dest to be provided.")

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    run(get_args())