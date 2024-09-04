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
    metrics: np.ndarray = np.load(args.metric_file)
    match metrics.ndim:
        case 2:
            E, N = metrics.shape
            K = 1
        case 3:
            E, N, K = metrics.shape

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(str(args.metric_file))

    epcs = np.arange(E)

    for k in range(1, K):
        y = metrics[:, :, k].mean(axis=1)
        ax.plot(epcs, y, label=f"{k=}", linewidth=1.5)

    if K > 2:
        ax.plot(epcs, metrics.mean(axis=1).mean(axis=1), label="All classes", linewidth=3)
        ax.legend()
    else:
        ax.plot(epcs, metrics.mean(axis=1), linewidth=3)

    fig.tight_layout()
    if args.dest:
        fig.savefig(args.dest)

    if not args.headless:
        plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot data over time')
    parser.add_argument('--metric_file', type=Path, required=True, metavar="METRIC_MODE.npy",
                        help="The metric file to plot.")
    parser.add_argument('--dest', type=Path, metavar="METRIC_MODE.png",
                        help="Optional: save the plot to a .png file")
    parser.add_argument("--headless", action="store_true",
                        help="Does not display the plot and save it directly (implies --dest to be provided.")

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    run(get_args())
