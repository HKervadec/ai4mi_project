#!/usr/bin/env python3
"""Run viewer/viewer.py headlessly and save its figure to a PNG.

viewer/viewer.py opens an interactive GUI window and has no --save option.
This wrapper forces the non-interactive Agg backend and redirects plt.show()
to savefig, so the exact same rendering can be written to a file (useful over
SSH, in CI, or to share a result).

Usage (from the repo root) -- takes every flag viewer.py takes:

    python tools/save_viewer.py --out figures/toy2.png \
        --img_source data/TOY2/val/img \
        data/TOY2/val/gt results/toy2/ce/iter000/val results/toy2/ce/best_epoch/val \
        --show_img -C 256 --no_contour
"""
import argparse
import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)


def main() -> None:
    # Pull out --out, pass everything else through to viewer.py untouched.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", required=True, help="destination .png")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--size", type=float, nargs=2, default=(16, 9),
                        help="figure size in inches, W H")
    args, passthrough = parser.parse_known_args()

    def save_show(*_a, **_k):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            fig.set_size_inches(*args.size)
            fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print(f"saved -> {args.out}")

    plt.show = save_show

    sys.argv = ["viewer/viewer.py"] + passthrough
    runpy.run_path("viewer/viewer.py", run_name="__main__")


if __name__ == "__main__":
    main()
