#!/usr/bin/env python3
"""Regenerate every dataset-analysis figure from one raw NIfTI dataset directory.

Runs, in order:
  1. tools/dataset_profile.py             -> <out-dir>/profile/*.csv + histograms
  2. dataset_analysis/profile_figures/*   -> <out-dir>/profile/<name>/*.png
  3. tools/nnunet_planner_checks.py       -> <out-dir>/nnunet_checks.{json,md} + figures
  4. tools/explore_data.py                -> <out-dir>/fingerprint.{json,md} + figures
  5. dataset_analysis/analyze_dataset.py + validate_results.py
     -- only if --processed-dir is given and exists (needs the sliced 256x256
     PNG grid from slice_segthor.py, not just the raw NIfTI volumes)
  6. dataset_analysis/analyze_baseline.py + validate_results.py
     -- only if --predictions is given and exists (needs a completed training
     run's predictions on the validation split)

Every step above detects which labels actually have voxels in --data-dir
rather than assuming a fixed class count, so the same command works whether
the dataset omits the aorta annotation (label 4, the original course release)
or includes it (a full release).

Before/after: add --before-data-dir (e.g. the original 3-label data while --data-dir
is the corrected 4-label data). It is only profiled (tools/dataset_profile.py, into
<out-dir>/before/profile), and the HU intensity of both is drawn on one mirrored
axis (before upwards, --data-dir downwards) into --comparison-dir. Every other
figure comes from --data-dir alone.

Runs the same on a laptop or on Snellius (jobsAndOutputs/baseline/jobs/all_figures.job).

Usage:
    python tools/run_all_figures.py --data-dir data/segthor_part1/train --out-dir figures/segthor_part1

    # corrected data for every figure, original data only for the before/after intensity
    python tools/run_all_figures.py --data-dir data/segthor_part1_corrected/train --out-dir figures \\
        --before-data-dir data/segthor_part1/train --names "3 labels (aorta merged)" "4 labels (aorta separate)"

    # fast end-to-end check on a few patients, skipping the slow nnU-Net checks
    python tools/run_all_figures.py --max-patients 3 --skip-nnunet-checks --out-dir /tmp/smoke

    # also run the processed-PNG and baseline stages
    python tools/run_all_figures.py --data-dir data/segthor_part1/train --out-dir figures/segthor_part1 \\
        --processed-dir data/SEGTHOR --predictions results/segthor/ce/best_epoch/val
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# dataset_analysis/profile_figures scripts, in the order dataset_analysis/FIGURES.md
# documents them. Paths are relative to dataset_analysis/profile_figures/.
PROFILE_FIGURES = [
    "scan_geometry",
    "scan_intensity/center_edge_outside",
    "scan_intensity/circle_grid",
    "scan_intensity/nnunet_normalization",
    "scan_intensity/scan_vs_labels",
    "scan_intensity/slices_3d",
    "scan_intensity/spike_and_ridges",
    "label_intensity",
    "label_hu_distribution",
    "label_background_hu",
    "label_size_and_intensity",
    "label_shapes_and_sizes",
    "slices_and_change",
    "label_bounding_box",
    "organ_position_3d",
    "connected_components",
    "label_pairs",
    "label_pairs_strip",
]
# These reopen the raw volumes directly (for 3D meshes / per-voxel scans)
# instead of only reading the tables tools/dataset_profile.py already wrote.
NEEDS_DATA_DIR = {
    "scan_geometry",
    "scan_intensity/center_edge_outside",
    "scan_intensity/nnunet_normalization",
    "scan_intensity/slices_3d",
    "label_size_and_intensity",
    "label_shapes_and_sizes",
    "connected_components",
    "label_pairs",
}


Step = tuple[str, "list[str] | None"]  # (label, argv); argv None = skipped, label says why


def dataset_steps(data_dir: Path, out_dir: Path, args: argparse.Namespace, wants) -> list[Step]:
    """Steps 1-4: dataset profile, profile figures, nnU-Net checks and the fingerprint for one dataset.

    Args:
        data_dir: Raw NIfTI dataset directory.
        out_dir: Root for this dataset's figures and reports (profile tables go to <out_dir>/profile).
        args: Parsed command line.
        wants: Predicate telling whether a stage name is selected.

    Returns:
        Ordered (label, argv) steps.
    """
    py, profile_dir = sys.executable, out_dir / "profile"
    steps: list[Step] = []
    if wants("profile"):
        cmd = [py, "-B", "tools/dataset_profile.py", "--data-dir", str(data_dir), "--out-dir", str(profile_dir)]
        if args.processed_dir:
            cmd += ["--processed-dir", str(args.processed_dir)]
        if args.max_patients:
            cmd += ["--max-patients", str(args.max_patients)]
        steps.append(("dataset_profile", cmd))
        for name in PROFILE_FIGURES:
            cmd = [py, "-B", str(REPO / "dataset_analysis/profile_figures" / f"{name}.py"), "--profile-dir", str(profile_dir)]
            if name in NEEDS_DATA_DIR:
                cmd += ["--data-dir", str(data_dir)]
            steps.append((f"profile_figures/{name}", cmd))
    if wants("nnunet") and not args.skip_nnunet_checks:
        steps.append(("nnunet_planner_checks", [py, "-B", "tools/nnunet_planner_checks.py", "--data-dir", str(data_dir),
                                                "--out-dir", str(out_dir)]))
    if wants("explore"):
        steps.append(("explore_data", [py, "-B", "tools/explore_data.py", "--data-dir", str(data_dir),
                                       "--out-dir", str(out_dir)]))
    return steps


def analysis_steps(args: argparse.Namespace, wants) -> list[Step]:
    """Steps 5-6: the processed-PNG analysis, the baseline analysis and their validation.

    validate_results.py reads both analyses' tables, so it only runs after the baseline one.

    Args:
        args: Parsed command line.
        wants: Predicate telling whether a stage name is selected.

    Returns:
        Ordered (label, argv) steps; a skipped stage is (reason, None).
    """
    py = sys.executable
    common = ["--original-data", str(args.data_dir), "--processed-data", str(args.processed_dir),
              "--output-dir", str(args.dataset_analysis_out)]
    steps: list[Step] = []
    have_processed = bool(args.processed_dir) and args.processed_dir.is_dir()
    if wants("dataset"):
        if have_processed:
            cmd = [py, "-B", "dataset_analysis/analyze_dataset.py", *common]
            if args.max_patients:
                cmd += ["--max-patients", str(args.max_patients)]
            steps.append(("analyze_dataset", cmd))
        else:
            steps.append(("analyze_dataset skipped: no --processed-dir, or it doesn't exist", None))
    have_predictions = bool(args.predictions) and args.predictions.is_dir()
    if wants("baseline"):
        if have_processed and have_predictions:
            cmd = [py, "-B", "dataset_analysis/analyze_baseline.py", *common, "--predictions", str(args.predictions)]
            if args.reconstructed_volumes:
                cmd += ["--reconstructed-volumes", str(args.reconstructed_volumes)]
            steps.append(("analyze_baseline", cmd))
        else:
            steps.append(("analyze_baseline skipped: needs analyze_dataset and an existing --predictions", None))
    if any(label == "analyze_baseline" for label, _ in steps):
        steps.append(("validate_results", [py, "-B", "dataset_analysis/validate_results.py", *common]))
    return steps


def build_plan(args: argparse.Namespace) -> list[Step]:
    """Every step to run, in order.

    Args:
        args: Parsed command line.

    Returns:
        Ordered (label, argv) steps; a skipped stage is (reason, None).
    """
    stages = [s.strip() for s in args.only.split(",")] if args.only else None

    def wants(stage: str) -> bool:
        return stages is None or stage in stages

    steps = dataset_steps(args.data_dir, args.out_dir, args, wants)
    steps += analysis_steps(args, wants)
    if args.before_data_dir and wants("compare"):
        before_profile = args.out_dir / "before" / "profile"
        steps.append(("before/dataset_profile", [
            sys.executable, "-B", "tools/dataset_profile.py", "--data-dir", str(args.before_data_dir),
            "--out-dir", str(before_profile), *(["--max-patients", str(args.max_patients)] if args.max_patients else []),
        ]))
        steps.append(("label_hu_distribution (before/after)", [
            sys.executable, "-B", str(REPO / "dataset_analysis/profile_figures/label_hu_distribution.py"),
            "--profile-dir", str(before_profile), str(args.out_dir / "profile"),
            "--names", *args.names, "--out-dir", str(args.comparison_dir),
        ]))
        steps.append(("connected_components (before/after)", [
            sys.executable, "-B", str(REPO / "dataset_analysis/profile_figures/connected_components.py"),
            "--data-dir", str(args.data_dir), "--before-data-dir", str(args.before_data_dir),
            "--names", *args.names, "--out-dir", str(args.comparison_dir),
        ]))
        steps.append(("label_correction_per_patient (before/after)", [
            sys.executable, "-B", str(REPO / "dataset_analysis/profile_figures/label_correction_per_patient.py"),
            "--profile-dir", str(before_profile), str(args.out_dir / "profile"),
            "--names", *args.names, "--out-dir", str(args.comparison_dir),
        ]))
        steps.append(("label_makeup_rings (before/after)", [
            sys.executable, "-B", str(REPO / "dataset_analysis/profile_figures/label_makeup_rings.py"),
            "--profile-dir", str(before_profile), str(args.out_dir / "profile"),
            "--names", *args.names, "--out-dir", str(args.comparison_dir),
        ]))
        steps.append(("label_correction_example (before/after)", [
            sys.executable, "-B", str(REPO / "dataset_analysis/profile_figures/label_correction_example.py"),
            "--data-dir", str(args.data_dir), "--before-data-dir", str(args.before_data_dir),
            "--out-dir", str(args.comparison_dir),
        ]))
        steps.append(("label_example", [
            sys.executable, "-B", str(REPO / "dataset_analysis/profile_figures/label_example.py"),
            "--data-dir", str(args.data_dir), "--out-dir", str(args.comparison_dir),
        ]))
    return steps


def run(steps: list[Step]) -> None:
    """Run the steps in order, printing each one's wall time; a nonzero exit aborts the driver.

    Args:
        steps: (label, argv) pairs; argv None only prints the label.
    """
    for label, cmd in steps:
        if cmd is None:
            print(f"\n=== {label} ===")
            continue
        print(f"\n=== {label} ===")
        print(" ".join(cmd))
        start = time.monotonic()
        subprocess.run(cmd, cwd=REPO, check=True)
        print(f"--- {label} done in {time.monotonic() - start:.1f}s ---")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments, default sys.argv[1:].

    Returns:
        The parsed namespace, with the comparison folder default filled in.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/segthor_part1/train"),
                    help="Raw NIfTI dataset directory, one Patient_XX/ folder per patient.")
    ap.add_argument("--out-dir", type=Path, default=Path("figures"),
                    help="Root for the tools/-produced figures and JSON/Markdown reports.")
    ap.add_argument("--before-data-dir", type=Path, default=None,
                    help="Raw NIfTI dataset to compare --data-dir against in the HU intensity figure "
                         "(e.g. the original 3-label data); only profiled, no other figures.")
    ap.add_argument("--comparison-dir", type=Path, default=None,
                    help="Where the before/after figure goes; default <out-dir>/comparison.")
    ap.add_argument("--names", nargs=2, default=["before", "after"], metavar=("BEFORE", "AFTER"),
                    help="Names of --before-data-dir and --data-dir in the comparison figure.")
    ap.add_argument("--processed-dir", type=Path, default=None,
                    help="Sliced 256x256 PNG grid (slice_segthor.py output). "
                         "If given and it exists, also runs analyze_dataset.py (first dataset only).")
    ap.add_argument("--dataset-analysis-out", type=Path, default=Path("dataset_analysis/results"),
                    help="Output directory for analyze_dataset.py/analyze_baseline.py "
                         "(must stay under dataset_analysis/, per their own convention).")
    ap.add_argument("--predictions", type=Path, default=None,
                    help="Validation predictions from a completed training run. "
                         "If given and it exists, also runs analyze_baseline.py.")
    ap.add_argument("--reconstructed-volumes", type=Path, default=None,
                    help="Passed through to analyze_baseline.py.")
    ap.add_argument("--max-patients", type=int, default=None,
                    help="Deterministic subset, for a fast smoke run over every step.")
    ap.add_argument("--skip-nnunet-checks", action="store_true", help="Skip tools/nnunet_planner_checks.py (slow).")
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated stage names to run, skipping the rest. "
                         "Choices: profile, nnunet, explore, dataset, baseline, compare.")
    args = ap.parse_args(argv)
    if args.comparison_dir is None:
        args.comparison_dir = args.out_dir / "comparison"
    return args


def main() -> None:
    """Run the whole pipeline."""
    args = parse_args()
    steps = build_plan(args)
    run(steps)
    done = [label for label, cmd in steps if cmd is not None]
    print(f"\nAll requested stages complete ({len(done)} steps). Figures: {args.out_dir}"
          + (f", comparison in {args.comparison_dir}" if args.before_data_dir else ""))


if __name__ == "__main__":
    main()
