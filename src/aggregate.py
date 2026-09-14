"""Build comparison tables from every run copied into metrics/.

    python -m src.aggregate                 # -> metrics/comparison_runs.csv, metrics/comparison.md
    python -m src.aggregate --filter enet   # only experiments whose name contains "enet"

comparison_runs.csv : one row per run (experiment x seed), all numbers, for plotting/pandas.
comparison.md       : one row per experiment, mean ± std over seeds, paste-ready for the report.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

from src.config import REPO

def report_columns(runs: list[dict]) -> list[tuple[str, str, int]]:
    """(markdown label, flattened summary key, digits); one 3D Dice column per evaluated class."""
    per_class = []
    for r in runs:
        per_class += [k for k in r if k.startswith("eval.val_dice_") and k != "eval.val_dice_fg"
                      and k not in per_class]
    return ([("val Dice 2D", "best.val_dice_fg", 3), ("3D Dice fg", "eval.val_dice_fg", 3)]
            + [(f"3D Dice {k.removeprefix('eval.val_dice_')}", k, 3) for k in per_class]
            + [("HD95 mm", "eval.val_hd95_fg", 1), ("ASSD mm", "eval.val_assd_fg", 2)])


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out |= flatten(v, f"{prefix}{k}.")
        else:
            out[f"{prefix}{k}"] = v
    return out


def load_runs(metrics_dir: Path, name_filter: str) -> list[dict]:
    runs = [flatten(json.loads(p.read_text())) for p in sorted(metrics_dir.glob("*/*/summary.json"))]
    return [r for r in runs if name_filter in r["experiment"]]


def fmt(values: list, digits: int) -> str:
    values = [v for v in values if isinstance(v, (int, float)) and v == v]  # drop missing / NaN
    if not values:
        return "–"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-dir", type=Path, default=REPO / "metrics")
    parser.add_argument("--filter", default="", help="substring of experiment names to keep")
    args = parser.parse_args(argv)

    runs = load_runs(args.metrics_dir, args.filter)
    if not runs:
        raise SystemExit(f"no summary.json under {args.metrics_dir}/<experiment>/<seed>/")

    columns = sorted({k for r in runs for k in r}, key=lambda k: (k.count("."), k))
    with (args.metrics_dir / "comparison_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(runs)

    by_experiment: dict[str, list[dict]] = {}
    for r in runs:
        by_experiment.setdefault(r["experiment"], []).append(r)
    report = report_columns(runs)
    header = ["experiment", "model", "loss", "seeds", *(label for label, _, _ in report)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, group in sorted(by_experiment.items()):
        cells = [name, group[0]["model"], group[0]["loss"], str(len(group))]
        cells += [fmt([r.get(key) for r in group], digits) for _, key, digits in report]
        lines.append("| " + " | ".join(cells) + " |")
    note = ("\n\nmean ± std over seeds. 2D Dice: patient-level at 256x256 (training selection metric). "
            "3D: best checkpoint on the original CT grid, foreground = eval.classes. See docs/metrics.md.\n")
    (args.metrics_dir / "comparison.md").write_text("\n".join(lines) + note)
    print("\n".join(lines))
    print(f"\n{len(runs)} runs -> {args.metrics_dir / 'comparison_runs.csv'}, {args.metrics_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
