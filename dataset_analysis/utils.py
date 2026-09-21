"""Shared I/O and explicit measurement conventions for SegTHOR exploration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

import nibabel as nib
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from plot_style import LABEL_COLORS  # noqa: E402

# Canonical label mapping for SegTHOR: some data-dirs (the original course
# release) have zero aorta voxels, some (a full release) don't -- CLASSES lists
# all four either way, and callers report an absent class as blank/NaN rather
# than assuming which case they're in. COLORS is re-exported from
# tools/plot_style.py so every figure in the repo colors organs the same way.
CLASSES = {1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}
COLORS = LABEL_COLORS


def parser(description: str) -> argparse.ArgumentParser:
    """Paths default to this checkout, independent of the current directory."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--repo-root", type=Path, default=REPO)
    p.add_argument("--original-data", type=Path)
    p.add_argument("--processed-data", type=Path)
    p.add_argument("--output-dir", type=Path)
    return p


def paths(args):
    """Restrict writes to an analysis subdirectory, never an input dataset."""
    root = args.repo_root.resolve()
    original = (args.original_data or root / "data/segthor_part1/train").resolve()
    processed = (args.processed_data or root / "data/SEGTHOR").resolve()
    output = (args.output_dir or root / "dataset_analysis/results").resolve()
    analysis = root / "dataset_analysis"
    if not output.is_relative_to(analysis) or output == analysis:
        raise ValueError("--output-dir must be a subdirectory of dataset_analysis")
    for source in (original, processed):
        if not source.is_dir():
            raise FileNotFoundError(source)
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output and input directories must not overlap")
    for sub in ("tables", "plots", "examples"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    return root, original, processed, output


def pyplot(output: Path):
    """Use a headless backend and keep Matplotlib's cache with analysis outputs."""
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 120})
    return plt


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write missing or undefined numeric values as blank CSV cells."""
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None or isinstance(v, (float, np.floating))
                             and not np.isfinite(v) else v for k, v in row.items()})


def read_csv(path: Path) -> list[dict]:
    """Load generated tables, retaining strings until explicitly converted."""
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def identity(path: Path) -> tuple[str, int]:
    match = re.fullmatch(r"(Patient_\d+)_(\d{4})", path.stem)
    if match is None:
        raise ValueError(f"Unexpected slice filename: {path}")
    return match[1], int(match[2])


def discover(processed: Path) -> dict:
    """Pair by exact filename and reject patient overlap or missing image/mask pairs."""
    patients = {}
    for split in ("train", "val"):
        images = {p.name: p for p in (processed / split / "img").glob("*.png")}
        masks = {p.name: p for p in (processed / split / "gt").glob("*.png")}
        if not images or images.keys() != masks.keys():
            raise ValueError(f"Missing/mismatched image and GT files in {split}")
        for name in sorted(masks):
            patient, z = identity(masks[name])
            entry = patients.setdefault(patient, {"split": split, "slices": {}})
            if entry["split"] != split or z in entry["slices"]:
                raise ValueError(f"Patient leakage or duplicate index: {patient}, {z}")
            entry["slices"][z] = (images[name], masks[name])
    return dict(sorted(patients.items()))


PNG_LABEL_VALUES = np.array([0, 63, 126, 189, 252])  # background, then classes 1-4


def load_png(path: Path) -> np.ndarray:
    """Decode exact 63-spaced labels, 0-4; whether 252 (aorta) occurs depends on the dataset."""
    with Image.open(path) as img:
        a = np.asarray(img)
    if a.ndim != 2 or a.dtype != np.uint8 or not np.isin(a, PNG_LABEL_VALUES).all():
        raise ValueError(f"Invalid PNG label encoding: {path}")
    return a // 63


def load_original(folder: Path, patient: str):
    """Validate CT/GT geometry and units before interpreting physical quantities."""
    ct = nib.load(folder / patient / f"{patient}.nii.gz")
    gt = nib.load(folder / patient / "GT.nii.gz")
    if len(gt.shape) != 3 or ct.shape != gt.shape or not np.allclose(ct.affine, gt.affine):
        raise ValueError(f"CT/GT geometry mismatch for {patient}")
    spacing = np.asarray(gt.header.get_zooms()[:3], dtype=float)
    if (gt.header.get_xyzt_units()[0] != "mm" or not np.all(spacing > 0)
            or not np.allclose(spacing, np.linalg.norm(gt.affine[:3, :3], axis=0))):
        raise ValueError(f"Unusable physical geometry for {patient}")
    data = np.asanyarray(gt.dataobj)
    # 0-4: whether label 4 (aorta) actually occurs depends on the dataset.
    if not np.issubdtype(data.dtype, np.integer) or data.min() < 0 or data.max() > 4:
        raise ValueError(f"Expected supplied annotations 0–4 for {patient}")
    return gt, data


def normalized_z(index: int, count: int) -> float:
    return index / (count - 1) if count > 1 else 0.0


def extent(positive: list[int], z: int | None = None) -> dict:
    """An absent class has undefined extent; a one-slice class is at relative z=0.5."""
    first, last = (positive[0], positive[-1]) if positive else (None, None)
    present = z is not None and z in positive
    return {"first_positive_slice": first, "last_positive_slice": last,
            "distance_from_first": z - first if present else None,
            "distance_from_last": last - z if present else None,
            "organ_relative_z": ((z - first) / (last - first) if last != first else 0.5)
            if present else None}


def overlap(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Exact hard Dice; joint-empty Dice is NaN, not a successful segmentation."""
    if gt.shape != pred.shape:
        raise ValueError("Mask shapes differ")
    g, p = int(np.count_nonzero(gt)), int(np.count_nonzero(pred))
    intersection = int(np.count_nonzero(gt & pred))
    return {"gt_present": bool(g), "pred_present": bool(p), "joint_empty": not (g or p),
            "gt_area": g, "pred_area": p, "intersection": intersection,
            "dice": 2 * intersection / (g + p) if g + p else np.nan,
            "fp_pixels": p - intersection, "fn_pixels": g - intersection}


def distribution(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict.fromkeys(("mean", "min", "max", "p05", "p25", "median", "p75", "p95"), np.nan)
    qs = np.percentile(a, [5, 25, 50, 75, 95])
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
            **dict(zip(("p05", "p25", "median", "p75", "p95"), map(float, qs)))}


def provenance(output: Path, stage: str, args, inputs: list[Path], extra: dict) -> None:
    """Record versions, code hashes and source file size/mtime without modifying inputs."""
    rows = [{"path": str(p.resolve()), "size_bytes": p.stat().st_size,
             "mtime_ns": p.stat().st_mtime_ns} for p in sorted(set(inputs))]
    write_csv(output / "tables" / f"{stage}_inputs.csv", rows)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo_root,
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    info = {"utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git_commit": commit, "versions": {p: importlib.metadata.version(p)
            for p in ("numpy", "nibabel", "matplotlib", "pillow")},
            "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in Path(__file__).parent.glob("*.py")}, **extra}
    (output / f"{stage}_run.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
