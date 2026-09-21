# SegTHOR exploratory analysis

This analysis covers annotated classes 1 (esophagus), 2 (heart), 3 (trachea),
and 4 (aorta). The original course release omits the aorta annotation
entirely (the professor confirmed this is intentional, not corruption); a
full 4-class release doesn't. Every script here reports which classes are
actually present in the data it's given rather than assuming either case, and
an absent class is left out of quality summaries rather than counted as zero.
Background means the supplied label 0, not necessarily anatomically empty
space.

For the full figure set (dataset profile, nnU-Net planner checks, EDA) in one
command, see `tools/run_all_figures.py` (documented in `tools/README.md`),
which also runs this directory's `analyze_dataset.py`/`analyze_baseline.py`
when given `--processed-dir`/`--predictions`.

Run from any directory with the existing `ai4mi` environment. Paths default to
the checkout containing these scripts; use `--help` for path overrides:

```bash
python -B dataset_analysis/analyze_dataset.py
python -B dataset_analysis/analyze_baseline.py
python -B dataset_analysis/validate_results.py
```

From the repository root, submit the CPU-only combined SLURM job:

```bash
sbatch jobsAndOutputs/baseline/jobs/dataset_analysis.job
```

It follows the existing genoa/module/conda setup. Logs go to the existing
`jobsAndOutputs/baseline/outputs`; CSVs, plots and examples go exclusively to
`dataset_analysis/results`. Scripts overwrite their own named outputs but never
write to data, predictions, checkpoints, or existing training logs. Run manifests
record arguments, library versions, code hashes, Git commit, and source size/mtime.
Results and caches are ignored by Git. A dataset run must precede a baseline run.

For numerical checks and a deterministic small end-to-end run:

```bash
python -B dataset_analysis/test_analysis.py
python -B dataset_analysis/analyze_dataset.py --max-patients 2 --output-dir dataset_analysis/results/smoke
python -B dataset_analysis/analyze_baseline.py --output-dir dataset_analysis/results/smoke
python -B dataset_analysis/validate_results.py --output-dir dataset_analysis/results/smoke
```

## Measurement conventions

- `patient_inventory.csv`, `patient_class_stats.csv`, and
  `class_frequency_original.csv` use ORIGINAL NIfTI labels. Physical volume is
  voxel count times the absolute determinant of the affine's spatial matrix;
  mm units and CT/GT geometry are checked. mL = mm³ / 1000. Bounding-box extent
  includes the full occupied voxel cells. Occupied-slice count can differ from
  first-to-last span when labels contain gaps. Background's fraction of
  foreground is undefined, recorded as blank.
- `slice_class_stats.csv` uses PROCESSED 256×256 PNGs, retaining every slice for
  each annotated class. Areas are pixels or fractions of image pixels, not mm².
  `slice_presence_summary.csv` gives pooled slice summaries by train/val/all.
- Scan-relative z is `index / (Z - 1)`; for Z=1 it is 0. This is not anatomical
  registration. Processed class-relative z is measured between its own first
  and last nonempty processed slices. It is undefined on GT-empty slices;
  a one-slice class gets 0.5 (middle). First/last distances are in slice indices.
- Area-versus-z plots include zeros. Slices are first averaged within each
  patient/bin; the line and IQR summarize those patient means. Tables retain
  both patient-bin values and train/val/all aggregates. IQR is descriptive,
  not a confidence interval. Five validation patients are not hundreds of
  independent observations despite their many correlated slices.
- `baseline_slice_metrics.csv` pairs GT and predictions by exact stem. Dice is
  `2*intersection/(GT area + predicted area)`, with no smoothing. Joint-empty
  Dice is undefined (blank), not 1. FP-only and complete-miss Dice are 0.
  Main Dice summaries use GT-positive slices; FP-only and joint-empty counts
  are reported separately. Predictions may contain class 4, but it has no
  metric row; predicting it on classes 1–3 still creates false negatives.
- Area bins use within-class quintiles with duplicate edges collapsed. Z bins
  use 10 equal scan intervals; class-extent bins use [0,.2), [.2,.8), [.8,1].
  Final bin endpoints are inclusive. Both pooled-slice and equal-patient
  summaries are saved in `baseline_binned_summary.csv`; detailed patient/bin
  summaries are in `baseline_patient_bins.csv`. Plotted shading is the IQR
  of patient mean Dice within the bin, not a slice-level confidence interval.
- `baseline_patient_metrics.csv` separates GT-positive mean/median slice Dice
  (processed grid) from full-mask 3D Dice (original grid). Existing reconstructed
  labels must have matching geometry and match the current prediction PNGs
  after the repository's exact 2× nearest-neighbor expansion. No reconstruction
  or inference is launched. Missing reconstructed files leave 3D cells blank.
- Example selections are deterministic (ties follow patient/slice order).
  Each class has six panels: smallest, nearest median, largest, lowest Dice,
  highest Dice and lowest-Dice extremity. The same slice can satisfy multiple
  criteria. Cyan is GT, orange is prediction. Crops retain PNG coordinates.

## Scope and limitations

Original anatomy and processed appearance are deliberately separate. Resizing
can alter small targets, extent and boundaries; expanding predictions to 512²
cannot recover lost detail. Frequency summaries pool voxels, whereas physical
volumes account for varying spacing. Validation slice distributions pool slices,
while explicitly named patient summaries give patients equal weight. These are
exploratory measurements, not a finalized course metric suite or a new method.

No dependency installation, GPU, training, complex morphology, clustering or
model modification is required. NumPy and standard-library CSV avoid adding
Pandas to the current environment.
