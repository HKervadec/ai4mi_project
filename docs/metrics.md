# Metrics

All metrics follow the conventions already used in `dataset_analysis/`, so numbers are comparable
across the whole project:

- **hard Dice, no smoothing**: `2|G∩P| / (|G|+|P|)`
- **both empty → NaN** (undefined, excluded from means). An empty/empty case is not a success, and
  counting it as 1.0 (as the course `dice_coef` with `smooth=1e-8` does) inflates scores for small organs.
- **one empty → Dice 0**, surface distances NaN
- foreground (`*_fg`) = mean over `eval.classes`, currently `[1, 2, 3]` (esophagus, heart, trachea).
  The aorta (4) has **no voxels in any ground truth** of `segthor_part1`, so it is excluded from `*_fg`.
  It still gets per-class rows in the 3D evaluation: Dice 0 if the model predicts it anywhere, NaN otherwise.

## During training (`epochs.csv`, W&B)

From every slice the engine keeps `(intersection, gt area, pred area)` per class. Then:

| column | meaning |
|---|---|
| `{split}_loss` | mean batch loss |
| `{split}_dice_<class>` | **patient-level Dice at 256×256**: counts summed over all slices of a patient (a 3D Dice on the resized grid), then averaged over patients |
| `{split}_dice_fg` | mean of the above over `eval.classes`. **Default checkpoint-selection metric** |
| `{split}_dice_legacy_fg` | the course metric, `log_dice_val[e, :, 1:].mean()`: per-slice smoothed Dice averaged over slices and all foreground classes incl. aorta. Kept only for comparison with the original baseline (0.807). Don't report it |

`val_counts_best.npz` keeps the raw per-slice counts of the best epoch, so any other 2D metric can be
computed later without retraining.

## Final evaluation (`src.evaluate`)

The best checkpoint predicts every val slice. Predictions are stitched back onto the **original CT grid**
(nearest-neighbour, exactly as `stitch.py`; `tests/test_stitch.py` checks voxel-for-voxel equality) and
compared with the original `GT.nii.gz`, using the voxel spacing from the NIfTI header (mm):

| metric | definition | empty handling |
|---|---|---|
| Dice | overlap, as above | both empty → NaN |
| HD95 (mm) | max of the two directed 95th-percentile surface distances | either empty → NaN |
| ASSD (mm) | mean of all symmetric surface-to-surface distances | either empty → NaN |

Surfaces are voxels of the mask with a 6-connected background neighbour. These are the "3D metrics"
the course asks for (see metrics-reloaded: an overlap measure plus a boundary measure).

Outputs: `eval/metrics_3d.csv` (patient × class), `eval/val/{dice,hd95,assd}.npz` in the submission format
(`Patient_XX → (K,)`, background entry NaN), and an `eval` block in `summary.json`:
`val_<metric>_<class>` = mean over patients, `val_<metric>_fg` = mean over `eval.classes`.
The orientation and spacing of each volume are logged in `eval.log`.

## Building tables

After runs finish, their small files are copied to `metrics/<experiment>/seed<seed>/` (commit them).
Then, on any checkout:

```bash
python -m src.aggregate                   # all experiments
python -m src.aggregate --filter segthor  # subset by experiment-name substring
```

- `metrics/comparison.md`: one row per experiment, mean ± std over seeds (paste into the report)
- `metrics/comparison_runs.csv`: one row per run, **every** field of `summary.json` flattened
  (`best.val_dice_heart`, `eval.val_hd95_fg`, `lr`, `git_commit`, ...), for custom tables/plots:

```python
import csv
rows = list(csv.DictReader(open("metrics/comparison_runs.csv")))
# or: import pandas as pd; pd.read_csv("metrics/comparison_runs.csv")
```

For per-patient analysis, concatenate the `metrics_3d.csv` files in `metrics/*/*/`.
Report at least 3 seeds per configuration before claiming an improvement: with 5 validation patients
and GPU nondeterminism, single-run differences of ~0.01 Dice are noise.
