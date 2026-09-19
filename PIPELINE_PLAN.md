# Pipeline plan — SegTHOR experiments

Status: **workflow implemented, no method changes.** A run with `configs/current.yaml` does
exactly what `main.py --experiment refined_window` does (same slicing, transforms, training
loop and seeding; checked on one training step). The components contain only what the repo
already had: ENet and the shallow CNN, cross-entropy, Adam, HU windowing. Everything else
gets added by the group on branches.

**Goal:** everyone tries ideas on their own branch by adding code in the right file and
writing a small config. Results are comparable and shared. Current choices live in one
file and can always be changed.

---

## 1. Overview

```
configs/current.yaml          <- the group's current choices (editable)
        ▲ base:
configs/experiments/*.yaml    <- one small file per experiment: only what it changes

python run.py --config configs/experiments/<name>.yaml
        ├─ train       (components picked by name from the config; best epoch chosen on 2D val Dice)
        └─ evaluate    best_epoch/val stitched into 3D volumes (stitch.py) -> 3D Dice
                       -> results/<experiment>/<run>/summary.json (2D val Dice and 3D Dice)

python compare.py             <- RESULTS.md
DECISIONS.md                  <- what we chose, why, when to revisit
```

---

## 2. Assignment options (at least 5)

| Option | File | Config key |
|---|---|---|
| Additional pre-processing | `segpipe/data.py` | `data`, `input` |
| Data augmentation | `segpipe/augment.py` | `augment` |
| 2.5D network | `segpipe/data.py` (not implemented yet) | `input.context_slices` |
| Different optimizer | `segpipe/optim.py` | `optimizer`, `scheduler` |
| Non-CNN architecture (Transformer / ViT) | `segpipe/models.py` | `model` |
| Different network architecture | `segpipe/models.py` | `model` |
| Post-processing | `segpipe/postprocess.py` (not wired in yet) | |
| Different loss function | `segpipe/losses.py` | `loss` |
| Regularizer at the loss level | `segpipe/losses.py` | `regularizers` |
| Pre-training / hybrid supervision | `segpipe/pretrain.py` | `model.init_from` |

Owners and status: table at the top of `DECISIONS.md`.

---

## 3. Files

```
configs/current.yaml           current choices
configs/experiments/           one file per experiment
splits/holdout.json            15 train / 5 val (Patient_01, 11, 15, 17, 19, same as before)
splits/cv4.json                4 folds of 5 patients
segpipe/
  config.py                    YAML loading, base: inheritance, --set overrides
  data.py                      splits, slice cache, SliceDataset
  augment.py                   AUGMENTS
  models.py                    MODELS
  losses.py                    LOSSES, REGULARIZERS
  optim.py                     OPTIMIZERS, SCHEDULERS
  postprocess.py               POSTPROCESS (not wired in yet)
  evaluate.py                  METRICS, 3D evaluation of best_epoch/val
  train.py                     training loop (from main.py)
  pretrain.py                  empty
run.py                         one experiment end-to-end
compare.py                     all summaries -> RESULTS.md
scripts/run.job                Snellius: sbatch scripts/run.job <config> [--set ...]
RESULTS.md                     generated
DECISIONS.md                   written by hand
```
Existing scripts (`main.py`, `fix_segthor_gt.py`, `stitch.py`, `audit_data.py`, ...) are unchanged.

---

## 4. Configs

`configs/current.yaml` holds every setting. An experiment only lists what it changes:
```yaml
base: ../current.yaml
owner: <name>
idea: "<one sentence>"
loss: {name: <name>}
```
A quick change without a new file:
```bash
python run.py --config configs/current.yaml --set train.seed=43 data.split=cv4 data.fold=2
```

---

## 5. Running

```bash
make data/gt/watershed_refined          # once: refined GT
python run.py --config configs/current.yaml       # builds the data cache the first time
python run.py --config configs/current.yaml --debug --set train.epochs=1 train.num_workers=0   # quick test
python compare.py                       # update RESULTS.md
```
A run writes to `results/<experiment>/<split>-f<fold>-s<seed>/`:
- `config.yaml`, `summary.json` (committed)
- `log.csv`, `*.npy`, `bestweights.pt`, `bestmodel.pkl`, `best_epoch.txt`, `iter###/val`, `best_epoch/val` (not committed; same files as `main.py`)
- `volumes/val/*.nii.gz`, `metrics_3d/<metric>.npz` (not committed; 3D evaluation, skipped for `--debug` runs)

The slice cache (`data/cache/<gt>_<window>_<H>x<W>/`) holds all patients, sliced once with
`slice_segthor.slice_patient`. A split only selects patients, so changing splits never requires
re-slicing. A different window or shape gets its own cache folder.

---

## 6. Adding something

1. Write the class or function in the file from §2.
2. Add it to the dictionary in that file (`MODELS`, `LOSSES`, `AUGMENTS`, ...). The expected
   signature is in the comment above the dictionary.
3. Use its name in an experiment config.

`run.py` and the training loop don't change. A new 3D metric goes into `METRICS` in `evaluate.py`
and `eval.metrics_3d` in the config. Post-processing has its file but is not wired into `run.py` yet.

---

## 7. Making choices

- `compare.py` shows every experiment against `current`: 2D val Dice and 3D Dice (mean ± std over runs), 3D Dice per organ.
- Guideline: run a promising idea with 3 seeds before adopting it; use `cv4` for close calls.
- Adopting or undoing a choice: edit the line in `current.yaml` and add an entry to `DECISIONS.md`.
- Old results keep their own saved `config.yaml`, so they stay readable after `current.yaml` changes.

---

## 8. Working together

- `feature/<thing>` branches for new code, `exp/<name>` branches for experiment configs and
  results. Merge both via PR, including negative results.
- Commit configs, `results/*/*/summary.json`, `RESULTS.md` and `DECISIONS.md`. Checkpoints and
  volumes go to a shared Snellius folder.
- If `RESULTS.md` conflicts in a merge, rerun `compare.py`.

---

## 9. To agree on
- Main metric for comparing runs
- Owners for the options in §2
