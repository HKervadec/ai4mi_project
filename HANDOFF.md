# `fix-data` branch — handoff notes

> **Update (pipeline_redesign branch).** The experiment framework described below
> (`experiments.py` + `main.py --experiment`) has since been replaced by a
> config-driven pipeline in `segpipe/`, driven by `run.py` and YAML configs. See
> [`PIPELINE_PLAN.md`](PIPELINE_PLAN.md) for the current workflow; use it for all
> new experiments. This document is still the reference for **why the data needed
> fixing** (aorta/esophagus GT split, HU windowing) and how those fixes were made —
> the new pipeline consumes the same corrected GT (`data/gt/watershed_refined`).
> The old `main.py --experiment` path still runs but is not the way forward.
>
> **3D metrics status:** `segpipe/evaluate.py` wires in a 3D-Dice evaluation
> (`evaluate_3d`, stitch → per-organ 3D Dice), but it is **not yet validated on a
> full run** — a group member is finalizing it, so treat the 3D numbers as
> provisional until then. The training / 2D-Dice / best-epoch path is confirmed
> working end-to-end.

This branch makes the SegTHOR data trainable and sets up an **experiment
framework** so we can measure, one technique at a time, whether a change actually
improves organ segmentation. Nothing from the original pipeline was deleted — the
old behavior is preserved as the `original` experiment and behind flags.

It contains three things:

1. **Aorta/esophagus GT split** — repairs the corrupted ground truth. Two methods.
2. **Experiment pipeline** — one registry + `--experiment` flag to switch and A/B techniques.
3. **First preprocessing experiment** — HU windowing (vs the original min-max), plus reproducible **seeding**.

---

## 0. TL;DR — get training in 4 commands

```bash 
source ai4mi/bin/activate # activate venv

git submodule update --init                 # pull the viewer submodule
python -m pip install -r requirements.txt   # (in your venv; scipy is new)

make data-experiments                        # build all GT-fix variants + sliced datasets
OR
make data/experiments/refined_window         # for example if you only want to build one version

python main.py --experiment watershed_window --epochs 25 --gpu # train a certain experiment
```

Results land in `data/experiments/watershed_window/results/`. Swap the experiment
name for any of the four (see §3) to run the other points of the comparison.

---

## 1. Files added / changed

| File | What it is |
|------|------------|
| `fix_gt.py` | GT fix method #1 (**watershed**): splits aorta out of esophagus + restamps the CT affine. |
| `fix_segthor_gt.py` | GT fix method #2 (**watershed_refined**): same idea + per-slice crumb cleanup, higher-quality split. |
| `audit_data.py` | Read-only data integrity check (raw 3D + sliced 2D). |
| `preprocessing.py` | HU windowing (fixed clinical window → uint8). |
| `experiments.py` | The experiment registry — single source of truth for technique configs. |
| `slice_segthor.py` | Applies HU windowing; takes `--experiment`. |
| `main.py` | Takes `--experiment` and `--seed`; loss factory; self-contained results dir. |
| `Makefile` | Targets to build each GT-fix variant and each experiment dataset. |
| `requirements.txt` | Added `scipy`. |

### Data layout

The layout has **two axes** and is named so the folders read as the ablation:

```
data/
├── segthor_part1/          raw source — CORRUPTED GT (aorta merged, affine stripped)
│
├── gt/                     corrected-GT variants, one folder per FIX METHOD
│   ├── watershed/          <- fix_gt.py           (self-test aorta 0.95 / eso 0.79)
│   └── watershed_refined/  <- fix_segthor_gt.py   (self-test aorta 0.9995 / eso 0.998)
│
└── experiments/            sliced datasets, named <gtfix>_<intensity>
    ├── original/           corrupted GT   + min-max        (the literal "before")
    ├── watershed_minmax/   watershed fix  + min-max
    ├── watershed_window/   watershed fix  + HU window
    └── refined_window/     refined fix    + HU window
        └── each holds  {train, val, results}
```

Everything under `data/` is gitignored — teammates **rebuild locally** with
`make`; only code is pushed.

**Label convention** (everywhere): `0=background, 1=esophagus, 2=heart, 3=trachea, 4=aorta`.
In the sliced PNGs each class `k` is stored as `k*63` (i.e. `{0,63,126,189,252}`).

---

## 2. Aorta/esophagus GT fix (`fix_gt.py`, `fix_segthor_gt.py`)

### The problem
The raw part-1 ground truth has two defects (both caught by `audit_data.py`):
1. The **aorta (class 4) is merged into the esophagus (class 1)** in every patient
   → the network can never learn the aorta.
2. Every `GT.nii.gz` has its **affine stripped to identity** → it no longer shares
   the CT's world geometry, which breaks 3D metrics and stitching.

### The two fix methods
Both use the same idea — the aorta is a fat tube (~10–14 mm), the esophagus a thin
one (~4 mm), so a 3D **distance transform** turns caliber into a scalar and a
seeded **watershed** cuts at the thin neck between them (CT intensity can't help:
non-contrast scans). Both re-stamp the CT affine. **Patient_07** ships a gold
`GT2.nii.gz`; it's used directly for that patient and as an honest self-test.

| method | folder | script | Patient_07 self-test | notes |
|--------|--------|--------|----------------------|-------|
| watershed | `data/gt/watershed` | `fix_gt.py` | aorta 0.95 / eso 0.79 | first version; also flags aorta-fraction outliers |
| watershed_refined | `data/gt/watershed_refined` | `fix_segthor_gt.py` | aorta 0.9995 / eso 0.998 | + padded bbox, per-slice crumb reassignment |

### Run them
```bash
make data/gt/watershed            # fix_gt.py
make data/gt/watershed_refined    # fix_segthor_gt.py
# audit either:
python audit_data.py --raw_dir data/gt/watershed_refined/train
```
(The two scripts have slightly different CLIs; the Makefile absorbs that.)

### ⚠️ Important caveats
- **Recovered GT is approximate for 19/20 patients** — only Patient_07 is gold.
  This includes all 5 validation patients. Comparisons stay valid because the GT
  is held constant within a comparison, but absolute Dice is against approximate
  labels.
- **Inspect flagged patients in ITK-SNAP.** `audit_data.py` flags e.g.
  `Patient_05` on the refined fix (esophagus 2.2× cohort median → possible
  incomplete aorta split). `fix_gt.py` prints its own outlier list too.

### Which fix is "better"?
Two different questions:
- **More correct?** → the **Patient_07 self-test** answers it directly; refined
  wins decisively (0.998 vs 0.79 esophagus).
- **Better downstream Dice?** → run `watershed_window` vs `refined_window`, **but**
  each is scored against its *own* fix's recovered val labels, which is circular.
  For a fair downstream comparison, score both models against **one common
  reference val GT** (the better fix's, or a hand-corrected val set).

---

## 3. Experiment pipeline

`experiments.py` is the **single source of truth**. Each experiment bundles its
slice-stage options (`SliceConfig`: `source_dir`, `window`, `shape`, `retains`)
and train-stage options (`TrainConfig`: `mode`, `loss`, `augment`). Both
`slice_segthor.py` and `main.py` take `--experiment <name>`.
**Design rule: never delete an old code path — branch on config instead.**
Precedence: **explicit CLI flag > experiment config > built-in default.**

### The four experiments (a two-axis ablation)
Named `<gtfix>_<intensity>`; each adjacent pair changes exactly one variable:

| experiment | GT fix (`source_dir`) | intensity (`window`) |
|------------|-----------------------|----------------------|
| `original`         | `data/segthor_part1` (corrupted) | min-max |
| `watershed_minmax` | `data/gt/watershed`              | min-max |
| `watershed_window` | `data/gt/watershed`              | HU mediastinal window |
| `refined_window`   | `data/gt/watershed_refined`      | HU mediastinal window |

| Compare | Isolates |
|---|---|
| `original` → `watershed_minmax` | the **GT fix** (does splitting the aorta help) |
| `watershed_minmax` → `watershed_window` | the **HU windowing** |
| `watershed_window` → `refined_window` | the **fix method** (basic vs refined split) |

The **validation split is identical** across all four (patients `01,11,15,17,19`;
15 train / 5 val), so comparisons are fair.

### Build & train
```bash
make data-experiments                                     # build all four, OR:
make data/experiments/refined_window                      # just one (builds its GT fix first)

python main.py --experiment watershed_window --epochs 25 --gpu
python main.py --experiment refined_window  --epochs 25 --gpu
```
`--dest` defaults to `data/experiments/<name>/results/`. To force a dataset
rebuild, delete it first: `rm -rf data/experiments/refined_window`.

### How to add your own experiment
1. **Register it** in `experiments.py`:
   ```python
   register(Experiment(
       name="refined_augment",                             # <gtfix>_<technique>
       description="what it changes vs its neighbour",
       slice=SliceConfig(source_dir="data/gt/watershed_refined", window=MEDIASTINAL),
       train=TrainConfig(augment=True, loss="ce"),
   ))
   ```
2. **Copy a Makefile rule** (→ builds into `data/experiments/refined_augment/`).
   Add `| data/gt/<method>` as an order-only prereq so its GT source auto-builds.
3. `make data/experiments/refined_augment`
4. `python main.py --experiment refined_augment --gpu`

Extension points already wired: `SliceConfig` / `TrainConfig` fields, and
`build_loss(name, K, mode)` in `main.py` (add `dice`/`focal` there; anything but
`"ce"` currently raises a clear `NotImplementedError`).

---

## 4. First preprocessing experiment — HU windowing (`preprocessing.py`)

### Why
CT voxels are calibrated Hounsfield Units (an absolute scale), so a **fixed window
is physically meaningful across patients**. The original **per-volume min-max is
not**: one high-attenuation outlier (metal, contrast, artifact) blows up the range
and crushes soft tissue into a few grey levels. In our data this is real — e.g.
Patient_02 reaches ~26,600 HU while its 99th percentile is ~370 HU, so under
min-max the mediastinal soft tissue that separates esophagus/aorta/heart collapses
to ~2–4 grey levels (worst for the **esophagus**, our hardest class). Measured on
that slice: min-max → `max=18, std=4`; HU window → full `0–255, std=66`.

Research basis: fixed windowing is standard clinical CT reading; nnU-Net clips CT
intensities before normalization (Isensee et al., *Nature Methods* 2021).

### What it does
`apply_hu_window(volume, level, width)` clips to `[level-width/2, level+width/2]`
and scales to uint8 anchored to those **fixed** bounds. Default is the clinical
**mediastinal / soft-tissue window** (`level=40, width=400` → clip `[-160, 240]`).

### Trade-off / knobs
A soft-tissue window maps the air-filled **trachea lumen to 0** (like external
air), so the trachea is learned from wall/shape. If trachea recall suffers, try the
lung window (already defined as `CT_WINDOWS["lung"]`):
```bash
python slice_segthor.py --experiment watershed_window \
    --window_level -600 --window_width 1500 --dest_dir data/experiments/lung_window
```
Ad-hoc back to min-max: `slice_segthor.py ... --legacy_norm`.

---

## 5. Seeding / reproducibility (`main.py`)

Training is **reproducible** so an A/B difference reflects the technique, not
run-to-run noise.

- `--seed` (default **42**) → `seed_everything()` seeds `random`, `numpy`, `torch`
  (+CUDA), cuDNN deterministic; the train `DataLoader` uses a seeded generator +
  `seed_worker`. Verified: same seed → identical Dice; different seed → different.

**Two different seeds — don't confuse them:**
- `slice_segthor.py --seed` (default **0**) → the **validation split** at slice
  time. Leave at 0 so all experiments share the same split.
- `main.py --seed` (default **42**) → the **training** RNG.

**Measure variance** — run a few seeds and average:
```bash
for s in 42 43 44; do python main.py --experiment watershed_window --seed $s \
  --epochs 25 --gpu --dest data/experiments/watershed_window/results_seed$s; done
```
Same-machine reproducibility is guaranteed; exact numbers can differ across
GPU/CPU hardware (normal; doesn't affect within-machine comparisons).

---

## 6. Full workflow from a fresh clone

```bash
# 1. environment
git clone <repo> && cd <repo>          # (or: git checkout fix-data && git pull)
git submodule update --init
python -m venv ai4mi && source ai4mi/bin/activate
python -m pip install -r requirements.txt

# 2. data  (data/segthor_part1 must already be present — see readme.md "Getting the data")
make data-experiments            # builds data/gt/* and data/experiments/*

# 3. sanity check before spending GPU time
python audit_data.py --raw_dir data/gt/watershed/train --sliced_dir data/experiments/watershed_window

# 4. train (results -> data/experiments/<name>/results/)
python main.py --experiment watershed_minmax --epochs 25 --gpu
python main.py --experiment watershed_window --epochs 25 --gpu
python main.py --experiment refined_window   --epochs 25 --gpu
```

`--gpu` uses CUDA if available, else Apple MPS, else CPU. If you hit a
"cannot pickle" DataLoader error, set `num_workers=0` (see `readme.md` known
issues); it's `5` in `main.py:setup`.

---

## 7. Open items / TODO for whoever takes over

1. **Data augmentation is NOT wired yet.** The `augment` flag flows from
   `TrainConfig` → `SliceDataset`, but `SliceDataset.__getitem__` applies no
   transforms, and the image and GT transforms are applied **independently** — so
   coupled spatial augmentation (rotate/flip/elastic must use the *same* geometry
   on image and mask) can't be expressed. **Needs a `__getitem__` refactor**:
   joint spatial transform, train-split only. Same for the unused `equalize` flag.
   *(Branch `Testing-data-augmentation` is on this.)*
2. **3D metrics** (rubric requires 3D). In-loop Dice is 2D per-slice. Build a step
   that consumes `data/experiments/*/results/best_epoch/val/`, stitches with
   `stitch.py`, writes an `.npz` per experiment. *(Branch `3D-Metrics`.)*
3. **More losses** (Dice / focal / combos): add a branch in `build_loss`, set
   `TrainConfig.loss`.
4. **Inspect flagged patients** in ITK-SNAP (`Patient_05` on the refined fix, plus
   whatever `fix_gt.py` flags) and hand-correct bad splits.
5. **Fix-method comparison** should be scored against a common val GT (see §2).
6. **`num_workers`** is hardcoded to 5 — parametrize if it causes issues.

---

## 8. Quick reference

```bash
# build a GT-fix variant
make data/gt/watershed            # fix_gt.py
make data/gt/watershed_refined    # fix_segthor_gt.py
# audit (read-only)
python audit_data.py --raw_dir data/gt/<method>/train --sliced_dir data/experiments/<name>
# build a sliced dataset (auto-builds its GT source)
make data/experiments/<name>
# train
python main.py --experiment <name> --epochs 25 --seed 42 --gpu
# ad-hoc slicing overrides (bypass a config)
python slice_segthor.py --experiment watershed_window --window_level 40 --window_width 400 --dest_dir <dir>
python slice_segthor.py --experiment watershed_minmax --legacy_norm --dest_dir <dir>
```
