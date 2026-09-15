# `fix-data` branch — handoff notes

This branch makes the SegTHOR data trainable and sets up an **experiment
framework** so we can measure, technique by technique, whether a change actually
improves organ segmentation. Nothing from the original pipeline was deleted — the
old behavior is preserved as a named experiment and behind flags.

It contains three things:

1. **Aorta/esophagus GT split** — repairs the corrupted ground truth.
2. **Experiment pipeline** — one registry + `--experiment` flag to switch and A/B techniques.
3. **First preprocessing experiment** — HU windowing (vs the original min-max), plus reproducible **seeding**.

---

## 0. TL;DR — get training in 4 commands

```bash
git submodule update --init                 # pull the viewer submodule
python -m pip install -r requirements.txt   # (in your venv; scipy is new)

make data-experiments                        # build all 3 sliced datasets
python main.py --experiment hu_window --epochs 25 --gpu
```

Results land in `data/experiments/hu_window/results/`. Swap `hu_window` for
`baseline` or `original` to run the other points of the comparison.

---

## 1. Files added / changed

| File | What it is |
|------|------------|
| `fix_gt.py` | Repairs GT: splits aorta out of esophagus + restamps the CT affine. **New.** |
| `audit_data.py` | Read-only data integrity check (raw 3D + sliced 2D). **New.** |
| `preprocessing.py` | HU windowing (fixed clinical window → uint8). **New.** |
| `experiments.py` | The experiment registry — single source of truth for technique configs. **New.** |
| `slice_segthor.py` | Now applies HU windowing and takes `--experiment`. **Modified.** |
| `main.py` | Now takes `--experiment` and `--seed`; loss factory; self-contained results dir. **Modified.** |
| `Makefile` | Targets to build the corrected GT and per-experiment datasets. **Modified.** |
| `requirements.txt` | Added `scipy` (used by `fix_gt.py`). **Modified.** |

### Data layout

```
data/
├── segthor_part1/          raw source — CORRUPTED GT (aorta merged, affine stripped)
├── segthor_fixed/          corrected GT (output of fix_gt.py) — shared source for experiments
└── experiments/            all our runs, grouped (keeps data/ tidy)
    ├── original/           {train,val,results}   (sliced from part1)
    ├── baseline/           {train,val,results}   (sliced from segthor_fixed)
    └── hu_window/          {train,val,results}   (sliced from segthor_fixed)
```

Everything under `data/` is gitignored — teammates **rebuild datasets locally**
with `make`; only code is pushed.

**Label convention** (everywhere): `0=background, 1=esophagus, 2=heart, 3=trachea, 4=aorta`.
In the sliced PNGs each class `k` is stored as `k*63` (i.e. `{0,63,126,189,252}`).

---

## 2. Aorta/esophagus split (`fix_gt.py`)

### The problem
The raw part-1 ground truth has two defects (both caught by `audit_data.py`):
1. The **aorta (class 4) is merged into the esophagus (class 1)** in every patient
   → the network can never learn the aorta.
2. Every `GT.nii.gz` has its **affine stripped to identity** → it no longer shares
   the CT's world geometry, which breaks 3D metrics and stitching.

### The fix
- **Patient_07** ships a second, correct label (`GT2.nii.gz`) with the aorta
  separated and the right affine. We use it directly for Patient_07, and as a
  **held-out self-test** to measure recovery quality every run.
- For the other patients we can't use intensity (non-contrast scan: aortic blood
  ≈ soft tissue) and the organs touch, so connected components can't split them.
  Instead we use **caliber**: the aorta is a fat tube (~10–14 mm), the esophagus a
  thin one (~4 mm). A **3D distance transform** (anisotropy-aware, uses real mm)
  turns that into a scalar, and a **watershed** floods from an aortic-core seed and
  an esophagus seed, cutting at the thin neck between them.
- The corrected GT is re-stamped with the **CT affine**.

### Quality (self-test on the gold Patient_07)
```
aorta Dice ≈ 0.95   esophagus Dice ≈ 0.79
```

### Run it
```bash
python fix_gt.py --source_dir data/segthor_part1/train --dest data/segthor_fixed/train
python audit_data.py --raw_dir data/segthor_fixed/train      # should report: all checks passed
```
`--copy_ct` copies CTs instead of symlinking (use on filesystems without symlinks).

### ⚠️ Important caveat
For **19 of 20 patients the aorta/esophagus labels are geometrically *recovered***
(approximate), not manual truth — only Patient_07 is gold. This includes all 5
validation patients. `fix_gt.py` prints an **aorta-fraction outlier list**
(`<-- INSPECT`); someone should open those in ITK-SNAP and hand-correct if wrong.
Because the recovered GT is held **constant** across experiments, relative A/B
comparisons are still valid — but remember absolute Dice is measured against
approximate labels.

---

## 3. Experiment pipeline

### The idea
`experiments.py` is the **single source of truth**. Each experiment bundles its
slice-stage options (`SliceConfig`) and train-stage options (`TrainConfig`). Both
`slice_segthor.py` and `main.py` take `--experiment <name>` and read from it.
**Design rule: never delete an old code path — branch on config instead.**

Precedence everywhere: **explicit CLI flag > experiment config > built-in default.**

### The three experiments (an ablation ladder)
Each step changes exactly one thing vs the step above, so the Dice difference is attributable:

| name | GT source | intensity norm | isolates |
|------|-----------|----------------|----------|
| `original`  | `segthor_part1` (corrupted) | per-volume min-max | the literal old pipeline |
| `baseline`  | `segthor_fixed` (corrected) | per-volume min-max | `original → baseline` = the **GT fix** |
| `hu_window` | `segthor_fixed` (corrected) | mediastinal HU window | `baseline → hu_window` = the **windowing** |

The **validation split is identical** across all three (patients
`01, 11, 15, 17, 19`; 15 train / 5 val), so comparisons are fair.

### Build & train
```bash
make data-experiments                                   # build all three, OR:
make data/experiments/hu_window                         # just one

python main.py --experiment baseline  --epochs 25 --gpu
python main.py --experiment hu_window --epochs 25 --gpu
```
`--dest` defaults to `data/experiments/<name>/results/` (each experiment is
self-contained). Pass `--dest <path>` to override.

> Rebuilding a dataset: `make` treats an existing `data/experiments/<name>/` as
> done. To force a rebuild, delete it first: `rm -rf data/experiments/hu_window`.

### How to add your own experiment
1. **Register it** in `experiments.py`:
   ```python
   register(Experiment(
       name="my_technique",
       description="what it changes vs baseline",
       slice=SliceConfig(window=MEDIASTINAL),          # or window=None for min-max
       train=TrainConfig(augment=True, loss="ce"),     # your knobs
   ))
   ```
2. **Copy a Makefile rule** (→ builds into `data/experiments/my_technique/`).
3. `make data/experiments/my_technique`
4. `python main.py --experiment my_technique --gpu`

Extension points already wired for you:
- **`SliceConfig`**: `source_dir`, `window`, `shape`, `retains`.
- **`TrainConfig`**: `mode` (`full`/`partial`), `loss`, `augment`.
- **`build_loss(name, K, mode)`** in `main.py`: add `dice`/`focal` here (currently
  only `"ce"`; anything else raises a clear `NotImplementedError`).

---

## 4. First preprocessing experiment — HU windowing (`preprocessing.py`)

### Why
CT voxels are calibrated Hounsfield Units (an absolute scale), so a **fixed window
is physically meaningful across patients**. The original **per-volume min-max is
not**: one high-attenuation outlier (metal, contrast, artifact) blows up the range
and crushes soft tissue into a few grey levels. In our data this is real — e.g.
Patient_02 reaches ~26,600 HU while its 99th percentile is ~370 HU, so under
min-max the mediastinal soft tissue that separates esophagus/aorta/heart collapses
to ~2–4 grey levels (worst for the **esophagus**, our hardest class).

Measured effect on the same Patient_02 slice: min-max → `max=18, std=4`; HU window
→ full `0–255, std=66`. Same tissue, ~15× more contrast, and now **consistent
across patients**.

Research basis: fixed windowing is standard clinical CT reading; nnU-Net likewise
clips CT intensities before normalization (Isensee et al., *Nature Methods* 2021).

### What it does
`apply_hu_window(volume, level, width)` clips to `[level-width/2, level+width/2]`
and scales to uint8 anchored to those **fixed** bounds. Default is the clinical
**mediastinal / soft-tissue window** (`level=40, width=400` → clip `[-160, 240]`).

### Trade-off
A soft-tissue window maps the air-filled **trachea lumen to 0** (like external
air), so the trachea is learned from its wall/shape. If trachea recall suffers, try
the lung window — it's already defined (`CT_WINDOWS["lung"]`), e.g.:
```bash
python slice_segthor.py --experiment hu_window --window_level -600 --window_width 1500 --dest_dir data/experiments/lung_window
```

### A/B it against the baseline
`baseline` (min-max) vs `hu_window` (windowed) on the same corrected GT and same
val split — that difference is the windowing effect. To go back to min-max ad-hoc:
`slice_segthor.py ... --legacy_norm`.

---

## 5. Seeding / reproducibility (`main.py`)

Training is now **reproducible** so an A/B difference reflects the technique, not
run-to-run noise.

- `--seed` (default **42**) → `seed_everything()` seeds `random`, `numpy`, `torch`
  (+CUDA) and sets cuDNN deterministic. The train `DataLoader` uses a seeded
  generator + `seed_worker` so shuffle order (and future augmentation) is fixed.
- Verified: same seed → identical Dice; different seed → different Dice.

**Two different seeds — don't confuse them:**
- `slice_segthor.py --seed` (default **0**) chooses the **validation split** at
  slice time. Leave it at 0 so all experiments share the same split.
- `main.py --seed` (default **42**) controls the **training** RNG.

**To measure variance:** run each experiment with a few seeds and average:
```bash
for s in 42 43 44; do python main.py --experiment hu_window --seed $s --epochs 25 --gpu \
  --dest data/experiments/hu_window/results_seed$s; done
```
Same-machine reproducibility is guaranteed; exact numbers can differ across
GPU/CPU hardware (normal — it doesn't affect within-machine comparisons).

---

## 6. Full workflow from a fresh clone

```bash
# 1. environment
git clone <repo> && cd <repo>          # (or: git checkout fix-data && git pull)
git submodule update --init
python -m venv ai4mi && source ai4mi/bin/activate
python -m pip install -r requirements.txt

# 2. data  (data/segthor_part1 must already be present — see readme.md "Getting the data")
make data/segthor_fixed          # corrected GT (skip if data/segthor_fixed exists)
make data-experiments            # build original / baseline / hu_window

# 3. sanity check the data before spending GPU time
python audit_data.py --raw_dir data/segthor_fixed/train --sliced_dir data/experiments/baseline

# 4. train (results -> data/experiments/<name>/results/)
python main.py --experiment baseline  --epochs 25 --gpu
python main.py --experiment hu_window --epochs 25 --gpu
```

`--gpu` uses CUDA if available, else Apple MPS, else CPU. If you hit a
"cannot pickle" DataLoader error, run with `num_workers=0` (see `readme.md` known
issues) — it's set to 5 in `main.py:setup`.

---

## 7. Open items / TODO for whoever takes over

1. **Data augmentation is NOT wired yet.** The `augment` flag flows from
   `TrainConfig` → `SliceDataset`, but `SliceDataset.__getitem__` applies no
   transforms, and the image and GT transforms are applied **independently** — so
   a coupled spatial augmentation (rotate/flip/elastic must apply the *same*
   geometry to image and mask) can't be expressed yet. **Needs a `__getitem__`
   refactor**: apply a joint spatial transform, train-split only (never on val).
   Same story for the unused `equalize` flag.
2. **3D metrics** (rubric requires 3D). The in-loop Dice is 2D per-slice. Build a
   metrics step that consumes `data/experiments/*/results/best_epoch/val/`, stitches
   with `stitch.py`, and writes an `.npz` per experiment (Assignment-3 format).
3. **More losses** (Dice / focal / combos): add a branch in `build_loss` and set
   `TrainConfig.loss`.
4. **Inspect the flagged patients** from `fix_gt.py` in ITK-SNAP and hand-correct
   any bad aorta recoveries.
5. **`num_workers`** is hardcoded to 5 — parametrize if it causes issues on some
   machines.

---

## 8. Quick reference

```bash
# repair GT
python fix_gt.py --source_dir data/segthor_part1/train --dest data/segthor_fixed/train
# audit (read-only)
python audit_data.py --raw_dir data/segthor_fixed/train --sliced_dir data/experiments/<name>
# build a dataset
make data/experiments/<name>
# train
python main.py --experiment <name> --epochs 25 --seed 42 --gpu
# ad-hoc slicing overrides (bypass a config)
python slice_segthor.py --experiment hu_window --window_level 40 --window_width 400 --dest_dir <dir>
python slice_segthor.py --experiment baseline  --legacy_norm --dest_dir <dir>
```
