# Experiment pipeline

One config-driven path for every experiment: **config → train → 3D evaluation → shared tables**.
Every run gets its own directory with its exact config, provenance and results, can be resumed
after a crash, and logs to Weights & Biases.

- Adding a model, loss, augmentation or dataset: [extending.md](extending.md)
- What each metric means and how tables are built: [metrics.md](metrics.md)

The original `main.py` / `stitch.py` / `plot.py` still work and are untouched; `src/` reuses
`ENet.py`, `ShallowNet.py`, `losses.py`, `dataset.py` and `utils.py` from the repo root.

## One-time setup (per person)

```bash
# 1. environment (adds pyyaml + wandb to the ai4mi env)
module load 2024 Anaconda3/2024.06-1
source "$EBROOTANACONDA3/etc/profile.d/conda.sh" && conda activate ai4mi
pip install -r requirements.txt

# 2. W&B: log in once; the key is stored in ~/.netrc and picked up by jobs
wandb login

# 3. run storage on scratch (checkpoints would fill the 20 GB home quota)
mkdir -p /scratch-shared/$USER/ai4mi_project/runs
ln -sfn /scratch-shared/$USER/ai4mi_project/runs runs
```

Scratch is wiped after 14 days without access. Start the self-resubmitting keepalive once
(`sbatch jobsAndOutputs/pipeline/jobs/keepalive_scratch.job`); it touches `/scratch-shared/$USER/ai4mi_project/`
every 10 days. `squeue -u $USER` should always show a pending `keepalive_scratch` job (reason `BeginTime`).
The small result files are also copied into the git-tracked `metrics/` dir, so tables survive
a wipe even if checkpoints don't.

Also data preparation, as before: `make data/SEGTHOR` (see the main readme).

## Running experiments

All compute goes through Slurm. **Never train or evaluate on the login node.** Submit from the repo root.

```bash
# 0. tests (CPU, ~5 min): legacy parity, resume, stitching, metric conventions
sbatch jobsAndOutputs/pipeline/jobs/test.job

# 1. smoke test on the cheapest GPU: 2 epochs on 16 slices + full 3D eval (~5 min)
sbatch --export=ALL,CONFIG=configs/segthor_enet_ce.yaml jobsAndOutputs/pipeline/jobs/smoke.job

# 2. the real run: train, then evaluate in 3D
sbatch --job-name=segthor_enet_ce --export=ALL,CONFIG=configs/segthor_enet_ce.yaml \
       jobsAndOutputs/pipeline/jobs/train.job

# 3. overrides without a new file (each distinct config gets its own run dir, see below).
#    SET is split on spaces, so values must not contain any: tags=[a,b], not [a, b]
sbatch --job-name=enet_lr1e3 --export=ALL,CONFIG=configs/segthor_enet_ce.yaml,SET="optim.kwargs.lr=1e-3 seed=1" \
       jobsAndOutputs/pipeline/jobs/train.job

# 4. several configs / seeds as one array job: one "<config> [key=value ...]" per line
printf 'configs/segthor_enet_ce.yaml seed=0\nconfigs/segthor_enet_ce.yaml seed=1\nconfigs/segthor_enet_ce.yaml seed=2\n' > sweeps/enet_seeds.txt
sbatch --array=0-0 --export=ALL,SWEEP=sweeps/enet_seeds.txt jobsAndOutputs/pipeline/jobs/sweep.job   # one task first
sbatch --array=1-2 --export=ALL,SWEEP=sweeps/enet_seeds.txt jobsAndOutputs/pipeline/jobs/sweep.job

# 5. comparison table across everyone's runs (after git pull); cheap, fine on the login node
python -m src.aggregate
```

Slurm output lands in `jobsAndOutputs/pipeline/outputs/<job-name>_<jobid>.out`; the same log is
in the run directory as `train.log` / `eval.log`. `train.job` asks for 2 h on `gpu_a100`, which is
plenty for ENet (about 15 min). Lower `--time` for small models, since billing counts the requested time.

Inside a job the commands are simply:

```bash
python -O -m src.train --config configs/segthor_enet_ce.yaml [--set key=value ...] [--smoke] [--force]
python -O -m src.evaluate --run runs/segthor_enet_ce/seed0
python -m src.run --config configs/segthor_enet_ce.yaml      # prints the run directory
```

## Configs

`configs/base.yaml` lists **every** setting with its default and a comment. An experiment file
only contains what it changes:

```yaml
# configs/segthor_enet_ce.yaml
experiment: segthor_enet_ce          # names the run directory; make it unique and descriptive
notes: ENet + cross-entropy, course baseline settings
model: {name: enet, kwargs: {kernels: 8, factor: 2}}
loss:  {name: cross_entropy, kwargs: {idk: [0, 1, 2, 3, 4]}}
```

Resolution order is `base.yaml` → experiment file → `--smoke` settings → `--set` overrides.
A key that does not exist in `base.yaml` is an error, so typos fail immediately. The exception is
anything under a `kwargs:` block, which is passed straight to the component (model, loss, optimizer...).

Common switches:

| to change | set |
|---|---|
| dataset / preprocessing variant | `data.root` (+ `num_classes`, `class_names`, `label_scale` if they differ) |
| architecture | `model.name`, `model.kwargs` |
| loss | `loss.name`, `loss.kwargs` |
| optimizer / LR schedule | `optim.name/kwargs`, `scheduler.name/kwargs` (`none`, `cosine`, `step`) |
| length, batch size | `train.epochs`, `data.batch_size` |
| seed | `seed` (a separate run directory per seed) |
| selection metric | `train.select_metric` (`val_dice_fg` default; `val_dice_legacy_fg` = the course metric) |

Name experiments `<dataset>_<model>_<loss>[_<variant>]`, e.g. `segthor_unet_dice_ce_aug`.

## The run directory

`runs/<experiment>/seed<seed>/` (on scratch through the `runs` symlink):

```
config.yaml            fully resolved config: exactly what ran
manifest.json          status, user, host, Slurm job id, git commit (+ dirty flag), python/torch/CUDA/GPU,
                       command line, start/end times, every resume, W&B id and URL
train.log, eval.log    everything that was printed
epochs.csv             one row per epoch: losses, per-class and fg Dice (train/val), lr, time, is_best
summary.json           best epoch and its metrics; `eval` block added by src.evaluate
plots/curves.png       loss and validation Dice curves
checkpoints/last.pt    model + optimizer + scheduler + RNG state, every epoch (for resuming)
checkpoints/best.pt    best weights (state_dict) by train.select_metric
checkpoints/best_model.pkl   whole pickled network (the submission format)
val_counts_best.npz    per-slice per-class intersection/areas at the best epoch (recompute any 2D metric)
predictions/val/*.png  2D predictions of the best model (grey = class * label_scale, viewer-compatible)
volumes/val/*.nii.gz   predictions stitched back onto the original CT grid
eval/metrics_3d.csv    per patient and class: Dice, HD95, ASSD, voxel counts
eval/val/{dice,hd95,assd}.npz   submission format: patient -> (K,) array
wandb/                 local W&B files
.done                  marker: training finished
```

What happens when you (re)submit the same config:

| existing run dir | behaviour |
|---|---|
| none | fresh run |
| same config, finished (`.done`) | nothing to do, exits immediately |
| same config, unfinished (crash, timeout, `scancel`) | **resumes automatically** from `last.pt`: same RNG stream, result identical to an uninterrupted run |
| different config | refuses: rename `experiment`, or pass `--force` (moves the old dir to `seed0.old-<timestamp>`, never deletes) |

"Same config" means the same hash of the resolved config (`notes` and `wandb` settings excluded).
Changing code without changing the config does **not** create a new run, so use `--force` or a new
experiment name when you re-run after a code change. `manifest.json` records the git commit either way.

Smoke runs go to `runs/_smoke/...`, always start fresh and are never copied to `metrics/`.

## Sharing results

Two channels, both automatic:

1. **`metrics/<experiment>/seed<seed>/`** (git-tracked, a few KB): `config.yaml`, `manifest.json`,
   `summary.json`, `epochs.csv`, `metrics_3d.csv`, copied at the end of training and again after
   evaluation. Commit these with your code change (`git add metrics/<experiment>`); every run has a
   unique path, so teammates never conflict. `python -m src.aggregate` turns all of them into
   `metrics/comparison.md` (mean ± std over seeds, paste-ready) and `metrics/comparison_runs.csv`.
2. **Weights & Biases**: project `ai4mi-segthor`, one run per experiment/seed, grouped by `experiment`
   and tagged with model, loss, dataset and user. Logged: the full config, every `epochs.csv` column
   per epoch, prediction/ground-truth overlays for 4 fixed validation slices every `wandb.image_every`
   epochs, and the 3D evaluation results as `eval3d/*` in the run summary.
   Set `wandb.entity` in `configs/base.yaml` to the team's entity so all runs land in one project.

W&B is **fail-soft**: if it can't start or loses connection (no login, no network), the run logs a
warning and continues with local files only. Nothing ever blocks on it. To upload runs made offline:
`wandb sync runs/<experiment>/seed0/wandb/offline-run-*`. `--set wandb.mode=disabled` turns it off.

## Reproducibility notes

- Seeds: python, numpy, torch and CUDA are seeded from `seed`; DataLoader workers derive theirs from it.
  On CPU a run is bit-exact; on GPU cuDNN / ENet's max-unpooling are not deterministic, so expect
  small run-to-run differences. That is why the tables report mean ± std over seeds.
- Data split: the train/val patient split is fixed at slicing time (`slice_segthor.py --retain 5`,
  seed 0; val = Patient_01, 11, 15, 17, 19), so all experiments on the same `data.root` share it.
- `python -O` (used in the jobs) disables the course code's many `assert`s for speed; the pipeline
  does not rely on asserts.
