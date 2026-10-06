# Metrics plan

How we evaluate our SegTHOR organ segmentations, how each metric is implemented,
why we picked it, and what to look for in the results. The choices follow the
[Metrics Reloaded](https://metrics-reloaded.dkfz.de/metric-library) recommendations
for semantic segmentation of anatomical structures.

SegTHOR has 5 classes: `background, esophagus, heart, trachea, aorta`
(`CLASS_NAMES` in `segpipe/data.py`). All averages below leave out the background.

---

## 1. Two levels of evaluation

| Level | When | What | Where |
|---|---|---|---|
| **2D, per slice, per epoch** | during training | Dice, IoU — cheap monitoring, used to pick the best epoch | logged by `main.py` / `segpipe/train.py`; summarised by `evaluate.py` |
| **3D, per patient volume** | after training, on the best epoch | Dice + boundary metrics, the numbers we actually report | `segpipe/evaluate.py`, driven by `run.py` |

The best epoch is still selected on 2D validation Dice; the 3D metrics are the
final quality measure. This separation matters: a per-slice 2D score averaged over
slices is **not** the same as a true volumetric score, and boundary error only
makes sense measured on the whole 3D organ.

---

## 2. Which metrics, and the metric "families"

The assignment asks for **at least three different *types* of metric**. Dice and
IoU are the *same* family (both overlap-based and monotonically related), so they
count as one type. We therefore use three genuinely distinct families:

| Family | Metric(s) | Question it answers |
|---|---|---|
| **Overlap** | **Dice** (DSC), IoU | Do the predicted and true regions cover the same voxels? |
| **Boundary — distance** | **HD**, **HD95**, **ASSD** | How far, in mm, is the predicted surface from the true surface? |
| **Boundary — tolerance** | **NSD** | What fraction of the surface is within a clinically acceptable margin? |

All boundary metrics are computed in **millimetres**, using the real voxel
spacing, so they are physically meaningful and comparable across patients.

---

## 3. Each metric: definition, implementation, interpretation

All 3D metrics live in the `METRICS` registry in `segpipe/evaluate.py` and share
one signature: `fn(pred_mask, gt_mask, spacing) -> float`, called once per organ
`k`. The four boundary metrics all derive from a **single pass over each organ's
surface voxels** (`_surface_distances`): we take the surface (foreground minus its
binary erosion), scale the voxel coordinates by the spacing to get mm, and build a
KD-tree to find, for every surface voxel, the nearest surface voxel on the other
mask. From those two distance arrays (`pred→gt` and `gt→pred`) every boundary
number is cheap to derive.

### Dice (DSC) — overlap
- **Definition:** `2·|A∩B| / (|A|+|B|)`, in [0, 1], higher is better.
- **Implementation:** voxel counts weighted by physical voxel volume (mm³); for
  Dice the weighting cancels, but it keeps the metric in physical units.
- **Interpretation:** the standard overlap score; robust and easy to compare, but
  it saturates on large organs and says nothing about *where* errors are.

### IoU (Jaccard) — overlap
- **Definition:** `|A∩B| / |A∪B|`.
- **Interpretation:** same information as Dice (monotonic with it), reported for
  completeness. It penalises errors slightly more harshly than Dice.

### HD — max Hausdorff distance (boundary, mm)
- **Definition:** the largest of all nearest-surface distances in either direction
  — the single worst boundary point.
- **Interpretation:** worst-case error. **Very sensitive to outliers**: one stray
  predicted voxel far from the organ blows it up. We keep it *specifically* to
  contrast with HD95 (see §5).

### HD95 — 95th-percentile Hausdorff (boundary, mm)
- **Definition:** as HD, but the 95th percentile of the distances instead of the
  max.
- **Interpretation:** the robust version of HD — Metrics Reloaded recommends it
  over raw HD for exactly this reason. It captures "how bad are the bad regions"
  without being dominated by a single noisy voxel.

### ASSD — average symmetric surface distance (boundary, mm)
- **Definition:** the mean of all nearest-surface distances in both directions.
- **Interpretation:** the *typical* boundary error over the whole surface, not the
  worst. Complements HD95: HD95 is the tail, ASSD is the average.

### NSD — Normalised Surface Dice (boundary tolerance, [0, 1])
- **Definition:** the fraction of surface voxels (both directions) whose nearest
  distance is within a tolerance `τ`. We report two tolerances
  (`segpipe/evaluate.py`): `nsd` at `NSD_TAU_MM = 1.0` mm (strict) and `nsd3` at
  `NSD3_TAU_MM = 3.0` mm. The GT slice spacing is 2.0 or 2.5 mm, so at 1 mm a
  surface one slice off in z already counts as wrong; 3 mm forgives a one-slice
  error for every patient (2 mm would not, for the 2.5 mm volumes).
- **Interpretation:** "what share of the boundary is good enough?" It reflects
  clinical usability — small sub-tolerance wobbles don't count against you, but
  real deviations do. Higher is better.

---

## 4. How it runs and what gets saved

Enable the metrics per experiment in the config (`configs/current.yaml`):

```yaml
eval: {metrics_3d: [dice, hd, hd95, assd, nsd, nsd3]}
```

Then:

```bash
python run.py --config configs/current.yaml
```

`run.py` → `segpipe/evaluate.py`:
1. stitches the best epoch's val predictions into per-patient `.nii.gz` volumes
   (`stitch.py`);
2. reads the matching ground-truth volume and its **voxel spacing straight from
   the NIfTI header** (`get_zooms()`), so anisotropy (SegTHOR is ~0.98 mm in-plane
   vs 2.0–2.5 mm through-plane) is handled correctly;
3. scores every organ of every patient and writes:
   - `run_dir/metrics_3d/<metric>.npz` — raw per-patient × per-class arrays (for
     plotting / statistics / `compare.py`);
   - a JSON summary with `mean`, `per_class`, and `per_patient`.

Aggregation uses `np.nanmean`. A boundary metric is **NaN only when an organ is
absent from both** the prediction and the ground truth (nothing to measure), so
that case is left out of the mean. When the organ is absent from **only one** of
them (a missed organ, or one predicted that is not there) it gets the **worst
case**: the volume's diagonal in mm for HD / HD95 / ASSD, and 0 for NSD. Earlier
versions returned NaN here too, which silently dropped missed organs from the
mean and made a model that misses the esophagus look better than one that
segments it badly.

---

## 5. What could be interesting in the findings

- **HD vs HD95 gap = outlier diagnosis.** A large gap between max-HD and HD95 for
  an organ means a few stray voxels far from the structure (false-positive
  islands), not a systematically wrong boundary. This is a concrete argument for
  *why* HD95 is the metric to trust, and it can motivate a post-processing step
  (e.g. keep-largest-component) — check whether it closes the HD–HD95 gap while
  leaving Dice unchanged.
- **Per-organ difficulty.** Expect the **esophagus** to be the hardest: it is
  thin, low-contrast, and variable, so it should show the lowest Dice/NSD and the
  highest surface distances. **Heart** and **aorta** are large and high-contrast
  and should score well. Reporting per-organ (not just the mean) tells the real
  story.
- **Overlap can hide boundary problems.** An organ can have a good Dice but a poor
  HD95/ASSD if the errors sit on the boundary — thin or elongated structures are
  prone to this. Comparing the overlap family against the boundary family across
  organs is where the "use several metric types" argument pays off.
- **NSD as the practical score.** Because NSD forgives sub-`τ` wobble, it often
  separates "clinically fine" from "clinically off" better than Dice does. Worth
  reporting how NSD ranks the experiments vs how Dice ranks them — if they
  disagree, that disagreement is a finding.
- **Comparing experiments.** The same metric panel across `original`,
  `refined_window`, `watershed_*` etc. shows whether a preprocessing/postprocessing
  choice improves *overlap*, *boundary accuracy*, or both — they don't always move
  together.

---

## 6. Why these, and caveats

**Why chosen.** Metrics Reloaded advises pairing an overlap metric with a
boundary metric for anatomical segmentation, computing distances in physical units
on surfaces, and preferring HD95 over raw HD. Our panel does exactly that, adds
ASSD (average boundary error) and NSD (tolerance-based, clinically oriented), and
deliberately keeps raw HD only as a contrast for HD95.

**Caveats to keep in mind when reading the numbers.**
- **Absent organs → NaN, missed organs → worst case.** If a patient's GT lacks an
  organ and the model predicts none, that organ's boundary metrics are NaN and
  excluded from the mean; if the model predicts it anyway, it scores the worst case. See the known SegTHOR issue that
  part-1 ground truth is missing the aorta — those patients will read NaN for
  aorta, so report how many patients contribute to each organ's average.
- **NSD tolerance is a choice.** `τ = 1.0 mm` is a reasonable default, but a thin
  organ (esophagus) and a large one (aorta) arguably deserve different tolerances.
  If we want per-organ `τ`, `NSD_TAU_MM` in `segpipe/evaluate.py` is the single
  place to change (currently one global value).
- **HD/HD95 depend on spacing accuracy.** They are read from the GT NIfTI header;
  if an affine were stripped or wrong, distances would be off — worth a sanity
  check that spacings look like `~(0.98, 0.98, 2.0–2.5)` mm.
