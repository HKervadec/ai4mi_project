# Decisions

Current choices live in `configs/current.yaml`. Nothing here is final: to change or undo a
choice, edit the line in `current.yaml` and add a new entry below.

## Assignment options (at least 5)
| Option | Owner | Status | Chosen? |
|---|---|---|---|
| Additional pre-processing | Elena | done on holdout40: HU window, resampling, spacing / crop, lung window channel, narrower window; see [Best so far](#best-so-far) | native resolution: proposed, pending a 2nd seed |
| Data augmentation (online or offline) | | not started | |
| 2.5D network | | implemented (`configs/experiments/enet_25d.yaml`), not run | |
| Different optimizer | | not started | |
| Non-CNN architecture (Transformer / ViT) | | not started | |
| Different network architecture (modular) | | not started | |
| Post-processing | Elena | implemented (`largest_cc`, options `min_fraction`, `skip`); experiments `post_lcc_min_fraction`, `post_lcc_skip_esophagus` via `postprocess_run.py`, and `post_native_lcc_*` on the native-resolution run | keep components ≥10% of the largest (`min_fraction: 0.1`): best over 2 seeds at native resolution (HD95 21.2 → 12.5 mm); proposed |
| Different loss function | | not started | |
| Other regularizer at the loss level | Britt | implemented (`boundary`: Sobel edge L1, option `per_class`); experiments `boundary{,_w1,_w5}` (merged foreground) and `boundary_per_organ{,_w1,_w5}` (per organ), weights 0.1 / 1 / 5, not run | |
| Pre-training / hybrid supervision (public dataset) | | not started | |

<a id="best-so-far"></a>
## Best so far (2026-10-04)

Only runs on the full data (`holdout40`: 32 train / 8 val patients) are compared; older `holdout`
rows in RESULTS.md use 15 train patients and older metric code and are not comparable. All runs
below are on a MacBook (mps), seed 42 unless noted, scored in 3D on the native GT grid. Δ is against
`current` (0.98 → 1.95 mm resampling, 256×256, HU window [40, 400], ENet + CE, 25 epochs).
"Up" = validation patients whose mean Dice went up; p = paired Wilcoxon over the 8 patients
(0.008 is the smallest possible with 8 patients).

| Experiment | What changes | 3D Dice | Δ | Up | p | Esophagus | Trachea | Verdict |
|---|---|---|---|---|---|---|---|---|
| `spacing_native_384`, seeds 42 + 43 | 0.98 mm in-plane, 2.0 mm z, 384×384 crop | **0.836** (0.849 / 0.822) | **+0.049** | 8/8, 7/8 | 0.008, 0.016 | **0.659** (+0.098) | 0.861 (+0.053) | **best in both seeds; proposed as new `current`** |
| `spacing_1.95_crop192` | 1.95 mm, 2.0 mm z, 192×192 crop (same field of view as native) | 0.803 | +0.016 | 7/8 | 0.016 | 0.611 | 0.838 | small gain, and 25% faster than `current` |
| `spacing_1.5_256` | 1.5 mm in-plane, same pixel count | 0.801 | +0.014 | 6/8 | 0.08 | 0.612 | 0.819 | same direction, borderline |
| `lung_window_channel` | lung window as 2nd input channel | 0.790 | +0.003 | 4/8 | 0.74 | 0.582 | 0.839 | trade-off: trachea up, heart/aorta down |
| `enet_25d` (David) | 2.5D input, 1 neighbour per side | 0.789 | +0.002 | | | 0.570 | 0.822 | no difference yet |
| `augmentation-new-data` (Githa) | affine/roll/elastic/intensity | 0.784 | −0.003 | | | 0.583 | 0.801 | no difference yet (see below) |
| `spacing_1.95_z2` | only z: 2.0 instead of 2.5 mm | 0.780 | −0.007 | 5/8 | 1.0 | 0.564 | 0.792 | no effect; doubles as a noise estimate |
| `no_resampling` | per-slice resize instead of resampling | 0.767 | −0.020 | 4/8 | 0.84 | 0.597 | 0.782 | resampling kept (see below) |
| `window_narrow` | window [40, 300] instead of [40, 400] | 0.762 | −0.025 | 1/8 | 0.016 | 0.524 | 0.755 | worse; keep [40, 400] |
| `no_hu_window` | min-max instead of HU window | 0.751 | −0.036 | 0/8 | 0.008 | 0.532 | 0.828 | HU window confirmed |
| `full_data_baseline` | starter code: no window, no resampling | 0.703 | −0.084 | 0/8 | 0.008 | 0.468 | 0.786 | reference |

**Why native resolution works.** The model sees the scan at the scanner's own pixel size
(~0.98 mm) instead of a 2× downsampled copy, so detail is no longer thrown away before training.
- Small organs get enough pixels: the esophagus (1–2 cm across) is 5–10 px wide at 1.95 mm
  and 10–20 px at 0.98 mm, and the 1–2 mm fat plane that separates it from aorta and trachea is
  below one pixel at 1.95 mm.
- ENet downsamples 8× internally: its coarsest cells are ~16 mm at 1.95 mm (the size of the whole
  esophagus) and ~8 mm at 0.98 mm.
- A one-pixel boundary error costs 0.98 instead of 1.95 mm (NSD@1mm 0.440 → 0.559).
- No loss mapping back to the native grid: pushed through the old pipeline, even the GT itself
  only reaches esophagus Dice 0.922 (mean 0.953); at 0.98 × 0.98 × 2.0 mm the round trip is lossless.
- The gain grows as organs get smaller (esophagus +0.124, trachea +0.065, aorta +0.046, heart
  +0.010, seed 42), and it comes from in-plane resolution, not slice thickness (`spacing_1.95_z2`:
  no effect). 1.95 → 1.5 → 0.98 mm gives 0.787 → 0.801 → 0.836 (2-seed mean).
- About two thirds of the gain is resolution, one third the tighter crop. The native run also
  crops to a 376 mm field of view (vs 499 mm). At 1.95 mm the same crop (`spacing_1.95_crop192`)
  gives 0.803, so crop ≈ +0.016 and resolution ≈ +0.033 (0.803 → 0.836). The crop part is within
  seed noise (see caveats), the resolution part is not.
- Cost: 2.6× training time (571 vs 222 min on a MacBook), and still improving at epoch 24/25.

**Other findings.**
- *HU window [40, 400]* is worth +0.036 (all 8 patients) and halves HD95; the only organ that
  does slightly better without it is the trachea (+0.020), whose air lumen the window clips to 0.
- *Resampling* mainly protects the odd scan: without it the mean drops 0.020, almost all from
  Patient_35, the only scan with 1.37 mm pixels (−0.147); HD95 is worse in 7/8 patients.
  Window and resampling together are worth more than the sum of the two (−0.084 vs −0.056).
- *Lung window* helps the trachea (+0.031, 6/8) but costs heart and aorta (each worse in 7/8).
  Retest at native resolution (`lung_window_native_384`) before dropping it.
- *Post-processing* (`largest_cc`) barely moves Dice but removes far-away stray blobs that blow
  up HD95 (e.g. a heart fragment 126 mm away in Patient_30). Keep every component ≥10% of the
  largest (`min_fraction: 0.1`). Mean over the two native seeds (Dice / HD95 / ASSD):
  raw 0.836 / 21.2 / 3.26 mm, `min_fraction` **0.840 / 12.5 / 2.44 mm**, skip-esophagus
  0.835 / 18.4 / 2.99 mm. Skip-esophagus looked best on seed 42 alone (HD95 9.2 mm), but seed 43
  predicts small flat esophagus blobs ~6 cm in front of the esophagus in every patient
  (esophagus HD95 70 mm), which only a filter on the esophagus removes; plain largest-CC also
  cuts real aorta pieces there (aorta HD95 13.2 → 20.5 mm). At 1.95 mm `min_fraction` was also
  best (HD95 14.5 → 11.2 mm).
- *Augmentation and 2.5D* show no difference at 25 epochs, but every run peaks at epoch 22–24,
  so models that learn slower (augmentation) are cut off early. Rerun with longer training before
  concluding. The `roll` augmentation wraps the image around (anatomically impossible) and
  should become a translation.

**Caveats.** 8 patients, and mostly one seed. Training randomness is larger than it looked:
the two native seeds differ by 0.027 in mean Dice, and seed 43 is lower in *all* 8 patients. A
seed shifts every patient together, so the paired per-patient p-values above only cover the
choice of patients, not training noise: a single-seed difference below ~0.03 is not reliable
however many patients agree. `current` itself has only one seed. The best epoch is chosen on the same 8 patients it is scored on. Distance metrics are NaN
(and dropped from the mean) when an organ is missing from the prediction, which flatters runs
that miss an organ; NSD's 1 mm tolerance is below the 1.95 mm pixel size of most runs.

**Proposed change to `current.yaml`** (not applied yet; group decision):
`data.target_spacing: [0.98, 0.98, 2.0]`, `data.shape: [384, 384]` and
`eval.postprocess: [{name: largest_cc, min_fraction: 0.1}]`. The second seed confirmed the
gain (+0.035 and +0.062 over `current`). Every other experiment then has to be rerun on it.

## Log
<!--
## YYYY-MM-DD — <setting>: <old> -> <new>
Why: <evidence, e.g. RESULTS.md rows>
Revisit if: <condition>
-->

## 2026-09-16 — data.gt: watershed_refined
Why: best Patient_07 self-test (aorta 0.9995 / esophagus 0.998 vs 0.95 / 0.79), see HANDOFF.md §2.
Revisit if: visual inspection shows bad splits (e.g. Patient_05).

## 2026-09-29 — data.gt: data/gt/watershed_refined -> data/segthor_full; data.split: holdout -> holdout40
Why: the full SegTHOR training release (40 patients) arrived. Its GT is clean: all 5
labels present (aorta no longer merged into esophagus) and the GT affine matches the CT,
so no GT fix is needed. Patients 01–20 have identical CTs to segthor_part1 and their GT
agrees with watershed_refined (Dice 0.99–1.0 per organ, checked on Patient_01/07/15),
which confirms the fix. New splits, seed 0: `splits/cv5_40.json` (5 folds of 8 val
patients) and `splits/holdout40.json` (= cv5_40 fold 0, 32 train / 8 val).
Results in RESULTS.md from before this date use 15 train patients and are not comparable
to runs on the full data; rerun `current` and `full_data_baseline` as new reference points.
Old splits (`holdout`, `cv4`) are kept so old runs stay reproducible; an experiment that
still sets `data.gt: data/gt/...` must also set `data.split: holdout` or `cv4`.
Revisit if: a separate test release changes which patients we can use for validation.
