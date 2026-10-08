# Final presentation — notes

Working notes for the final presentation (13 / 15 Oct, 10 min + 3 min questions). Everything is
taken from RESULTS.md, DECISIONS.md, the configs and the git log as of 2026-10-07. Numbers are
3D on the native GT grid, `holdout40` split (32 train / 8 val patients), seed 42 unless noted.

**Rule from the brief:** the numbers on the slides must be the numbers in the submission. Freeze one
final config and one results table before making the result slides.

---

## 1. Where the midterm left off (≈ 21 Sep)

What we had shown then (on the 20-patient part-1 data, 15 train / 5 val):

- GT repair: aorta split out of the esophagus label (watershed, `watershed_refined`), affine restored.
- HU window [40, 400] instead of per-volume min-max.
- Resampling to 1.95 × 1.95 × 2.5 mm with crop/pad instead of per-slice resize.
- 3D metrics (Dice, HD95, ASSD, NSD) on the stitched volumes.
- First tries of augmentation, 2.5D and transformer-in-ENet, all on the old split.

**Check with the group:** is this what was presented? Adjust the start of the story if not.

## 2. What changed since then (the story)

The thread through everything: **the esophagus is the hard organ** (small, thin, low contrast, touches
aorta and trachea). Almost every kept change helps it most.

| Step | Mean 3D Dice | Esophagus Dice | HD95 (mm) | Source row |
|---|---|---|---|---|
| Starter code on full data (no window, no resampling) | 0.703 | 0.468 | 23.3 | `full_data_baseline` |
| Midterm pipeline on full data (window + 1.95 mm resampling) | 0.787 | 0.561 | 14.5 | `current` holdout40 |
| + native resolution 0.98 mm, 384×384 (2 seeds) | 0.836 | 0.659 | 21.2 | `spacing_native_384` |
| + post-processing (keep CCs ≥10% of largest) (2 seeds) | 0.840 | 0.666 | 12.5 | `post_native_lcc_min_fraction` |
| + GPU augmentation (2 seeds), post-processed | 0.863 | 0.720 | 8.4 | `augmentation-native384` |
| + focal + Dice loss (1 seed), post-processed | 0.879 | 0.767 | 8.6 | `loss_focal_dice` |
| Combination (focal+Dice + sampler, ± cosine, ± boundary) | pending | | | `combo_*` |

Note: `configs/current.yaml` is now native resolution + augmentation + post-processing (=
`augmentation-native384`). The `current` row in RESULTS.md is still the **old** 1.95 mm pipeline, so
say "midterm pipeline" on the slides, not "current".

### 2a. Data: full 40-patient release (29 Sep)
- Clean GT (aorta present, affine correct); 01–20 agree with our repaired GT (Dice 0.99–1.0), which
  validates the midterm GT fix.
- New split `holdout40` (32/8) and `cv5_40`. Every old `holdout` run is no longer comparable.

### 2b. Preprocessing ablation on the full data (Elena)
Kept:
- **HU window [40, 400]**: removing it costs −0.036 Dice (0/8 patients better, p = 0.008) and
  doubles HD95 (14.5 → 29.4 mm).
- **Resampling**: removing it costs −0.020, almost all from Patient_35 (the only 1.37 mm scan, −0.147).
  Window + resampling together are worth more than the sum (−0.084 vs −0.056).
- **Native in-plane resolution 0.98 mm, 384×384** — biggest single gain: +0.049 (both seeds, 8/8 and
  7/8 patients up), esophagus +0.098. Gain grows as organs get smaller. About ⅔ resolution, ⅓ the
  tighter crop (`spacing_1.95_crop192` = +0.016). Cost: 2.6× training time.

Thrown away:
- `window_narrow` [40, 300]: −0.025, 1/8 patients better.
- `spacing_1.95_z2` (only slice thickness 2.5 → 2.0 mm): no effect (−0.007). Shows the gain is in-plane.
- `spacing_1.5_256`: +0.014, borderline (6/8, p = 0.08); superseded by native.
- `lung_window_channel` at 1.95 mm: +0.003, trade-off (trachea up, heart/aorta down).
- `lung_window_native_384`: same Dice as native seed 42 (0.847 vs 0.849) but HD95 15.7 → 10.2 mm.
  Not kept alone; it reappears in 2.5D + lung (2e).

### 2c. Post-processing (Elena)
- Largest connected component per organ removes far stray blobs that blow up HD95 (e.g. a heart
  fragment 126 mm away in Patient_30). Dice hardly moves; HD95 roughly halves.
- **Kept:** keep every 3D component ≥10% of the largest (`min_fraction: 0.1`).
- **Thrown away:** plain largest-CC (cuts the esophagus, which is predicted as several pieces along z)
  and skip-esophagus (best on seed 42, but seed 43 predicts flat esophagus blobs ~6 cm off in every
  patient → esophagus HD95 70 mm). Good slide on *why one seed lies*.
- On every later run it gives −2 to −11 mm HD95 at +0.000 to +0.003 Dice (RESULTS.md post table).

### 2d. Data augmentation (Githa)
- Affine, elastic, brightness/contrast (+ roll), moved to the GPU (CPU augmentation was the bottleneck).
- At 1.95 mm and 25 epochs: no gain (`augmentation-new-data` −0.003; 2.0: +0.025).
- **At native resolution: +0.026** (0.836 → 0.861), esophagus 0.659 → 0.720, and the two seeds now
  agree within 0.001 (was 0.027 without augmentation). Kept.
- Per-sample vs per-batch draw: 0.865 vs 0.861 — within noise; post-processed HD95 8.2 vs 8.4 mm.
  Pick one for the final config.
- Known flaw: `roll` wraps the image around (anatomically impossible) — limitation / should be a
  translation.

### 2e. 2.5D input (David)
- 2.5D alone (slice ± 1 neighbour) at 1.95 mm: +0.002 → no effect. Old-split 3-seed run unstable.
- **2.5D + lung window** at native + augmentation: 0.870 (+0.009 over `augmentation-native384`),
  best raw HD95 of all (10.4 mm), esophagus 0.731. One seed, and 2.5D and lung window were not
  separated. Training is slow (738 min on a MacBook).

### 2f. Class imbalance: loss and sampling (Puck, David)
- At 1.95 mm: CE + Dice +0.023 (esophagus +0.065, HD95 14.5 → 12.4); CE + Tversky +0.024 but worse
  HD95 (19.9). **Tversky thrown away.**
- At native + augmentation (vs CE 0.861):
  - **Focal (γ=2) + Dice: 0.877** (+0.016), esophagus 0.767, HD95 12.8 → 8.6 post. Best single run.
  - CE + Dice: 0.873 (+0.012), esophagus 0.761, but HD95 16.8 → 10.1 post.
  - Weighted slice sampler (empty slices ×0.7, esophagus ×2): 0.870 (+0.009), best post HD95 (7.3 mm),
    but raw trachea HD95 38.9 mm (stray blobs that post-processing removes).
- All single seed.

### 2g. Architecture: transformer in ENet (Junis)
- One transformer layer inserted at bottleneck / stage1 / stage2 / decoder1, 1 vs 2 layers,
  positional embedding. 3 seeds each.
- Best: stage2 placement (3D 0.666 vs 0.569 for `current` on the same old split).
- **Thrown away / not carried forward:** only run on the old 20-patient split with the 1.95 mm
  pipeline; very seed-sensitive (e.g. stage2_2layer 0.371–0.713). Not comparable to the final numbers.
- **Check:** the `_nores` and with-resampling rows in RESULTS.md are identical number for number —
  looks like the same runs under two names. Sort out before showing.

### 2h. Implemented but not (yet) run
- Boundary regularizer (Britt): Sobel-edge L1, merged or per organ, weights 0.1 / 1 / 5. Only
  appears in `combo_full`.
- Early stopping, 50 epochs (David): `early_stopping.yaml`. Relevant: every run peaks at epoch 20–24
  of 25, so we may be under-training.
- Cosine lr decay: `schedule_cosine.yaml`, and in `combo_cosine`.
- Combination ladder (configs added 2026-10-07): `combo_core_focal` → `combo_core_ce` (loss choice)
  → `combo_cosine` → `combo_full`.

## 3. Status overview

| Idea | Owner | Status | Verdict |
|---|---|---|---|
| Full 40-patient data | all | done | kept |
| HU window [40, 400] | Elena | done | **kept** |
| Narrow window [40, 300] | Elena | done | dropped |
| Resampling + crop/pad | Elena | done | **kept** |
| Native 0.98 mm, 384×384 | Elena | done, 2 seeds | **kept** |
| 1.5 mm / z = 2.0 mm / 1.95 mm crop192 | Elena | done | dropped (explain the native gain) |
| Lung window channel | Elena | done | dropped alone; part of 2.5D + lung |
| Post-processing ≥10% of largest CC | Elena | done, 2 seeds | **kept** |
| Largest CC / skip esophagus | Elena | done | dropped |
| GPU augmentation | Githa | done, 2 seeds | **kept** |
| Augmentation per sample vs per batch | Githa / Elena | done, 2 seeds | tie — choose one |
| 2.5D (±1 slice) | David | done | dropped alone |
| 2.5D + lung window | David / Elena | done, 1 seed | candidate |
| CE + Dice | Puck | done, 1 seed | candidate (in `combo_core_ce`) |
| Focal + Dice | Puck | done, 1 seed | candidate, best single run |
| CE + Tversky | Puck | done | dropped |
| Weighted slice sampler | David | done, 1 seed | candidate (in combos) |
| Transformer in ENet | Junis | done on old split | not carried forward |
| Boundary regularizer | Britt | implemented | not run (only in `combo_full`) |
| Early stopping / 50 epochs | David | implemented | not run |
| Cosine lr | Elena | implemented | not run (in `combo_cosine`) |
| Combination ladder | Elena | configs ready | **pending — needed for final** |

## 4. Limitations (for the critical-analysis slide)

- **One seed** for loss, sampler and 2.5D + lung. Without augmentation, seeds differed by 0.027; with
  augmentation by 0.001–0.010. Differences of ~0.01 between candidates are within that range.
- **8 validation patients, no test set**, and the best epoch is chosen on those same 8 patients →
  optimistic numbers. Cross-validation (`cv5_40`) exists but was never run.
- **Under-training:** best epoch is 20–24 of 25 in almost every run (early stopping not tested).
- **Esophagus still worst** (0.77 vs 0.93 heart); predicted in fragments along z, which 2D slices
  can't fix and post-processing has to work around.
- **Distance metrics:** HD95 is NaN when an organ is missing from the prediction and dropped from
  the mean, which flatters runs that miss an organ.
- `roll` augmentation is anatomically impossible.
- Compute: native resolution 2.6× slower; 2.5D + lung 738 min on a laptop.
- Possible solutions: more seeds / 5-fold CV, longer training with early stopping, 3D or more
  2.5D context for the esophagus, translation instead of roll.

## 5. Suggested 10-minute flow (~6 speakers)

1. Recap + the problem in one slide: esophagus is the bottleneck (midterm numbers, per-organ). ~1 min
2. Full data + preprocessing ablation → native resolution, with the "why" (pixels per esophagus,
   ENet's 8× downsampling). ~1.5 min
3. Augmentation: gain only at native resolution, and stabilises seeds. ~1.5 min
4. Class imbalance: loss + sampler; what was dropped (Tversky). ~1.5 min
5. Architecture: 2.5D + lung; transformer as the tried-and-dropped path. ~1.5 min
6. Post-processing: HD95 halves; the seed-43 esophagus story. ~1 min
7. Final model vs baseline (Dice + HD95 + per organ + a qualitative 3D figure), limitations. ~2 min

## 6. Open decisions before the slide deadline

- [ ] Run the combo ladder (≈7–8 h each on a MacBook at native resolution) and pick the final config.
- [ ] Second seed for the final config (and ideally for focal + Dice).
- [ ] Per-sample vs per-batch augmentation.
- [ ] Is 2.5D + lung in the final model? (Slow; not combined with the new loss yet.)
- [ ] Resolve the duplicated transformer rows in RESULTS.md.
- [ ] Confirm the midterm cut-off (section 1).
- [ ] Freeze RESULTS.md numbers = submission numbers.
