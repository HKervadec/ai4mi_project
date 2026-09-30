# Decisions

Current choices live in `configs/current.yaml`. Nothing here is final: to change or undo a
choice, edit the line in `current.yaml` and add a new entry below.

## Assignment options (at least 5)
| Option | Owner | Status | Chosen? |
|---|---|---|---|
| Additional pre-processing | | not started | |
| Data augmentation (online or offline) | | not started | |
| 2.5D network | | implemented (`configs/experiments/enet_25d.yaml`), not run | |
| Different optimizer | | not started | |
| Non-CNN architecture (Transformer / ViT) | | not started | |
| Different network architecture (modular) | | not started | |
| Post-processing | Elena | implemented (`largest_cc`, options `min_fraction`, `skip`); experiments `post_lcc_min_fraction`, `post_lcc_skip_esophagus` via `postprocess_run.py` | |
| Different loss function | | not started | |
| Other regularizer at the loss level | Britt | implemented (`boundary`: Sobel edge L1, option `per_class`); experiments `boundary{,_w1,_w5}` (merged foreground) and `boundary_per_organ{,_w1,_w5}` (per organ), weights 0.1 / 1 / 5, not run | |
| Pre-training / hybrid supervision (public dataset) | | not started | |

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
