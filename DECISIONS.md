# Decisions

Current choices live in `configs/current.yaml`. Nothing here is final: to change or undo a
choice, edit the line in `current.yaml` and add a new entry below.

## Assignment options (at least 5)
| Option | Owner | Status | Chosen? |
|---|---|---|---|
| Additional pre-processing | | not started | |
| Data augmentation (online or offline) | | not started | |
| 2.5D network | | not started | |
| Different optimizer | | not started | |
| Non-CNN architecture (Transformer / ViT) | | not started | |
| Different network architecture (modular) | | not started | |
| Post-processing | | not started | |
| Different loss function | | not started | |
| Other regularizer at the loss level | | not started | |
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
