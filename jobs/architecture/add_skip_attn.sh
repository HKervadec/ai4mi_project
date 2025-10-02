#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Attn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=outfiles/skip_attn_%A.out

set -Eeuo pipefail
trap 'echo "[ERR] Line $LINENO failed. Exiting." >&2' ERR

# ------------------------------
# Environment
# ------------------------------
module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.4.0

cd "$HOME/ai4mi_project" || exit 1
source ai4mi/bin/activate

# Load .env (expects WANDB_ENTITY, WANDB_PROJECT)
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
  echo "[INFO] Loaded .env file"
else
  echo "[WARN] No .env file found, continuing without WANDB config"
fi

# ------------------------------
# Config
# ------------------------------
SEEDS=(42) # 420 37)
EPOCHS=25
RUN_NAME="skip_attn"
RESULTS_DIR="train_results_architecture"


# Constants for stitching
GRP_REGEX="(Patient_\\d\\d)_\\d\\d\\d\\d"
SRC_SCAN_PATTERN_FIXED="data/segthor_fixed/train/{id_}/GT.nii.gz"

# ------------------------------
# Data preprocessing, once only
# ------------------------------
if [[ ! -d data/SEGTHOR_CLEAN ]]; then
  echo "[INFO] Preparing SEGTHOR_CLEAN (first-time setup)…"
  make data/SEGTHOR_CLEAN CFLAGS=-O -n  # dry-run preview

  rm -rf data/segthor_fixed_tmp data/segthor_fixed
  python -O sabotage.py \
    --mode inv \
    --source_dir data/segthor_train \
    --dest_dir data/segthor_fixed_tmp \
    -K 2 \
    --regex_gt "GT.nii.gz" \
    -p 4
  mv data/segthor_fixed_tmp data/segthor_fixed

  rm -rf data/SEGTHOR_CLEAN_tmp data/SEGTHOR_CLEAN
  python -O slice_segthor.py \
    --source_dir data/segthor_fixed \
    --dest_dir data/SEGTHOR_CLEAN_tmp \
    --shape 256 256 \
    --retain 10 \
    -p -1
  mv data/SEGTHOR_CLEAN_tmp data/SEGTHOR_CLEAN
else
  echo "[INFO] Skipping preprocessing (data/SEGTHOR_CLEAN exists)."
fi

# ------------------------------
# One-time GT stitching for eval
# ------------------------------
# Build reference 3D GT once
mkdir -p val/gt
if [[ ! -d val/gt || -z "$(ls -A val/gt 2>/dev/null)" ]]; then
  echo "[INFO] Stitching ground-truth volumes → val/gt"
  python stitch.py \
    --data_folder data/SEGTHOR_CLEAN/val/gt \
    --dest_folder val/gt \
    --num_classes 5 \
    --grp_regex "$GRP_REGEX" \
    --source_scan_pattern "$SRC_SCAN_PATTERN_FIXED"
else
  echo "[INFO] Skipping GT stitching (val/gt already populated)."
fi

# ------------------------------
# Train/Eval loop
# ------------------------------
for SEED in "${SEEDS[@]}"; do
  DEST="${RESULTS_DIR}/${RUN_NAME}_${SEED}"
  PLOT_PDF="${DEST}/plots.pdf"
  PRED_DIR="${DEST}/val/pred" # where stitched predictions will be written

  echo "[INFO] === Seed ${SEED} ==="

  # Training
  python -O main.py \
    --dataset SEGTHOR_CLEAN \
    --mode full \
    --epoch "$EPOCHS" \
    --dest "$DEST" \
    --gpu \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --seed "$SEED" \
    --wandb_name "${RUN_NAME}_${SEED}" \
    --skip_attention  # Enable skip attention

  # Plotting
  python combined_plot.py \
    --results_dir "$DEST" \
    --output "$PLOT_PDF"

  # Stitch predictions (from best epoch val slices → 3D volumes)
  # Assumes per-slice predictions live under $DEST/best_epoch/val
  python stitch.py \
    --data_folder "$DEST/best_epoch/val" \
    --dest_folder "$PRED_DIR" \
    --num_classes 5 \
    --grp_regex "$GRP_REGEX" \
    --source_scan_pattern "$SRC_SCAN_PATTERN_FIXED"

  # Evaluation (3D Dice + HD95) vs stitched GT
  python distorch/compute_metrics.py \
    --ref_folder val/gt \
    --pred_folder "$PRED_DIR" \
    --ref_extension .nii.gz \
    --pred_extension .nii.gz \
    --num_classes 5 \
    --metrics 3d_dice 3d_hd95 3d_assd 3d_jaccard \
    --save_folder "${DEST}/metrics"

  echo "[INFO] Seed ${SEED} done. Results: $DEST | Plots: $PLOT_PDF | Metrics: $DEST/metrics"
done

echo "[INFO] All seeds completed."
