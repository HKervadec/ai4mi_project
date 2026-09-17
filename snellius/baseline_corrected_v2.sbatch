#!/usr/bin/env bash
#SBATCH --job-name=segthor_v2
#SBATCH --account=gpuuva084
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --output=logs/segthor_v2-%j.out
#SBATCH --error=logs/segthor_v2-%j.err

set -euo pipefail

PROJECT="$HOME/projects/ai4mi_project"
SCRATCH_PROJECT="/scratch-shared/$USER/ai4mi_project"
SOURCE="$HOME/data/segthor_part1"
CLEAN_V2="$SCRATCH_PROJECT/data/segthor_part1_clean_cc_v2"
SLICED_V2="$SCRATCH_PROJECT/data/SEGTHOR_CLEAN_v2"
RESULT_V2="$SCRATCH_PROJECT/results/segthor_clean/baseline_patient19_fixed"

cd "$PROJECT"

module purge
module load 2025
module load Anaconda3/2025.06-1
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Started: $(date)"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ ! -d "$SOURCE/train" ]]; then
    echo "Raw SegTHOR source is missing: $SOURCE/train" >&2
    exit 1
fi
if [[ -e "$CLEAN_V2" || -e "$SLICED_V2" || -e "$RESULT_V2" ]]; then
    echo "A v2 output already exists; refusing to mix or overwrite runs." >&2
    echo "CLEAN_V2=$CLEAN_V2" >&2
    echo "SLICED_V2=$SLICED_V2" >&2
    echo "RESULT_V2=$RESULT_V2" >&2
    exit 1
fi
if [[ -e data/SEGTHOR_CLEAN && ! -L data/SEGTHOR_CLEAN ]]; then
    echo "data/SEGTHOR_CLEAN exists but is not a symlink; refusing to replace it." >&2
    exit 1
fi

mkdir -p "$SCRATCH_PROJECT/data" "$SCRATCH_PROJECT/results/segthor_clean"

python split_merged_label_cc.py \
    --source-root "$SOURCE" \
    --dest-root "$CLEAN_V2"

python slice_segthor.py \
    --source_dir "$CLEAN_V2" \
    --dest_dir "$SLICED_V2" \
    --shape 256 256 \
    --retains 5 \
    --seed 0 \
    --fold 0 \
    --process 8

train_count=$(find "$SLICED_V2/train/img" -type f -name '*.png' | wc -l)
val_count=$(find "$SLICED_V2/val/img" -type f -name '*.png' | wc -l)
if [[ "$train_count" -ne 2817 || "$val_count" -ne 915 ]]; then
    echo "Unexpected slice counts: train=$train_count val=$val_count" >&2
    exit 1
fi

# main.py expects the fixed dataset name data/SEGTHOR_CLEAN.  Repoint it only
# after all correction, slicing, and count checks have completed successfully.
ln -sfn "$SLICED_V2" data/SEGTHOR_CLEAN
echo "SEGTHOR_CLEAN -> $(readlink -f data/SEGTHOR_CLEAN)"

python -u main.py \
    --epochs 25 \
    --dataset SEGTHOR_CLEAN \
    --mode full \
    --dest "$RESULT_V2" \
    --gpu

echo "Finished: $(date)"
