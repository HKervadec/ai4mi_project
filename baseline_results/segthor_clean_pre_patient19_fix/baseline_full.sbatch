#!/bin/bash
#SBATCH --job-name=segthor_baseline
#SBATCH --account=gpuuva084
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=50G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

PROJECT_DIR="$HOME/projects/ai4mi_project"
RUN_DIR="/scratch-shared/$USER/ai4mi_project/results/segthor_clean/baseline_full"
FINAL_DIR="$PROJECT_DIR/results/segthor_clean/baseline_full"

cd "$PROJECT_DIR"
source .venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

test -d data/SEGTHOR_CLEAN/train/img
test -d data/SEGTHOR_CLEAN/train/gt
test -d data/SEGTHOR_CLEAN/val/img
test -d data/SEGTHOR_CLEAN/val/gt

if [[ -e "$RUN_DIR" || -e "$FINAL_DIR" ]]; then
    echo "A baseline output directory already exists."
    echo "Scratch: $RUN_DIR"
    echo "Final:   $FINAL_DIR"
    echo "Refusing to overwrite existing results."
    exit 1
fi

nvidia-smi

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU"

print("GPU:", torch.cuda.get_device_name(0))
PY

mkdir -p "$(dirname "$RUN_DIR")"

python main.py \
    --dataset SEGTHOR_CLEAN \
    --mode full \
    --epochs 25 \
    --dest "$RUN_DIR" \
    --gpu

mkdir -p "$FINAL_DIR"
rsync -a "$RUN_DIR/" "$FINAL_DIR/"

echo "Finished: $(date)"
echo "Final results: $FINAL_DIR"
echo "Baseline training completed successfully."
