#!/bin/bash
#SBATCH --job-name=segthor_2d_prep
#SBATCH --output=logs/2d_prep_%j.out
#SBATCH --error=logs/2d_prep_%j.err
#SBATCH --account=gpuuva084
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# Create logging directory if it doesn't exist and fail script on any non-zero exit code
mkdir -p logs
set -e

# Sanity adjustment for Python module version to match your environment setup
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual environment
source ~/envs/medseg/bin/activate

# Toggle debug mode: now full training; set to "--debug" for a fast dry run
DEBUG_FLAG=""

# Print diagnostic info to job output
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node running job: $SLURM_JOB_NODELIST"
echo "Running 2D Preprocessed Training..."
echo "=================================================="

# Run Training: 2D + 192x192 CenterCrop + Empty slice filtering + Spatial augmentation
# NB: Upon inspection of dry-run, the crop can be less agressive to allow for further rotation
python main.py \
  --gpu \
  --dataset SEGTHOR \
  --mode full \
  --crop_size 192 \
  --filter_empty \
  --augment \
  $DEBUG_FLAG \
  --epochs 20 \
  --dest ./results/2d_preprocessed

echo "Job finished successfully at: $(date)"