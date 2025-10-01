#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:50:00
#SBATCH --output=outfiles/test_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.4.0
#

source ai4mi/bin/activate
#python main.py --dataset SEGTHOR_CLEAN --mode full --epoch 1 --dest train_results --gpu

python -O main.py --dataset SEGTHOR_CLEAN --mode full --epoch 25 --dest train_results/train_results_baseline_42 --gpu --wandb_entity michal.mazuryk@student.uva.nl --wandb_project ai4med --seed 42 --wandb_name "baseline_seed_42"
python -O main.py --dataset SEGTHOR_CLEAN --mode full --epoch 25 --dest train_results/train_results_baseline_420 --gpu --wandb_entity michal.mazuryk@student.uva.nl --wandb_project ai4med --seed 420 --wandb_name "baseline_seed_420"
python -O main.py --dataset SEGTHOR_CLEAN --mode full --epoch 25 --dest train_results/train_results_baseline_37 --gpu --wandb_entity michal.mazuryk@student.uva.nl --wandb_project ai4med --seed 37 --wandb_name "baseline_seed_37"
