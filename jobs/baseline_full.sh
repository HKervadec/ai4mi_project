#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=outfiles/full_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.4.0
#

cd "$HOME/ai4mi_project" || exit 1
source ai4mi/bin/activate

# make data/SEGTHOR_CLEAN CFLAGS=-O -n  # Will display the commands that will run, easy to inspect:
# rm -rf data/segthor_fixed_tmp data/segthor_fixed
# python -O sabotage.py --mode inv --source_dir data/segthor_train --dest_dir data/segthor_fixed_tmp -K 2 --regex_gt "GT.nii.gz" -p 4
# mv data/segthor_fixed_tmp data/segthor_fixed
# rm -rf data/SEGTHOR_CLEAN_tmp data/SEGTHOR_CLEAN
# python -O slice_segthor.py --source_dir data/segthor_fixed --dest_dir data/SEGTHOR_CLEAN_tmp \
#         --shape 256 256 --retain 10 -p -1
# mv data/SEGTHOR_CLEAN_tmp data/SEGTHOR_CLEAN


python -O main.py --dataset SEGTHOR_CLEAN --mode full --epoch 25 --dest train_results_baseline_42 --gpu --wandb_entity azywot --wandb_project ai4med --seed 42 --wandb_name "baseline_seed_42"
# python -O main.py --dataset SEGTHOR_CLEAN --mode full --epoch 25 --dest train_results_baseline_420 --gpu --wandb_entity michal.mazuryk@student.uva.nl --wandb_project ai4med --seed 420 --wandb_name "baseline_seed_420"
# python -O main.py --dataset SEGTHOR_CLEAN --mode full --epoch 25 --dest train_results_baseline_37 --gpu --wandb_entity michal.mazuryk@student.uva.nl --wandb_project ai4med --seed 37 --wandb_name "baseline_seed_37"

python plot_full.py --result_folder train_results_baseline_42 --output_pdf plot_full.pdf


python stitch_new.py \
  --data_folder train_results_baseline_42/best_epoch/val \
  --dest_folder val/pred \
  --num_classes 5 \
  --grp_regex "(Patient_\d\d)_\d\d\d\d" \
  --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"

mkdir -p val/gt

python stitch_new.py --data_folder data/SEGTHOR_CLEAN/val/gt --dest_folder val/gt --num_classes 5 --grp_regex "(Patient_\d\d)_\d\d\d\d" --source_scan_pattern "data/segthor_fixed/train/{id_}/GT.nii.gz"

python distorch/compute_metrics.py --ref_folder val/gt --pred_folder val/pred --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 --save_folder val