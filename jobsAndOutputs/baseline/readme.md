# Baseline from original repo

1. Create environment and setup data: `jobs/env.job`.
2.1 Training TOY2 dataset: `jobs/train_TOY2.job`.
2.2  Training SEGTHOR dataset: `jobs/train_segthor.job`.
3. Stitching back to 3D: `jobs/stitching.job`.

## Viusalize 2D results from the terminal 

TOY2:
```bash
module load 2024 Anaconda3/2024.06-1
source "$EBROOTANACONDA3/etc/profile.d/conda.sh" && conda activate ai4mi
python viewer/viewer.py --img_source data/TOY2/val/img \
    data/TOY2/val/gt results/toy2/ce/iter000/val results/toy2/ce/iter005/val results/toy2/ce/best_epoch/val \
    --show_img -C 256 --no_contour
```

SEGTHOR:
```bash
module load 2024 Anaconda3/2024.06-1
source "$EBROOTANACONDA3/etc/profile.d/conda.sh" && conda activate ai4mi
python viewer/viewer.py --img_source data/SEGTHOR/val/img \
    data/SEGTHOR/val/gt results/segthor/ce/iter000/val results/segthor/ce/best_epoch/val \
    --show_img -C 256 --no_contour
```