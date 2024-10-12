gammas=(0.5 1 2 5)
for gamma in ${gammas[@]}; do
    python -O main.py \
        --dataset SEGTHORCORRECT \
        --mode full \
        --epochs 100 \
        --dest results/segthor/focal/baseline/gamma_${gamma} \
        --gpu \
        --loss focal \
        --focal_loss_gamma ${gamma} \
        --plot_results
done

weights_icf=(1.0 22.3814 1.3688 29.9430 5.2261) # weights through inverse class frequency
weights_heuristic=(1.0 5.0 1.0 1.0 1.0)         # weights through heuristic

weight_sets=("weights_icf" "weights_heuristic")

for weight_set in "${weight_sets[@]}"; do
    python -O main.py \
        --dataset SEGTHORCORRECT \
        --mode full \
        --epochs 100 \
        --dest results/segthor/focal/weighted/${weight_set} \
        --gpu \
        --loss focal \
        --focal_loss_gamma 2 \
        --plot_results \
        --weights ${!weight_set}
done