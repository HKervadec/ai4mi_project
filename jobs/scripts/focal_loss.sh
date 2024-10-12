# unweighted focal loss
gammas=(0.5 1 2 5)
for gamma in ${gammas[@]}; do
    sbatch jobs/scripts/focal_loss_gamma_${gamma}.job
done

# weighted focal loss
sbatch jobs/scripts/focal_loss_weighted_icf.job
sbatch jobs/scripts/focal_loss_weighted_heuristic.job