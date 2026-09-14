# Sourced by every pipeline job: modules, conda env, repo root. Not a job itself.
module purge
module load 2024
module load Anaconda3/2024.06-1

# conda activate needs the shell hook in a non-interactive shell; conda.sh trips over set -u.
set +u
source "${EBROOTANACONDA3:-$CONDA_PREFIX}/etc/profile.d/conda.sh"
conda activate ai4mi
set -u

cd "$SLURM_SUBMIT_DIR"   # jobs are submitted from the repo root (relative --output paths need it)
[ -f src/train.py ] || { echo "submit from the repo root (sbatch jobsAndOutputs/pipeline/jobs/...)"; exit 1; }
echo "[$(date -Is)] job $SLURM_JOB_ID on $(hostname), python $(which python), commit $(git rev-parse --short HEAD)"
