#!/bin/bash

# ------------------------------
# User-tweakable
# ------------------------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/ai4mi_project}"
JOB_SCRIPT="${JOB_SCRIPT:-${PROJECT_DIR}/jobs/architecture/000_run_seed.job}"

# Slurm resources (can be overridden via env or CLI if you wish)
PARTITION="${PARTITION:-gpu_a100}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-9}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
OUTPUT_DIR_REL="outfiles"

# Concurrency control
MAX_JOBS="${MAX_JOBS:-9}"      # max concurrent jobs for this user
SLEEP_TIME="${SLEEP_TIME:-60}" # seconds between queue checks
RUN_PREP_FIRST="${RUN_PREP_FIRST:-false}" # Optional: run a single prep job first to avoid race conditions

# Seeds to launch (each becomes its own Slurm job)
SEEDS=(42 420 37)
# Training config to export
EPOCHS="${EPOCHS:-25}"
RUN_NAME="${RUN_NAME:-ExampleName}"
RESULTS_DIR="${RESULTS_DIR:-train_results_arch}"
EXTRA_PARAMS="${EXTRA_PARAMS:-}"
# EXTRA_PARAMS="${EXTRA_PARAMS:---model_class ExampleModel}"

mkdir -p "${PROJECT_DIR}/${OUTPUT_DIR_REL}"
USER_NAME=$(whoami)

wait_for_available_slot() {
  while true; do
    CURRENT_JOBS=$(squeue -u "$USER_NAME" -h | wc -l)
    if (( CURRENT_JOBS < MAX_JOBS )); then
      break
    fi
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$TIMESTAMP] ⏳ Too many jobs queued ($CURRENT_JOBS). Waiting for available slot..."
    sleep "$SLEEP_TIME"
  done
}

# ------------------------------
# Optional prep job
# ------------------------------
prep_job_id=""
if [[ "$RUN_PREP_FIRST" == "true" ]]; then
  wait_for_available_slot
  echo "📦 Submitting PREP job (data prep + GT stitching)…"
  PREP_OUT="${PROJECT_DIR}/${OUTPUT_DIR_REL}/prep_%A.out"
  prep_submit=$(sbatch \
    --job-name="data_prep" \
    --output="$PREP_OUT" \
    --partition="$PARTITION" \
    --gpus="$GPUS" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --time="$TIME_LIMIT" \
    --hint=nomultithread \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",MODE="prep" \
    "$JOB_SCRIPT")
  # Extract job ID (sbatch prints: "Submitted batch job <id>")
  prep_job_id=$(echo "$prep_submit" | awk '{print $4}')
  echo "🪪 PREP job id: $prep_job_id"
fi

# ------------------------------
# One job per seed
# ------------------------------
for seed in "${SEEDS[@]}"; do
  wait_for_available_slot

  TAG="${RUN_NAME}_s${seed}"
  OUT_PATH="${PROJECT_DIR}/${OUTPUT_DIR_REL}/${TAG}_%A.out"

  echo "📤 Submitting TRAIN job for seed=$seed"

  sbatch \
    --job-name="$TAG" \
    --output="$OUT_PATH" \
    --partition="$PARTITION" \
    --gpus="$GPUS" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --time="$TIME_LIMIT" \
    --hint=nomultithread \
    ${prep_job_id:+--dependency=afterok:${prep_job_id}} \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",MODE="train",SEED="$seed",EPOCHS="$EPOCHS",RUN_NAME="$RUN_NAME",RESULTS_DIR="$RESULTS_DIR",EXTRA_PARAMS="$EXTRA_PARAMS" \
    "$JOB_SCRIPT"
done

echo "✅ Launch complete."
