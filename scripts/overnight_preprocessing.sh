#!/bin/bash
# Run the preprocessing experiments one after the other (e.g. overnight on a laptop).
#
# Usage: bash scripts/overnight_preprocessing.sh [experiment[:seed] ...]
#   e.g.  bash scripts/overnight_preprocessing.sh spacing_1.95_crop192 spacing_native_384:43
#   seed defaults to 42; `current` runs configs/current.yaml.
#   default: the list below, most informative first. On a MacBook (mps) a 25-epoch run takes
#   ~2.5 h at 192x192, ~3.7 h at 256x256 and ~9.5 h at 384x384.
#
# - A run that already has a summary.json is skipped; an unfinished run folder is replaced.
# - A failing run is logged and the next one still starts.
# - Keeps the Mac awake (caffeinate) and rebuilds RESULTS.md at the end.
# Logs: logs/overnight_<timestamp>/<experiment>_s<seed>.log

set -uo pipefail
cd "$(dirname "$0")/.."

# Use the project venv even when the calling shell hasn't activated it.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ai4mi/bin/activate ]; then
    source ai4mi/bin/activate
fi
if ! command -v python >/dev/null; then
    echo "!! no python found: activate the ai4mi venv first (source ai4mi/bin/activate)"
    exit 1
fi

EXPERIMENTS=("$@")
if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    # Priority order (see DECISIONS.md, "Preprocessing findings").
    EXPERIMENTS=(
        spacing_1.95_crop192     # tight crop at 1.95 mm: splits native_384's gain into crop vs resolution
        spacing_native_384:43    # 2nd seed of the best run, before it becomes current
        lung_window_native_384   # lung window again, at native resolution
    )
fi

LOG_DIR="logs/overnight_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"

# Keep the machine awake for as long as this script runs (macOS only).
if command -v caffeinate >/dev/null; then
    caffeinate -i -w $$ &
fi

for item in "${EXPERIMENTS[@]}"; do
    exp="${item%%:*}"
    seed=42
    [[ "$item" == *:* ]] && seed="${item#*:}"
    run="holdout40-f0-s${seed}"  # current.yaml: split holdout40, fold 0
    name="${exp} (seed ${seed})"

    if [ "$exp" = "current" ]; then
        config="configs/current.yaml"
    else
        config="configs/experiments/${exp}.yaml"
    fi
    if [ ! -f "$config" ]; then
        echo "!! $name: $config not found, skipping" | tee -a "$LOG_DIR/overview.txt"
        continue
    fi
    if [ -f "results/${exp}/${run}/summary.json" ]; then
        echo "-- $name: already finished, skipping" | tee -a "$LOG_DIR/overview.txt"
        continue
    fi

    start=$(date +%s)
    echo ">> $name: started $(date '+%H:%M')" | tee -a "$LOG_DIR/overview.txt"
    # --overwrite only replaces an unfinished run (finished ones were skipped above).
    if python run.py --config "$config" --set "train.seed=${seed}" --overwrite > "$LOG_DIR/${exp}_s${seed}.log" 2>&1; then
        status="done"
    else
        status="FAILED (see $LOG_DIR/${exp}_s${seed}.log)"
    fi
    echo "<< $name: $status after $(( ($(date +%s) - start) / 60 )) min" | tee -a "$LOG_DIR/overview.txt"
done

python compare.py | tee -a "$LOG_DIR/overview.txt"
echo "All done $(date '+%H:%M'). Overview: $LOG_DIR/overview.txt"
