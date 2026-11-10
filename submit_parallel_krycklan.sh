#!/bin/bash
set -euo pipefail

RUN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$RUN_ROOT/logs"

export SPAFHY_FOLDER="$RUN_ROOT"
export SPAFHY_KRYCKLAN_ARRAY=1

for size in small medium large; do
    sbatch --chdir="$RUN_ROOT" \
        --export=ALL \
        --array=0-7 \
        --output="$RUN_ROOT/logs/%x-%A_%a.out" \
        --error="$RUN_ROOT/logs/%x-%A_%a.err" \
        "$RUN_ROOT/run_parallel_krycklan_${size}.slurm.sh"
done
