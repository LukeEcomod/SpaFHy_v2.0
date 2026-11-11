#!/bin/bash
#SBATCH --job-name=stitch-krycklan
#SBATCH --nodes=1
#SBATCH --account=project_2000908
#SBATCH --partition=small
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=32:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

RUN_ROOT="${SPAFHY_FOLDER:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
array_task_id="${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"

if [[ -f "$RUN_ROOT/.env" ]]; then
    set -a
    source "$RUN_ROOT/.env"
    set +a
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if command -v module >/dev/null 2>&1; then
    module load python-data
fi
if [[ -f "$RUN_ROOT/.venv/bin/activate" ]]; then
    source "$RUN_ROOT/.venv/bin/activate"
fi

cd "$RUN_ROOT"
python -u "$RUN_ROOT/stitch_krycklan.py" --array-task-id "$array_task_id" "$@"
