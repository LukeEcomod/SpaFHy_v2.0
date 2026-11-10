#!/bin/bash
#SBATCH --job-name=spafhy-krycklan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --account=project_2000908
#SBATCH --partition=small
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

RUN_ROOT="${SPAFHY_FOLDER:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
mkdir -p "$RUN_ROOT/logs"

array_mode="${SPAFHY_KRYCKLAN_ARRAY:-0}"
array_task_id="${SLURM_ARRAY_TASK_ID:-}"
selected_catchments="${CATCHMENT_NOS:-}"

if [[ -f "$RUN_ROOT/.env" ]]; then
    set -a
    source "$RUN_ROOT/.env"
    set +a
fi

if [[ "$array_mode" == "1" ]]; then
    case "$array_task_id" in
        0|1|2|3|4|5|6|7) ;;
        *)
            echo "Invalid SLURM_ARRAY_TASK_ID for Krycklan configurations: $array_task_id" >&2
            exit 1
            ;;
    esac
    resolutions=(20 40 80 160)
    stream_types=(channels 5haStreams)
    export SPAFHY_RESOLUTION="${resolutions[$((array_task_id % 4))]}"
    export SPAFHY_STREAMS="${stream_types[$((array_task_id / 4))]}"
    export RUN_MODE="subset"
    export CATCHMENT_NOS="${selected_catchments:?Catchment subset must be set by the batch script}"
    echo "Krycklan configuration: resolution=${SPAFHY_RESOLUTION}m, streams=${SPAFHY_STREAMS}, array task=${array_task_id}"
fi

export SPAFHY_FOLDER="$RUN_ROOT"
export PROJECT_FOLDER="${PROJECT_FOLDER:?Set PROJECT_FOLDER before submitting this job}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if command -v module >/dev/null 2>&1; then
    module load python-data
fi

RUN_MODE="${RUN_MODE:-all}"
CASE_ARGS=()
case "$RUN_MODE" in
    single)
        CASE_ARGS=(--catchment-no "${CATCHMENT_NO:?Set CATCHMENT_NO for RUN_MODE=single}")
        ;;
    subset)
        CASE_ARGS=(--catchment-nos ${CATCHMENT_NOS:?Set CATCHMENT_NOS for RUN_MODE=subset})
        ;;
    all)
        CASE_ARGS=(--all-catchments)
        ;;
    *)
        echo "Unsupported RUN_MODE: $RUN_MODE" >&2
        exit 1
        ;;
esac

if [[ -f "$SPAFHY_FOLDER/.venv/bin/activate" ]]; then
    # Use the repository virtual environment when present.
    source "$SPAFHY_FOLDER/.venv/bin/activate"
fi

python "$RUN_ROOT/run_parallel_krycklan.py" \
    "${CASE_ARGS[@]}" \
    --n-workers "${SLURM_CPUS_PER_TASK:-1}"