#!/bin/bash
#SBATCH --job-name=spafhy-krycklan-medium
#SBATCH --nodes=1
#SBATCH --account=project_2000908
#SBATCH --partition=small
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

RUN_ROOT="${SPAFHY_FOLDER:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"

export RUN_MODE="subset"
export CATCHMENT_NOS="8 9 12 13 20 21"
export HDF5_USE_FILE_LOCKING=FALSE

bash "$RUN_ROOT/run_parallel_krycklan.slurm.sh"