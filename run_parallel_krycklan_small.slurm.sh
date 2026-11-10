#!/bin/bash
#SBATCH --job-name=spafhy-krycklan-small
#SBATCH --nodes=1
#SBATCH --account=project_2000908
#SBATCH --partition=small
#SBATCH --cpus-per-task=7
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

RUN_ROOT="${SPAFHY_FOLDER:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"

export RUN_MODE="subset"
export CATCHMENT_NOS="1 2 3 4 5 6 7"

bash "$RUN_ROOT/run_parallel_krycklan.slurm.sh"