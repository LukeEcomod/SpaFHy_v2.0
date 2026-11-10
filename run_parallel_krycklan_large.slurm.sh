#!/bin/bash
#SBATCH --job-name=spafhy-krycklan-large
#SBATCH --nodes=1
#SBATCH --account=project_2000908
#SBATCH --partition=small
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=32:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

RUN_ROOT="${SPAFHY_FOLDER:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"

export RUN_MODE="subset"
export CATCHMENT_NOS="141 142 143 151 152 153 161 162 163 164 22 10"

bash "$RUN_ROOT/run_parallel_krycklan.slurm.sh"