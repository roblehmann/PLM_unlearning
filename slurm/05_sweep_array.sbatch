#!/bin/bash
#SBATCH --job-name=tb-sweep
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Phase 6: full attacker sweep as a SLURM array. Each task runs one shard; the script
# reads SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT and writes its own CSV.
# Adjust --array to your cell count (default grid = 540 cells; 16 shards ≈ 34 each).
#   sbatch slurm/05_sweep_array.sbatch model=esm2_35m
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/05_full_sweep.py "$@"
