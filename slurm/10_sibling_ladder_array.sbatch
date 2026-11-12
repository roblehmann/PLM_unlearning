#!/bin/bash
#SBATCH --job-name=tb-ladder
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Phase 10: sibling-ladder relearning sweep as a SLURM array. Each task runs one shard;
# the script reads SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT and writes its own CSV.
# Default grid = 2 models × 3 recipe-rank × 5 budgets × 3 seeds × 6 rungs = 540 cells;
# 16 shards ≈ 34 each (same per-shard cost as the Phase-6 sweep -> 12h budget carries over).
#   sbatch slurm/10_sibling_ladder_array.sbatch model=esm2_35m
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/10_sibling_ladder_sweep.py sweep=sibling_ladder "$@"
