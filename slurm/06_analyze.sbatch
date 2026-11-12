#!/bin/bash
#SBATCH --job-name=tb-analyze
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Phase 7: analysis (needs a GPU only for the benign-utility PPL of B and A').
#   sbatch slurm/06_analyze.sbatch model=esm2_35m
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/06_analyze.py "$@"
