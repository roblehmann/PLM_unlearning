#!/bin/bash
#SBATCH --job-name=tb-gate
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Phase 3: baseline attack + headroom GATE on A. Confirm PASS before training B.
#   sbatch slurm/02_baseline_attack.sbatch model=esm2_35m
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/02_baseline_attack.py "$@"
