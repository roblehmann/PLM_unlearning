#!/bin/bash
#SBATCH --job-name=tb-wspace
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Phase 8: weight-space mechanism (per-layer ‖θ−θ₀‖ + LMC barrier θ₀→θ_B).
# Reads checkpoints/{B,A_prime}; ~25–45 min (L2 instant, LMC = #α × {F,benign} evals).
#   sbatch slurm/08_weightspace.sbatch model=esm2_35m
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/08_weightspace.py "$@"
