#!/bin/bash
#SBATCH --job-name=tb-smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Phase 0 smoke test on the 8M model. Submit from the repo root:
#   sbatch slurm/00_smoke.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/00_smoke.py model=esm2_8m tracking.backend=none "$@"
