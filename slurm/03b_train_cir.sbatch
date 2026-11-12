#!/bin/bash
#SBATCH --job-name=tb-cir
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Phase 4 (CIR variant): CIR unlearning trainer -> checkpoint B (overwrites any prior
# MLAC B + B_compute.json; archive first if you want to keep both). `blocking=cir` is
# baked in so the CIR config is always used. Terminates early at the 0.1% retain-
# disruption budget, so wall-clock is usually well under the ceiling.
#   sbatch slurm/03b_train_cir.sbatch model=esm2_35m                                   # headline (Option A)
#   sbatch slurm/03b_train_cir.sbatch model=esm2_35m blocking.subspace_mode=discriminative  # A'
#   sbatch slurm/03b_train_cir.sbatch model=esm2_35m blocking.subspace_mode=forget_pca       # B (ablation)
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/03b_train_cir.py blocking=cir "$@"
