#!/bin/bash
#SBATCH --job-name=tb-cir-wspace
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Phase 8b (CIR): subspace-projected weight displacement (the CIR mechanism gate).
# Pure weight-space math on CPU (no GPU) — splits ΔW into kept vs collapsed subspaces
# using results/cir_subspace.npz and contrasts B with A'. Run scripts/03b_train_cir.py
# (and ideally scripts/08_weightspace.py for the LMC barrier) first.
#   sbatch slurm/08b_cir_subspace.sbatch model=esm2_35m
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/08b_cir_subspace.py "$@"
