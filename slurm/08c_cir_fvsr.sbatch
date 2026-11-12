#!/bin/bash
#SBATCH --job-name=tb-cir-fvsr
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# Phase 8c (CIR): F-distinctive-vs-R-shared subspace-projected displacement — the airtight
# CIR falsification. Builds an INDEPENDENT activation-PCA basis (R vs F) on θ₀ (one GPU
# pass), then decomposes ‖θ_B − θ₀‖ for every operating point into F_distinctive / R_shared
# / rest. Read F/total against the durability multiplier.
#   sbatch slurm/08c_cir_fvsr.sbatch model=esm2_35m blocking=cir \
#     +ops=cir35m_lowdis,cir35m_peak,cir_forgetpca/main
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/08c_cir_fvsr.py "$@"
