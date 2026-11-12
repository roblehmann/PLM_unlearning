#!/bin/bash
#SBATCH --job-name=tb-cir-overlap
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# Phase 8e (CIR): O(F,R) activation-subspace overlap — principal angles between F's and R's
# top-k activation PCAs on θ₀. The upstream measurement behind the CIR result: high overlap =>
# nothing family-distinctive to remove that a utility-preserving safeguard may touch.
#   sbatch slurm/08e_cir_overlap.sbatch model=esm2_35m blocking=cir
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/08e_cir_overlap.py "$@"
