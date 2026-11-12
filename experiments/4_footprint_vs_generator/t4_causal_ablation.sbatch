#!/bin/bash
#SBATCH --job-name=tb-fvg-t4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/4_footprint_vs_generator/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Test 4 — the decisive, CIR-independent check: project each candidate subspace out of the
# FFN down-projection input by hand and score PPPL. Positive control = the F_test generator
# (must break F_test); negative control = a random 24-frame. Needs t3's bases.npz.
#   sbatch experiments/4_footprint_vs_generator/t4_causal_ablation.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/4_footprint_vs_generator/t4_causal_ablation.py --device cuda "$@"
