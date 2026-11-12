#!/bin/bash
#SBATCH --job-name=tb-fvg-t5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --output=experiments/4_footprint_vs_generator/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# Test 5 — re-score the EXISTING Pfam CIR-B (the published 6.41x block) and controls on the
# decontaminated F_test from t0. No training. Tests whether the 6.41x was a leakage artifact.
# Needs t0's results/pfam/F_test_decon.fasta.
#   sbatch experiments/4_footprint_vs_generator/t5_cir_rescore.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/4_footprint_vs_generator/t5_cir_rescore.py --device cuda "$@"
