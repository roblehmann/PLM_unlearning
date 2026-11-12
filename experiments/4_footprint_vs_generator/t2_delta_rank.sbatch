#!/bin/bash
#SBATCH --job-name=tb-fvg-t2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/4_footprint_vs_generator/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Test 2 — Delta_acquire = theta_acq - theta_0: singular spectra / effective rank, and the
# causal rank-truncation ablation (remove-top-k and keep-top-k) scored on F_train_seen vs
# decontaminated F_test. Needs t1's theta_acq.
#   sbatch experiments/4_footprint_vs_generator/t2_delta_rank.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/4_footprint_vs_generator/t2_delta_rank.py --device cuda "$@"
