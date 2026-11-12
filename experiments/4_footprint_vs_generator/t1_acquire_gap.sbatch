#!/bin/bash
#SBATCH --job-name=tb-fvg-t1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/4_footprint_vs_generator/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Test 1 — theta_acq (8192 F_train seqs x 40 epochs full fine-tune, NO F_test early
# stopping) per arm, then the train/test PPPL gap. Also the ceiling check that tells us
# whether the OOD arm has any generalising capability to remove at all.
# theta_acq is saved into each arm's own run root (<run>/checkpoints/acquired_F) and is
# the input to t2/t3/t4, so this must complete first.
#   sbatch experiments/4_footprint_vs_generator/t1_acquire_gap.sbatch
#   sbatch experiments/4_footprint_vs_generator/t1_acquire_gap.sbatch --arms mgnify
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/4_footprint_vs_generator/t1_acquire_gap.py --device cuda "$@"
