#!/bin/bash
#SBATCH --job-name=tb-ladder-Cverify
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/3_scale_ladder/outputs/%x-%j.out
#SBATCH --mail-type=FAIL
# Phase C acceptance check: static forget fired, benign utility retained vs A', FLOP match
# held. Exits 1 if any scale fails, so the afterok chain stops before Phase D/E spend
# compute on checkpoints that would invalidate their own results.
#   sbatch experiments/3_scale_ladder/phaseC_verify.sbatch --scales 150M 650M
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/3_scale_ladder/phaseC_verify.py --device cuda "$@"
