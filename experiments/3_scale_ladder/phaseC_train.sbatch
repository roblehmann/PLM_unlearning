#!/bin/bash
#SBATCH --job-name=tb-ladder-train
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=experiments/3_scale_ladder/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Phase C: runs ONE existing pipeline script (03_train_blocked.py or 04_train_control.py)
# with the scale + run-root overrides appended by phaseC_train.sh. Generic on purpose —
# the tent-pole cost at 650M is B (3000 outer steps x (K=4 inner + outer)), with A'
# FLOP-matched to it by construction, so both fit the same 24h/120G envelope.
#   (submit via experiments/3_scale_ladder/phaseC_train.sh 650M — not directly)
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python "$@"
