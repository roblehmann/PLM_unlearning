#!/bin/bash
#SBATCH --job-name=tb-anchor-t5m3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/7_pfam_reps_vs_retain/outputs/%x-%j.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env_setup.sh"
export HF_HUB_OFFLINE=1
python -u experiments/7_pfam_reps_vs_retain/t5_members.py embed "$@"
