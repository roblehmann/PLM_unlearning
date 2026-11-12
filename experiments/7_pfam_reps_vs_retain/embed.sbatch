#!/bin/bash
#SBATCH --job-name=tb-pfam-embed
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/7_pfam_reps_vs_retain/outputs/%x-%j.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env_setup.sh"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
python -u experiments/7_pfam_reps_vs_retain/analyze.py embed "$@"
