#!/bin/bash
#SBATCH --job-name=tb-anchor-t5m4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/7_pfam_reps_vs_retain/outputs/%x-%j.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env_setup.sh"
python -u experiments/7_pfam_reps_vs_retain/t5_members.py report "$@"
