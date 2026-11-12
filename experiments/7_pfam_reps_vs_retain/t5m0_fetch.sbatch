#!/bin/bash
#SBATCH --job-name=tb-anchor-t5m0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/7_pfam_reps_vs_retain/outputs/%x-%j.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env_setup.sh"
python -u experiments/7_pfam_reps_vs_retain/t5_members.py fetch "$@"
