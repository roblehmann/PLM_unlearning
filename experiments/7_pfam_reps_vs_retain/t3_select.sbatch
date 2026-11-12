#!/bin/bash
#SBATCH --job-name=tb-anchor-t3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/7_pfam_reps_vs_retain/outputs/%x-%j.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env_setup.sh"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
python -u experiments/7_pfam_reps_vs_retain/t3_select.py "$@"
