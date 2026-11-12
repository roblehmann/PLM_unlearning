#!/bin/bash
#SBATCH --job-name=tb-anchor-t4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/7_pfam_reps_vs_retain/outputs/%x-%j.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env_setup.sh"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
python -u experiments/7_pfam_reps_vs_retain/t4_figures.py "$@"
