#!/bin/bash
#SBATCH --job-name=tb-anchor-t1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=experiments/7_pfam_reps_vs_retain/outputs/%x-%j.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env_setup.sh"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
python -u experiments/7_pfam_reps_vs_retain/t1_similarity.py "$@"
