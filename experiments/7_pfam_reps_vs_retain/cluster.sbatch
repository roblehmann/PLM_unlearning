#!/bin/bash
#SBATCH --job-name=tb-pfam-cluster
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/7_pfam_reps_vs_retain/outputs/%x-%j.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env_setup.sh"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
python -u experiments/7_pfam_reps_vs_retain/cluster_families.py "$@"
