#!/bin/bash
#SBATCH --job-name=tb-pfam-umap
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=experiments/7_pfam_reps_vs_retain/outputs/%x-%j.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env_setup.sh"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMBA_NUM_THREADS=16
python -u experiments/7_pfam_reps_vs_retain/analyze.py umap "$@"
