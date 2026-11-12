#!/bin/bash
#SBATCH --job-name=tb-mgreps-t3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/6_mgnify_reps_vs_retain/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# T3 — PCA-50 + one global UMAP over all 229,523 points, plus the kNN-mixing statistic.
# CPU only. Sized from a measured precedent rather than a guess: the same recipe fitted
# 258,298 points (more than this) in 3 min 20 s at 3.9 GB MaxRSS, so 64G/2h is ~16x memory
# and ~20x time headroom. Depends on T1 only, so it overlaps T2.
#   sbatch experiments/6_mgnify_reps_vs_retain/t3_umap.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
python experiments/6_mgnify_reps_vs_retain/t3_umap.py "$@"
