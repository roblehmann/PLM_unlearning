#!/bin/bash
#SBATCH --job-name=tb-mgreps-t4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/6_mgnify_reps_vs_retain/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# T4 — merges every phase, runs the statistics (the bootstraps dominate the ~5 min), draws
# the figures and generates RESULTS.md. Exits non-zero on any cross-phase inconsistency, so
# a broken merge cannot quietly produce a plausible figure.
#   sbatch experiments/6_mgnify_reps_vs_retain/t4_report.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/6_mgnify_reps_vs_retain/t4_report.py "$@"
