#!/bin/bash
#SBATCH --job-name=tb-anchor-tests
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=experiments/7_pfam_reps_vs_retain/outputs/%x-%j.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env_setup.sh"
python -u -m pytest -q experiments/7_pfam_reps_vs_retain/test_anchors.py "$@"
