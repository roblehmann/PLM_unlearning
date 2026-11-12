#!/bin/bash
#SBATCH --job-name=tb-data
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Phase 1 data build (CPU; uses mmseqs2 + hmmer modules + network).
# Set the Pfam ids on the CLI, e.g.:
#   sbatch slurm/01_build_data.sbatch data.forget_family.pfam_id=PF00069 \
#       data.adjacent_family.pfam_id=PF07714 data.clan.pfam_id=CL0016
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/01_build_data.py tracking.backend=none "$@"
