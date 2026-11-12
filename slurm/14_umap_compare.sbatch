#!/bin/bash
#SBATCH --job-name=tb-umap-cmp
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Base A vs task-blocked B, one joint UMAP fit per anchor (B differs between anchors, so
# a single global fit would be meaningless). ~430k points per anchor with full retain.
#   sbatch --dependency=afterok:<embedB_jobid> slurm/14_umap_compare.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/14_umap_compare.py "$@"
