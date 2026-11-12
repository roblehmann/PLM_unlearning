#!/bin/bash
#SBATCH --job-name=tb-umap-plot
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Stage 2 of the UMAP figure: PCA-50 + one global UMAP fit over the union (~240k points),
# then the three-panel figure. CPU only — no --gres. Chain it behind the embed job:
#   sbatch --dependency=afterok:<embed_jobid> slurm/13_umap_plot.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/13_umap_plot.py "$@"
