#!/bin/bash
#SBATCH --job-name=tb-umap-embed
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Stage 1 of the UMAP figure: mean-pooled ESM-2 35M embeddings for F / F' / ladder rungs
# across all three anchors, plus the full 194k retain corpus as ambient background.
# ~72M tokens total; the 2h budget is generous (measured retain alone is 55M tokens).
# Resumable — groups already cached under results/umap/embeddings are skipped.
#   sbatch slurm/12_embed_corpora.sbatch --smoke        # shape + throughput check first
#   sbatch slurm/12_embed_corpora.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/12_embed_corpora.py --device cuda "$@"
