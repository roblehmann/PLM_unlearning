#!/bin/bash
#SBATCH --job-name=tb-decon-static
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# Uniform homology-controlled STATIC re-score across all run roots. mmseqs (CPU) + one GPU.
#   sbatch slurm/11_decon_static_rescore.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/11_decon_static_rescore.py --device cuda "$@"
