#!/bin/bash
#SBATCH --job-name=tb-ladder-decon-analyze
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/10b_sibling_ladder_analyze.py \
    model=esm2_35m \
    paths.root=runs/MGYF18690_ladder_decon \
    data.forget_family.pfam_id=MGYF0000018690 \
    tracking.backend=none "$@"
