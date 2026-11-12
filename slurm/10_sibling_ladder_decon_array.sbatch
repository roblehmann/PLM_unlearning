#!/bin/bash
#SBATCH --job-name=tb-ladder-decon
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-23
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Phase 10 on the DECONTAMINATED ladder, in its own run root so the original
# runs/MGYF18690_metagenomic results are untouched.
#   Grid: 8 rungs x {B,A_prime} x {lora r8, lora r32, full_ft} x {16,64,128} x 3 seeds = 432 cells
#   ~870 s/cell (from the previous ladder run) -> 18 cells/shard ~= 4.4 h on 24 shards.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/10_sibling_ladder_sweep.py \
    sweep=sibling_ladder_decon \
    model=esm2_35m \
    paths.root=runs/MGYF18690_ladder_decon \
    data.forget_family.pfam_id=MGYF0000018690 \
    tracking.backend=none "$@"
