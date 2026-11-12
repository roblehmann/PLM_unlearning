#!/bin/bash
#SBATCH --job-name=tb-ladder-mpoint
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=experiments/3_scale_ladder/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Phase E: the one confirmatory M at 650M — ~6 budgets x {A, A', B} x 1 seed, full-FT
# attacker. Per-cell results are flushed to results/m_point_cells.json as they finish and
# the script resumes from them, so a time-limit hit costs only the cell in flight.
#   sbatch experiments/3_scale_ladder/phaseE_m_point.sbatch --scale 650M
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/3_scale_ladder/phaseE_m_point.py --device cuda "$@"
