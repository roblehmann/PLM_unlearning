#!/bin/bash
#SBATCH --job-name=tb-ladder-smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/3_scale_ladder/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# Phase A: prove the pipeline is scale-parameterized (no hard-coded 12-layer/480-width
# assumptions) by running it end to end at every rung on toy budgets. Cheap, and it is
# the only thing standing between a typo and six wasted GPU-hours at 650M.
#   sbatch experiments/3_scale_ladder/phaseA_smoke.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/3_scale_ladder/phaseA_smoke.py --device cuda "$@"
