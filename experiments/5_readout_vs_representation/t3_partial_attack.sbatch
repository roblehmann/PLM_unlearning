#!/bin/bash
#SBATCH --job-name=tb-rvr-t3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=experiments/5_readout_vs_representation/outputs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# T3 — head-only vs trunk-only relearning of B at n = 16/64/256 x 3 seeds. One array task
# per arm; the full-fine-tune baseline is read from the main sweep on disk, so no baseline
# compute is spent here. The only phase that trains, and it writes no checkpoints.
#   sbatch --array=0-4 experiments/5_readout_vs_representation/t3_partial_attack.sbatch
#   sbatch --array=0   experiments/5_readout_vs_representation/t3_partial_attack.sbatch
# Without --array (or with extra flags) it falls through to whatever "$@" specifies:
#   sbatch experiments/5_readout_vs_representation/t3_partial_attack.sbatch --arms mgnify
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"/experiments/5_readout_vs_representation/arm_from_array.sh

python experiments/5_readout_vs_representation/t3_partial_attack.py --device cuda \
  "${ARM_FLAG[@]}" "$@"
