#!/bin/bash
#SBATCH --job-name=tb-rvr-t2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=experiments/5_readout_vs_representation/outputs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL
# T2 — depth ladder. Walks the splice boundary down from the head (k = 0,1,2,4,8,12
# encoder layers), in both directions, turning T1's binary answer into a localization
# curve. k=0 reproduces T1's hybrids exactly; aggregate.py asserts it.
#   sbatch --array=0-4 experiments/5_readout_vs_representation/t2_depth_ladder.sbatch
#   sbatch experiments/5_readout_vs_representation/t2_depth_ladder.sbatch --arms mgnify
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"/experiments/5_readout_vs_representation/arm_from_array.sh
python experiments/5_readout_vs_representation/t2_depth_ladder.py --device cuda \
  "${ARM_FLAG[@]}" "$@"
