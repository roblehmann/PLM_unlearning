#!/bin/bash
#SBATCH --job-name=tb-rvr-t1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=experiments/5_readout_vs_representation/outputs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# T1 — the headline. theta_0 trunk + B head vs B trunk + theta_0 head, on F and on the
# benign holdout, for all three head sets, plus the A' splice controls and the head-only
# alpha path. Depends on T0 passing.
#   sbatch --array=0-4 experiments/5_readout_vs_representation/t1_hybrids.sbatch
#   sbatch experiments/5_readout_vs_representation/t1_hybrids.sbatch --arms mgnify
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"/experiments/5_readout_vs_representation/arm_from_array.sh
python experiments/5_readout_vs_representation/t1_hybrids.py --device cuda \
  "${ARM_FLAG[@]}" "$@"
