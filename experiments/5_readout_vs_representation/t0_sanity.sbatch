#!/bin/bash
#SBATCH --job-name=tb-rvr-t0
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/5_readout_vs_representation/outputs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL
# T0 — the gate. Reproduces each arm's stored forget_ppl_n0 with this experiment's eval
# settings (mgnify: 13.1149 and 81,344.31 -> 6,202x), checks that the identity splice is
# bitwise a no-op, and audits which keys each head set actually transfers.
# Exits non-zero if any arm fails, so the DAG's afterok dependency stops the run rather
# than producing hybrid numbers that cannot be compared to the pipeline.
#   sbatch --array=0-4 experiments/5_readout_vs_representation/t0_sanity.sbatch
#   sbatch experiments/5_readout_vs_representation/t0_sanity.sbatch --arms mgnify
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"/experiments/5_readout_vs_representation/arm_from_array.sh
python experiments/5_readout_vs_representation/t0_sanity.py --device cuda \
  "${ARM_FLAG[@]}" "$@"
