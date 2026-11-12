#!/bin/bash
#SBATCH --job-name=tb-anchor-t5m2
#SBATCH --array=0-7%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/7_pfam_reps_vs_retain/outputs/%x-%A_%a.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env_setup.sh"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
python -u experiments/7_pfam_reps_vs_retain/t5_members.py cluster \
  --shard "${SLURM_ARRAY_TASK_ID}" --n-shards "$(( ${SLURM_ARRAY_TASK_MAX:-7} + 1 ))" "$@"
