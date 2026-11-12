#!/bin/bash
# Map SLURM_ARRAY_TASK_ID -> the --shard/--n-shards flags for t2_ppl.py. Sourced by
# t2_ppl.sbatch. Mirrors experiments/5_readout_vs_representation/arm_from_array.sh: the
# flags are spliced in AHEAD of "$@", so an explicit --shard on the sbatch command line
# still wins.
SHARD_FLAG=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  SHARD_FLAG=(--shard "${SLURM_ARRAY_TASK_ID}" --n-shards "${TB_MGREPS_SHARDS:-24}")
fi
export SHARD_FLAG
