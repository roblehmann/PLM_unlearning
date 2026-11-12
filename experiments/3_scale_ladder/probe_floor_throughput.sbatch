#!/bin/bash
#SBATCH --job-name=tb-ladder-tprobe
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --output=experiments/3_scale_ladder/outputs/%x-%A_%a.out
#SBATCH --array=0-1
# Floor throughput+memory probe on the REAL anchor data, to choose the relaunch batch for
# 150M (array task 0) and 650M (array task 1) without OOMing on the long-sequence tail.
# Short (a few timed steps per config), lands in the 4h partition.
#   sbatch experiments/3_scale_ladder/probe_floor_throughput.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
SCALES=(150M 650M)
# bf16 run: bf16 halves activation memory, so batches that OOM in fp32 may now fit AND run
# ~2x faster. Test a ladder of batches at 1024 to find the fastest that fits under ~90% VRAM.
python experiments/3_scale_ladder/probe_floor_throughput.py \
  --scale "${SCALES[${SLURM_ARRAY_TASK_ID:-1}]}" --amp "${AMP:-bfloat16}" \
  --configs "${CONFIGS:-16x1024 24x1024 32x1024 48x1024 64x1024}"
