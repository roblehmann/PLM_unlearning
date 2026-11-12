#!/bin/bash
#SBATCH --job-name=tb-ladder-gate
#SBATCH --array=0-3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --output=experiments/3_scale_ladder/outputs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Phase B (GATING): per-scale headroom screen, ONE SCALE PER ARRAY TASK.
#
# An array rather than one serial job for two reasons: the four rungs are independent, so
# they should run in parallel; and a single job would have to fit 4 floor fits + 4 base
# sweeps (the 650M one alone can be several hours) inside one walltime. Per-scale tasks
# also mean a 650M timeout does not cost the 8M/35M/150M rows. Concurrent writes to
# headroom_ladder.json are safe — upsert_row takes an exclusive lock.
#
# Scales come from scale_ladder.yaml's `gate_scales`, or from a --scales list passed here;
# tasks whose index exceeds the list exit 0 as no-ops, so an oversized --array is harmless.
#   sbatch experiments/3_scale_ladder/phaseB_headroom_gate.sbatch
#   sbatch --array=0 experiments/3_scale_ladder/phaseB_headroom_gate.sbatch --scales 650M
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"

HERE="${TASKBLOCK_ROOT}/experiments/3_scale_ladder"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

# Split argv into an explicit scale list (after --scales) and everything else, so the
# array can index the scales while the remaining flags pass through untouched.
SCALES=(); PASSTHRU=(); MODE=""
for a in "$@"; do
  case "$a" in
    --scales) MODE="scales"; continue ;;
    --*) MODE="" ;;
  esac
  if [ "$MODE" = "scales" ]; then SCALES+=("$a"); else PASSTHRU+=("$a"); fi
done
if [ "${#SCALES[@]}" -eq 0 ]; then
  mapfile -t SCALES < <(python -c "
from omegaconf import OmegaConf
print('\n'.join(str(s) for s in OmegaConf.load('${HERE}/scale_ladder.yaml').gate_scales))")
fi

if [ "${IDX}" -ge "${#SCALES[@]}" ]; then
  echo "array task ${IDX}: no scale at this index (${#SCALES[@]} scales: ${SCALES[*]}) — nothing to do"
  exit 0
fi
SCALE="${SCALES[$IDX]}"
echo "array task ${IDX} -> scale ${SCALE}"
python "${HERE}/phaseB_headroom_gate.py" --device cuda --scales "${SCALE}" "${PASSTHRU[@]}"
