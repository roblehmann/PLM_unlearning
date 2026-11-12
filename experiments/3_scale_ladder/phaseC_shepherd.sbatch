#!/bin/bash
#SBATCH --job-name=tb-ladder-C-shepherd
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=experiments/3_scale_ladder/outputs/%x-%j.out
#SBATCH --mail-type=FAIL
# Phase C shepherd (CPU-only). Which scales get B + A' trained is not knowable when the
# DAG is submitted — it depends on which rungs passed the Phase-B headroom gate — so this
# job reads the gate manifest at run time, submits the training chain only for passing
# scales, then BLOCKS until those jobs finish and exits non-zero if any failed. Blocking
# is the point: it makes `afterok` on this job an honest barrier for Phase D, which needs
# the checkpoints to exist.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"

LADDER_ROOT="${LADDER_ROOT:-${TASKBLOCK_ROOT}/runs/scale_ladder}"
HERE="${TASKBLOCK_ROOT}/experiments/3_scale_ladder"
JOBS_FILE="${LADDER_ROOT}/slurm_jobs.txt"

PASSED=$(python - "$LADDER_ROOT" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "manifests" / "headroom_ladder.json"
rows = json.load(open(p)).get("rows", []) if p.exists() else []
print(" ".join(r["scale"] for r in rows if r.get("gate_pass")))
PY
)
if [ -z "${PASSED}" ]; then
  echo "no scale passed the Phase-B headroom gate — nothing to train, and nothing downstream is valid."
  exit 1
fi

# Only train the configured scales (config train_scales), intersected with gate passes:
# 8M/35M reuse cached checkpoints, so re-training them would burn compute for no new row.
TRAIN_SCALES=$(python - "$HERE" "$PASSED" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1] + "/scale_ladder.yaml")
want = set(str(s) for s in cfg.train_scales)
print(" ".join(s for s in sys.argv[2].split() if s in want))
PY
)
echo "gate passed at: ${PASSED}"
echo "training B + A' at: ${TRAIN_SCALES:-<none>}"
[ -z "${TRAIN_SCALES}" ] && { echo "nothing to train (all passing scales reuse cached checkpoints)"; exit 0; }

BEFORE=$(wc -l < "${JOBS_FILE}" 2>/dev/null || echo 0)
LADDER_ROOT="${LADDER_ROOT}" "${HERE}/phaseC_train.sh" ${TRAIN_SCALES} || exit 1
IDS=$(tail -n +$((BEFORE + 1)) "${JOBS_FILE}" | awk '{print $3}')
echo "waiting on: ${IDS}"

for id in ${IDS}; do
  while squeue -h -j "${id}" >/dev/null 2>&1 && [ -n "$(squeue -h -j "${id}" 2>/dev/null)" ]; do
    sleep 120
  done
  STATE=$(sacct -n -X -j "${id}" -o State 2>/dev/null | head -1 | tr -d ' ')
  echo "  job ${id}: ${STATE}"
  case "${STATE}" in COMPLETED) ;; *) echo "  -> Phase C job ${id} did not complete; stopping the chain"; exit 1 ;; esac
done
echo "Phase C complete for: ${TRAIN_SCALES}"
