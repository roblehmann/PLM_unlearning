#!/bin/bash
# One-command SLURM DAG for the 650M spot-check: A -> B -> (C per scale) -> D -> E -> F.
#
#   experiments/3_scale_ladder/run_ladder.sh                    # full ladder
#   experiments/3_scale_ladder/run_ladder.sh --mechanism-only   # skip the M point (Phase E)
#   experiments/3_scale_ladder/run_ladder.sh --from D           # resume at a phase
#
# Chaining is afterok, so a phase that exits non-zero stops everything downstream. That
# matters most at Phase B: it exits 2 when no scale clears the headroom gate and 3 on
# HEADROOM_EXHAUSTED, which is exactly when NOT to spend safeguard compute. Phase C is
# submitted by a small dependent shepherd job rather than up front, because which scales
# it may train is only known after B has run.
#
# If compute gets cut, drop Phase E, never Phase D: the ladder is the argument and the
# single M is confirmatory.
set -euo pipefail

REPO_ROOT="${TASKBLOCK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
HERE="${REPO_ROOT}/experiments/3_scale_ladder"
LADDER_ROOT="${LADDER_ROOT:-${REPO_ROOT}/runs/scale_ladder}"
cd "${REPO_ROOT}"
mkdir -p "${HERE}/outputs" "${LADDER_ROOT}"

MECHANISM_ONLY=0
FROM="A"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --mechanism-only) MECHANISM_ONLY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

phase_idx() { case "$1" in A) echo 0 ;; B) echo 1 ;; C) echo 2 ;; D) echo 3 ;; E) echo 4 ;; F) echo 5 ;;
                           *) echo "unknown phase '$1' (use A..F)" >&2; exit 2 ;; esac; }
FROM_IDX="$(phase_idx "${FROM}")"
want() { [ "$(phase_idx "$1")" -ge "${FROM_IDX}" ]; }   # run phase $1 iff at/after --from

command -v sbatch >/dev/null 2>&1 || { echo "sbatch not found — run the phases directly"; exit 2; }
export LADDER_ROOT
DEP=""
sub() {  # sub <script> [args...] -> echoes job id, chains afterok on the previous phase
  local script="$1"; shift
  local args=("$@")
  local id
  if [ -n "${DEP}" ]; then
    id=$(sbatch --parsable --dependency=afterok:"${DEP}" "${script}" "${args[@]}")
  else
    id=$(sbatch --parsable "${script}" "${args[@]}")
  fi
  echo "${id}"
}

echo "ladder root: ${LADDER_ROOT}"
if want A; then
  DEP=$(sub "${HERE}/phaseA_smoke.sbatch"); echo "  A smoke          : ${DEP}"
fi
if want B; then
  DEP=$(sub "${HERE}/phaseB_headroom_gate.sbatch"); echo "  B headroom gate  : ${DEP}"
fi
if want C; then
  # Shepherd: reads the gate manifest at run time and submits B/A' only where it passed,
  # then blocks until they finish — so afterok on it is a real barrier for the verify step.
  DEP=$(sub "${HERE}/phaseC_shepherd.sbatch"); echo "  C shepherd (B+A'): ${DEP}"
  DEP=$(sub "${HERE}/phaseC_verify.sbatch");   echo "  C verify         : ${DEP}"
fi
if want D; then
  DEP=$(sub "${HERE}/phaseD_mechanism.sbatch"); echo "  D mechanism      : ${DEP}"
fi
if want E && [ "${MECHANISM_ONLY}" -eq 0 ]; then
  DEP=$(sub "${HERE}/phaseE_m_point.sbatch"); echo "  E M point (650M) : ${DEP}"
fi
if want F; then
  DEP=$(sub "${HERE}/phaseF_aggregate.sbatch"); echo "  F aggregate      : ${DEP}"
fi
echo "job ids appended to ${LADDER_ROOT}/slurm_jobs.txt as Phase C submits them"
