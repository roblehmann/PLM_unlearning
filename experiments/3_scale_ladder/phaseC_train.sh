#!/bin/bash
# Phase C of the 650M spot-check — SAFEGUARD B + COMPUTE-MATCHED CONTROL A' at a scale.
#
# Deliberately a thin submitter, not a new trainer: B and A' are produced by the EXISTING
# pipeline scripts (scripts/03_train_blocked.py, scripts/04_train_control.py) with nothing
# but `model=` and `paths.root=` overridden. Scale is a parameter, never a fork — if this
# script ever needs a scale-specific branch, the pipeline has a hard-coded shape and the
# fix belongs upstream, not here.
#
#   experiments/3_scale_ladder/phaseC_train.sh 650M            # submit the B -> A' chain
#   experiments/3_scale_ladder/phaseC_train.sh 150M 650M       # both rungs
#   LADDER_ROOT=runs/scale_ladder experiments/3_scale_ladder/phaseC_train.sh 650M
#
# A' is chained afterok on B because it reads B's realized FLOP count as its own stopping
# budget (control_Aprime.match_on = realized_flops). Refuses to submit at any scale whose
# Phase-B headroom gate did not pass — that is the compute-protection rule: never spend
# safeguard compute where the instrument does not resolve.
set -euo pipefail

REPO_ROOT="${TASKBLOCK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LADDER_ROOT="${LADDER_ROOT:-${REPO_ROOT}/runs/scale_ladder}"
HERE="${REPO_ROOT}/experiments/3_scale_ladder"
cd "${REPO_ROOT}"
mkdir -p "${LADDER_ROOT}" "${HERE}/outputs"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <scale> [<scale> ...]   e.g. $0 150M 650M" >&2
  exit 2
fi

hydra_model() {  # scale -> configs/model/<group>.yaml
  case "$1" in
    8M)   echo esm2_8m ;;
    35M)  echo esm2_35m ;;
    150M) echo esm2_150m ;;
    650M) echo esm2_650m ;;
    *) echo "unknown scale '$1'" >&2; exit 2 ;;
  esac
}

for SCALE in "$@"; do
  MODEL="$(hydra_model "${SCALE}")"
  ROOT="${LADDER_ROOT}/${SCALE}"

  # --- gate check (Phase B must have passed at this scale) ---
  # ACCEPT_MARGINAL_GATE=1 carries a scale forward when the only unmet gate criterion is
  # marginal. This is the documented 650M case: the tau-floor margin is 2.8 SEs against the
  # 3.0 threshold, while headroom (0.29), floor convergence and the base crossing tau are
  # all met, so the instrument resolves and B/A' are trained for the mechanism ladder.
  ACCEPT_MARGINAL_GATE="${ACCEPT_MARGINAL_GATE:-0}" python - "$LADDER_ROOT" "$SCALE" <<'PY'
import json, os, sys, pathlib
ladder_root, scale = sys.argv[1], sys.argv[2]
p = pathlib.Path(ladder_root) / "manifests" / "headroom_ladder.json"
if not p.exists():
    raise SystemExit(f"no {p}: run phaseB_headroom_gate.py before Phase C")
rows = {r["scale"]: r for r in json.load(open(p)).get("rows", [])}
r = rows.get(scale)
if r is None:
    raise SystemExit(f"no headroom-gate row for {scale}; run Phase B at this scale first")
if not r.get("gate_pass"):
    reasons = '; '.join(r.get('gate_fail_reasons', []))
    if os.environ.get("ACCEPT_MARGINAL_GATE") == "1":
        print(f"[{scale}] gate criterion not met ({reasons}); accepted as marginal "
              f"(ACCEPT_MARGINAL_GATE=1)\n"
              f"  headroom={r.get('headroom_rel'):.3f}, tau={r.get('tau'):.4f} -> training B, A'")
    else:
        raise SystemExit(f"headroom gate not passed at {scale}: {reasons}\n"
                         "  not spending safeguard compute where the instrument does not resolve. "
                         "Set ACCEPT_MARGINAL_GATE=1 to proceed on a marginal margin.")
else:
    print(f"[{scale}] gate PASS (rel_headroom={r['headroom_rel']:.3f}, tau={r['tau']:.4f}) -> training B, A'")
PY

  if [ ! -e "${ROOT}/data" ]; then
    echo "missing ${ROOT}/data — Phase B creates it (symlink to the anchor data root)" >&2
    exit 2
  fi

  OVERRIDES="model=${MODEL} paths.root=${ROOT}"
  if command -v sbatch >/dev/null 2>&1; then
    JB=$(sbatch --parsable --job-name="tb-ladder-B-${SCALE}" \
         "${HERE}/phaseC_train.sbatch" scripts/03_train_blocked.py ${OVERRIDES})
    JA=$(sbatch --parsable --dependency=afterok:"${JB}" --job-name="tb-ladder-Ap-${SCALE}" \
         "${HERE}/phaseC_train.sbatch" scripts/04_train_control.py ${OVERRIDES})
    echo "  submitted: B=${JB} -> A'=${JA}  (root ${ROOT})"
    echo "${SCALE} B ${JB}"  >> "${LADDER_ROOT}/slurm_jobs.txt"
    echo "${SCALE} A' ${JA}" >> "${LADDER_ROOT}/slurm_jobs.txt"
  else
    echo "  no sbatch on PATH — running inline"
    python scripts/03_train_blocked.py ${OVERRIDES}
    python scripts/04_train_control.py ${OVERRIDES}
  fi
done
