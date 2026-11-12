#!/bin/bash
# Toy end-to-end run of every experiment-5 phase on the primary arm.
#
# TB_RVR_TAG redirects all artifacts to results_smoke/, so this can never overwrite a real
# result. The numbers are meaningless at this size — the point is that the splice builds,
# the untie works, the freeze-and-train path runs, and every phase writes its JSON.
# T0's reproduction gate is EXPECTED to fail here (it compares 16-sequence scores against
# the pipeline's 256-sequence numbers), so its exit code is deliberately not fatal.
set -uo pipefail

EXP="experiments/5_readout_vs_representation"
export TB_RVR_TAG="${TB_RVR_TAG:-smoke}"
ARMS="${SMOKE_ARMS:-mgnify}"
COMMON=(--arms ${ARMS} --device cuda --n-eval 16 --masks-per-pass 1)

echo "=== smoke: TB_RVR_TAG=${TB_RVR_TAG} arms=${ARMS} ==="

echo "--- T0 (gate failure expected at this size) ---"
python "${EXP}/t0_sanity.py" "${COMMON[@]}"
echo "t0 exit=$? (non-zero is expected: reproduction is checked against 256-seq numbers)"

set -e
echo "--- T1 ---"
python "${EXP}/t1_hybrids.py" "${COMMON[@]}"

echo "--- T2 ---"
python "${EXP}/t2_depth_ladder.py" "${COMMON[@]}" --k 0 1 12

echo "--- T3 ---"
python "${EXP}/t3_partial_attack.py" "${COMMON[@]}" \
  --budgets 16 --seeds 0 --max-epochs 2

echo "--- aggregate ---"
# aggregate.py exits non-zero when a consistency check fails, and the T0 gate is one of
# them — so at smoke size its failure is expected too. Every phase having produced its
# artifact is what this run is testing, so the smoke's own exit code must not inherit it.
set +e
python "${EXP}/aggregate.py" --arms ${ARMS}
agg_rc=$?
set -e
echo "aggregate exit=${agg_rc} (non-zero expected: it inherits the T0 gate)"

for arm in ${ARMS}; do
  for f in t0_sanity t1_hybrids t2_depth_ladder t3_partial_attack; do
    path="${EXP}/results_${TB_RVR_TAG}/${arm}/${f}.json"
    [ -s "${path}" ] || { echo "SMOKE FAILED: missing ${path}" >&2; exit 1; }
  done
done
for f in summary.json RESULTS.md fig_readout_vs_representation.png; do
  path="${EXP}/results_${TB_RVR_TAG}/${f}"
  [ -s "${path}" ] || { echo "SMOKE FAILED: missing ${path}" >&2; exit 1; }
done

echo "=== smoke OK — every phase wrote its artifact; see ${EXP}/results_${TB_RVR_TAG}/ ==="
