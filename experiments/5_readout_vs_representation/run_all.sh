#!/usr/bin/env bash
# Submit the whole experiment as a chained SLURM DAG (afterok, so a failed phase stops
# everything downstream). Submit from anywhere; it cds to the repo root itself.
#
#   experiments/5_readout_vs_representation/run_all.sh
#   experiments/5_readout_vs_representation/run_all.sh --arms mgnify     # one arm
#   experiments/5_readout_vs_representation/run_all.sh --from t2         # resume mid-DAG
#   experiments/5_readout_vs_representation/run_all.sh --to t2           # stop after t2
#   experiments/5_readout_vs_representation/run_all.sh --after <jobid>   # queue behind a job
#
# Phase order and why it is forced:
# Every GPU phase runs as an array, one task per arm (walls below are PER ARM, measured
# from the smoke run's per-eval cost, not guessed):
#   t0  (gpu, ~10 m)  gate: reproduce each arm's stored forget_ppl_n0 + audit the splice.
#                     afterok means a failed gate stops the DAG rather than producing
#                     hybrid numbers that cannot be compared to the pipeline.
#   t1  (gpu, ~2 h)   the headline hybrids + A' controls + head-only alpha path
#   t2  (gpu, ~1.2 h) depth ladder; k=0 reproduces t1, which aggregate.py asserts
#   t3  (gpu, ~11 h)  head-only vs trunk-only relearning of B — by far the dominant cost,
#                     because each of its 18 cells re-scores 256 sequences ~11 times, the
#                     same protocol the main sweep used
#   agg (cpu, ~5 m)   summary.json + RESULTS.md + figures + consistency checks
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
EXP=experiments/5_readout_vs_representation
mkdir -p "$EXP/outputs" "$EXP/results"

FROM="t0"
TO="agg"            # last phase to submit; --to t2 stops before the expensive t3
AFTER=""            # an external job id the first phase waits on (e.g. the smoke run)
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    --after) AFTER="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

PHASES=(t0 t1 t2 t3 agg)
declare -A SCRIPT=(
  [t0]="$EXP/t0_sanity.sbatch"
  [t1]="$EXP/t1_hybrids.sbatch"
  [t2]="$EXP/t2_depth_ladder.sbatch"
  [t3]="$EXP/t3_partial_attack.sbatch"
  [agg]="$EXP/aggregate.sbatch"
)

# Every GPU phase fans out one array task per arm — sharding by arm is safe because each
# writes its own results/<arm>/tN_*.json. Suppressed when the caller pinned --arms, where a
# single task running that one arm is what is wanted. afterok on an array waits for all
# tasks, so the DAG's gating still holds.
array_flag() {
  for a in "${ARGS[@]+"${ARGS[@]}"}"; do
    [ "$a" = "--arms" ] && return 0
  done
  echo "--array=0-4"
}

# aggregate.py takes --arms but none of the eval flags the GPU phases share, so anything
# beyond --arms is filtered out of its argument list.
agg_args() {
  local out=() take=0
  for a in "${ARGS[@]+"${ARGS[@]}"}"; do
    if [ "$a" = "--arms" ]; then take=1; out+=("$a"); continue; fi
    if [ "$take" -eq 1 ]; then
      case "$a" in --*) take=0 ;; *) out+=("$a"); continue ;; esac
    fi
  done
  printf '%s\n' "${out[@]+"${out[@]}"}"
}

started=0
dep="$AFTER"
declare -A JOB
for p in "${PHASES[@]}"; do
  if [ "$started" -eq 0 ] && [ "$p" != "$FROM" ]; then continue; fi
  started=1

  sbatch_opts=()
  phase_args=("${ARGS[@]+"${ARGS[@]}"}")
  if [ "$p" = "agg" ]; then
    mapfile -t phase_args < <(agg_args)
  else
    flag="$(array_flag)"
    [ -n "$flag" ] && sbatch_opts+=("$flag")
  fi
  [ -n "$dep" ] && sbatch_opts+=(--dependency=afterok:"$dep")

  id=$(sbatch --parsable "${sbatch_opts[@]+"${sbatch_opts[@]}"}" "${SCRIPT[$p]}" \
       "${phase_args[@]+"${phase_args[@]}"}")
  JOB[$p]=$id
  dep=$id
  echo "submitted $p -> job $id"
  [ "$p" = "$TO" ] && break
done

echo
echo "DAG: $(for p in "${PHASES[@]}"; do [ -n "${JOB[$p]:-}" ] && printf '%s(%s) ' "$p" "${JOB[$p]}"; done)"
echo "watch:  squeue -u \$USER --name=tb-rvr-t0,tb-rvr-t1,tb-rvr-t2,tb-rvr-t3,tb-rvr-agg"
echo "logs:   $EXP/outputs/"
