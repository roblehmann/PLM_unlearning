#!/usr/bin/env bash
# Submit the whole experiment as a chained SLURM DAG (afterok, so a failed phase stops
# everything downstream). Submit from the repo root:
#
#   experiments/4_footprint_vs_generator/run_all.sh
#   experiments/4_footprint_vs_generator/run_all.sh --arms mgnify     # one arm
#   experiments/4_footprint_vs_generator/run_all.sh --from t3         # resume mid-DAG
#   experiments/4_footprint_vs_generator/run_all.sh --after <jobid>   # queue behind a job
#
# Phase order is forced by real data dependencies, not by convenience:
#   t0 (cpu, mmseqs)  -> the decontaminated F_test both arms are scored on
#   t1 (gpu, ~2-3 h)  -> theta_acq, written into each arm's own run root
#   t2 (gpu, ~1.5 h)  -> needs theta_acq
#   t3 (gpu, ~1.5 h)  -> needs theta_acq; writes bases.npz
#   t4 (gpu, ~1 h)    -> needs bases.npz
#   aggregate (cpu)   -> reads the four json artifacts
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
EXP=experiments/4_footprint_vs_generator
mkdir -p "$EXP/outputs" "$EXP/results"

FROM="t0"
AFTER=""            # an external job id the first phase waits on (e.g. the smoke run)
REUSE_ACQ=0        # pass --reuse-checkpoint to t1 (re-score an existing theta_acq)
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --after) AFTER="$2"; shift 2 ;;
    --reuse-acq) REUSE_ACQ=1; shift ;;   # t1-only; see below
    *) ARGS+=("$1"); shift ;;
  esac
done

PHASES=(t0 t1 t2 t3 t4 agg)
declare -A SCRIPT=(
  [t0]="$EXP/t0_decontaminate.sbatch"
  [t1]="$EXP/t1_acquire_gap.sbatch"
  [t2]="$EXP/t2_delta_rank.sbatch"
  [t3]="$EXP/t3_subspaces.sbatch"
  [t4]="$EXP/t4_causal_ablation.sbatch"
  [agg]="$EXP/aggregate.sbatch"
)
# t0 takes no --arms-independent flags beyond the shared ones; every phase accepts them.
started=0
dep="$AFTER"
declare -A JOB
for p in "${PHASES[@]}"; do
  if [ "$started" -eq 0 ] && [ "$p" != "$FROM" ]; then continue; fi
  started=1
  # --reuse-checkpoint is a t1-only flag (t0/t2/t3/t4 argparse would reject it), so inject
  # it for that phase alone rather than through the shared ARGS.
  extra=()
  [ "$p" = "t1" ] && [ "$REUSE_ACQ" -eq 1 ] && extra=(--reuse-checkpoint)
  if [ -n "$dep" ]; then
    id=$(sbatch --parsable --dependency=afterok:"$dep" "${SCRIPT[$p]}" "${ARGS[@]+"${ARGS[@]}"}" "${extra[@]+"${extra[@]}"}")
  else
    id=$(sbatch --parsable "${SCRIPT[$p]}" "${ARGS[@]+"${ARGS[@]}"}" "${extra[@]+"${extra[@]}"}")
  fi
  JOB[$p]=$id
  dep=$id
  echo "submitted $p -> job $id"
done

echo
echo "DAG: $(for p in "${PHASES[@]}"; do [ -n "${JOB[$p]:-}" ] && printf '%s(%s) ' "$p" "${JOB[$p]}"; done)"
echo "watch:  squeue -u \$USER --name=tb-fvg-t0,tb-fvg-t1,tb-fvg-t2,tb-fvg-t3,tb-fvg-t4,tb-fvg-agg"
echo "logs:   $EXP/outputs/"
