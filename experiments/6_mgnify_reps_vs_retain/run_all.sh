#!/usr/bin/env bash
# Submit the whole experiment as a chained SLURM DAG (afterok, so a failed phase stops
# everything downstream). Submit from anywhere; it cds to the repo root itself.
#
#   experiments/6_mgnify_reps_vs_retain/run_all.sh
#   experiments/6_mgnify_reps_vs_retain/run_all.sh --after <jobid>    # queue behind the smoke
#   experiments/6_mgnify_reps_vs_retain/run_all.sh --from t2          # resume mid-DAG
#   experiments/6_mgnify_reps_vs_retain/run_all.sh --shards 48        # wider t2 array
#   experiments/6_mgnify_reps_vs_retain/run_all.sh --no-homology      # skip the mmseqs split
#
# Shape of the DAG and why:
#   t0  (cpu, ~2 m)    gate: corpora parse, the cached retain embeddings line up id-for-id,
#                      the HF snapshot is present, and the work is partitioned. Everything
#                      downstream is afterok on this, so a bad assumption costs no GPU time.
#   t1  (gpu, ~5 m)    embed the reps + verify the retain cache still reproduces.
#   t2  (gpu array)    the expensive phase — pseudo-perplexity for all 229,523 sequences.
#   t2x (gpu, ~15 m)   masks_per_pass fidelity: is the speedup biased differently per corpus?
#   t2b (cpu, ~40 m)   mmseqs novelty split; needs only the FASTAs, so it hangs off t0 and
#                      runs concurrently with everything else.
#   t3  (cpu, ~5 m)    the joint UMAP fit + kNN mixing. Needs only t1, so it overlaps t2.
#   t4  (cpu, ~5 m)    merge, statistics, figures, RESULTS.md — afterok on all of the above.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
EXP=experiments/6_mgnify_reps_vs_retain
mkdir -p "$EXP/outputs" "$EXP/results"

FROM="t0"
TO="t4"
AFTER=""
SHARDS=24
WITH_HOMOLOGY=1
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    --after) AFTER="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --no-homology) WITH_HOMOLOGY=0; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

# Resuming past t0 must not re-guess the shard count: t0 wrote the partition, and a
# different --n-shards here would silently re-slice the work, leaving sequences scored
# twice or not at all (t2 asserts on this too, as the second line of defence).
MANIFEST="$EXP/results/corpus_manifest.json"
if [ "$FROM" != "t0" ] && [ -f "$MANIFEST" ]; then
  FOUND=$(python3 -c "import json,sys; print(json.load(open('$MANIFEST')).get('n_shards',''))" 2>/dev/null || true)
  if [ -n "$FOUND" ] && [ "$FOUND" != "$SHARDS" ]; then
    echo "note: using n_shards=$FOUND from $MANIFEST (not $SHARDS)"
    SHARDS="$FOUND"
  fi
fi
export TB_MGREPS_SHARDS="$SHARDS"

phase_index() {
  case "$1" in t0) echo 0 ;; t1) echo 1 ;; t2) echo 2 ;; t2x) echo 2 ;; t2b) echo 2 ;;
               t3) echo 3 ;; t4) echo 4 ;; *) echo 99 ;; esac
}
want() { [ "$(phase_index "$1")" -ge "$(phase_index "$FROM")" ] && \
         [ "$(phase_index "$1")" -le "$(phase_index "$TO")" ]; }

declare -A JOB
dep_t0="$AFTER"

if want t0; then
  opts=(); [ -n "$dep_t0" ] && opts+=(--dependency=afterok:"$dep_t0")
  JOB[t0]=$(sbatch --parsable "${opts[@]+"${opts[@]}"}" "$EXP/t0_corpus.sbatch" \
            --n-shards "$SHARDS" "${ARGS[@]+"${ARGS[@]}"}")
  echo "submitted t0  -> ${JOB[t0]}"
fi

after_t0=""
[ -n "${JOB[t0]:-}" ] && after_t0="--dependency=afterok:${JOB[t0]}"
[ -z "${JOB[t0]:-}" ] && [ -n "$AFTER" ] && after_t0="--dependency=afterok:$AFTER"

if want t1; then
  JOB[t1]=$(sbatch --parsable ${after_t0} "$EXP/t1_embed.sbatch" "${ARGS[@]+"${ARGS[@]}"}")
  echo "submitted t1  -> ${JOB[t1]}"
fi

after_t1=""
[ -n "${JOB[t1]:-}" ] && after_t1="--dependency=afterok:${JOB[t1]}"

FANIN=()
if want t2; then
  JOB[t2]=$(sbatch --parsable ${after_t1} --array=0-$((SHARDS - 1)) \
            "$EXP/t2_ppl.sbatch" "${ARGS[@]+"${ARGS[@]}"}")
  echo "submitted t2  -> ${JOB[t2]} (array 0-$((SHARDS - 1)))"
  FANIN+=("${JOB[t2]}")

  JOB[t2x]=$(sbatch --parsable ${after_t1} "$EXP/t2x_exact_check.sbatch" "${ARGS[@]+"${ARGS[@]}"}")
  echo "submitted t2x -> ${JOB[t2x]}"
  FANIN+=("${JOB[t2x]}")

  if [ "$WITH_HOMOLOGY" -eq 1 ]; then
    JOB[t2b]=$(sbatch --parsable ${after_t0} "$EXP/t2b_homology.sbatch" "${ARGS[@]+"${ARGS[@]}"}")
    echo "submitted t2b -> ${JOB[t2b]}"
    FANIN+=("${JOB[t2b]}")
  fi
fi

if want t3; then
  JOB[t3]=$(sbatch --parsable ${after_t1} "$EXP/t3_umap.sbatch" "${ARGS[@]+"${ARGS[@]}"}")
  echo "submitted t3  -> ${JOB[t3]}"
  FANIN+=("${JOB[t3]}")
fi

if want t4; then
  dep=""
  if [ ${#FANIN[@]} -gt 0 ]; then
    dep="--dependency=afterok:$(IFS=:; echo "${FANIN[*]}")"
  fi
  JOB[t4]=$(sbatch --parsable ${dep} "$EXP/t4_report.sbatch" "${ARGS[@]+"${ARGS[@]}"}")
  echo "submitted t4  -> ${JOB[t4]}"
fi

echo
echo "DAG: $(for p in t0 t1 t2 t2x t2b t3 t4; do [ -n "${JOB[$p]:-}" ] && printf '%s(%s) ' "$p" "${JOB[$p]}"; done)"
echo "watch: squeue -u $USER --name=tb-mgreps-t0,tb-mgreps-t1,tb-mgreps-t2,tb-mgreps-t2x,tb-mgreps-t2b,tb-mgreps-t3,tb-mgreps-t4"
echo "logs:  $EXP/outputs/"
echo "results: $EXP/results/ (RESULTS.md, summary.json, plots/)"
