#!/usr/bin/env bash
# Submit the anchor-screen DAG. Runs from anywhere; everything goes through SLURM.
#
#   {t1 similarity, t2 seqstats}  ->  t3 select  ->  t4 figures
#                                                      -> t5m0 fetch -> t5m1 profile
#                                                      -> t5m2 cluster[array] -> t5m3 embed
#                                                      -> t5m4 report
#
# Usage:
#   bash run_anchors.sh                       # full chain
#   bash run_anchors.sh --after 12345         # start only if job 12345 succeeds (smoke gate)
#   bash run_anchors.sh --from t3 --to t4     # a slice of the chain
#   bash run_anchors.sh --no-members          # stages t1-t4 only
#   bash run_anchors.sh --top 25 -- --space cen   # args after -- go to every python stage
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
EXP=experiments/7_pfam_reps_vs_retain
mkdir -p "$EXP/outputs"

STAGES=(t1 t2 t3 t4 t5m0 t5m1 t5m2 t5m3 t5m4)
declare -A SCRIPT=( [t1]=t1_similarity.sbatch [t2]=t2_seqstats.sbatch [t3]=t3_select.sbatch
                    [t4]=t4_figures.sbatch [t5m0]=t5m0_fetch.sbatch [t5m1]=t5m1_profile.sbatch
                    [t5m2]=t5m2_cluster.sbatch [t5m3]=t5m3_embed.sbatch [t5m4]=t5m4_report.sbatch )
FROM=t1; TO=t5m4; AFTER=""; TOP=""; PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    --after) AFTER="$2"; shift 2 ;;
    --top) TOP="$2"; shift 2 ;;
    --no-members) TO=t4; shift ;;
    --) shift; PASS+=("$@"); break ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

index_of() { local want=$1 i=0; for s in "${STAGES[@]}"; do [[ "$s" == "$want" ]] && { echo "$i"; return; }; i=$((i+1)); done; echo "-1"; }
FI=$(index_of "$FROM"); TI=$(index_of "$TO")
[[ "$FI" -lt 0 || "$TI" -lt 0 ]] && { echo "unknown stage: --from $FROM --to $TO" >&2; exit 2; }

submit() {  # submit <stage> <dependency-list> -> prints the job id on stdout
  local stage=$1 dep=$2 extra=()
  [[ -n "$TOP" && "$stage" != t1 && "$stage" != t2 ]] && extra+=(--top "$TOP")
  local args=(--parsable)
  [[ -n "$dep" ]] && args+=(--dependency="afterok:${dep}")
  local id
  id=$(sbatch "${args[@]}" "$EXP/${SCRIPT[$stage]}" "${extra[@]}" ${PASS[@]+"${PASS[@]}"})
  id=${id%%;*}
  printf '%-6s %-10s after %s\n' "$stage" "$id" "${dep:-none}" >&2
  echo -e "${stage}\t${id}" >> "$EXP/outputs/latest_anchor_jobs.tsv"
  echo "$id"
}

: > "$EXP/outputs/latest_anchor_jobs.tsv"
echo "Anchor screen DAG (tag='${TB_ANCHOR_TAG:-}')"
declare -A JOB
for i in "${!STAGES[@]}"; do
  stage=${STAGES[$i]}
  (( i < FI || i > TI )) && continue
  case "$stage" in
    t1|t2) dep="$AFTER" ;;
    t3)    dep="${JOB[t1]:-}"
           [[ -n "${JOB[t2]:-}" ]] && dep="${dep:+${dep}:}${JOB[t2]}"
           dep=${dep:-$AFTER} ;;
    t4)    dep="${JOB[t3]:-$AFTER}" ;;
    t5m0)  dep="${JOB[t4]:-${JOB[t3]:-$AFTER}}" ;;
    t5m1)  dep="${JOB[t5m0]:-$AFTER}" ;;
    t5m2)  dep="${JOB[t5m1]:-$AFTER}" ;;
    t5m3)  dep="${JOB[t5m2]:-$AFTER}" ;;
    t5m4)  dep="${JOB[t5m3]:-$AFTER}" ;;
  esac
  JOB[$stage]=$(submit "$stage" "$dep")
done
echo "job ids -> $EXP/outputs/latest_anchor_jobs.tsv"
