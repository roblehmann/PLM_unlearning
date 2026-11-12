#!/usr/bin/env bash
# Submit from anywhere. Pass --out PATH for separate runs; --limit N for a smoke run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
EXP=experiments/7_pfam_reps_vs_retain
mkdir -p "$EXP/outputs"
EMBED=$(sbatch --parsable "$EXP/embed.sbatch" "$@")
EMBED=${EMBED%%;*}
echo "Embedding job: $EMBED"
UMAP=$(sbatch --parsable --dependency="afterok:$EMBED" "$EXP/umap.sbatch" "$@")
UMAP=${UMAP%%;*}
echo "UMAP job: $UMAP"
printf 'embed\t%s\numap\t%s\n' "$EMBED" "$UMAP" > "$EXP/outputs/latest_jobs.tsv"
