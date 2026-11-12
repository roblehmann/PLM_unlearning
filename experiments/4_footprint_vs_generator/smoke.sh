#!/usr/bin/env bash
# End-to-end CPU smoke of all five phases at toy sizes, with TB_FVG_TAG=smoke so every
# artifact (including the throwaway theta_acq) lands beside the real ones instead of on
# top of them. Catches shape/API/plumbing errors in minutes; the numbers are meaningless.
#
#   experiments/4_footprint_vs_generator/smoke.sh            # both arms
#   experiments/4_footprint_vs_generator/smoke.sh --arms mgnify
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
EXP=experiments/4_footprint_vs_generator
export TB_FVG_TAG=smoke
ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(--arms pfam mgnify)

echo "=== smoke: t0 (skipped if mmseqs is absent; downstream falls back to native F_test)"
python $EXP/t0_decontaminate.py "${ARGS[@]}" || echo "  t0 unavailable — continuing"

echo "=== smoke: t1"
python $EXP/t1_acquire_gap.py --device cpu --n-eval 3 --n-train 4 --epochs 1 --batch-size 2 "${ARGS[@]}"

echo "=== smoke: t2"
python $EXP/t2_delta_rank.py --device cpu --n-eval 3 --ks 1 2 "${ARGS[@]}"

echo "=== smoke: t3"
python $EXP/t3_subspaces.py --device cpu --n-eval 3 --n-pcs 4 --corpus-seqs 4 --grad-seqs 2 \
  --pc-batch 2 --grad-batch 2 "${ARGS[@]}"

echo "=== smoke: t4"
python $EXP/t4_causal_ablation.py --device cpu --n-eval 3 --n-random 1 "${ARGS[@]}"

echo "=== smoke: aggregate"
python $EXP/aggregate.py "${ARGS[@]}"

echo
echo "SMOKE OK — artifacts in $EXP/results_smoke/ (delete when done; the real run uses results/)"
