#!/bin/bash
# Promote a chosen CIR frontier cell's B into its own artifact tree and launch the
# durability pipeline on it, chained with SLURM dependencies:
#   Phase 5 control (A') -> 6 sweep -> 7 analyze ; and 8 weightspace -> 8b mechanism gate.
# This is the decisive test: does the surgical CIR deletion SURVIVE attacker fine-tuning?
#
# Usage:
#   ./slurm/run_cir_durability.sh <frontier_subdir> <cell_tag> <dest> [extra hydra overrides]
# e.g.
#   ./slurm/run_cir_durability.sh cir_frontier      r0.2_t1.01  cir_peak  model=esm2_35m
#   ./slurm/run_cir_durability.sh cir_frontier_fine r0.05_t1.001 cir_lowdis model=esm2_35m
#
# NOTE: thresholds.json (the τ / acquisition gate) is MODEL-SPECIFIC. The copied one is
# whatever Phase 3 produced (the 35M TAR run). For a durability run on a *different* model,
# regenerate it first:  python scripts/02_baseline_attack.py model=<m>  (then re-copy).
set -euo pipefail

FRDIR="${1:?frontier subdir, e.g. cir_frontier}"
TAG="${2:?cell tag, e.g. r0.2_t1.01}"
DEST="${3:?dest name, e.g. cir_peak}"
shift 3
EXTRA=("$@")   # extra hydra overrides, e.g. model=esm2_35m

ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
SRC_CK="checkpoints/${FRDIR}/${TAG}"; SRC_RS="results/${FRDIR}/${TAG}"
DST_CK="checkpoints/${DEST}";         DST_RS="results/${DEST}"

[ -d "${SRC_CK}/B" ] || { echo "ERROR: ${SRC_CK}/B not found — check the frontier subdir/tag."; exit 1; }
[ -f "${SRC_RS}/B_compute.json" ] || { echo "ERROR: ${SRC_RS}/B_compute.json not found."; exit 1; }

echo "promoting ${FRDIR}/${TAG}  ->  ${DEST}"
mkdir -p "${DST_CK}" "${DST_RS}"
rm -rf "${DST_CK}/B"; cp -r "${SRC_CK}/B" "${DST_CK}/B"
cp "${SRC_RS}/B_compute.json" "${SRC_RS}/cir_subspace.npz" "${DST_RS}/"
if [ -f results/thresholds.json ]; then cp results/thresholds.json "${DST_RS}/"; else
  echo "WARN: results/thresholds.json missing — Phase 7 (analyze) needs it. Run Phase 3 first."; fi

OV=(paths.checkpoints="${DST_CK}" paths.results="${DST_RS}")

j5=$(sbatch --parsable slurm/04_train_control.sbatch blocking=cir "${EXTRA[@]}" "${OV[@]}")
j6=$(sbatch --parsable --dependency=afterok:"${j5}" slurm/05_sweep_array.sbatch sweep=cir_budgets "${EXTRA[@]}" "${OV[@]}")
j7=$(sbatch --parsable --dependency=afterok:"${j6}" slurm/06_analyze.sbatch "${EXTRA[@]}" "${OV[@]}")
j8=$(sbatch --parsable --dependency=afterok:"${j5}" slurm/08_weightspace.sbatch "${EXTRA[@]}" "${OV[@]}")
j8b=$(sbatch --parsable --dependency=afterok:"${j8}" slurm/08b_cir_subspace.sbatch "${EXTRA[@]}" "${OV[@]}")

echo "submitted chain for ${DEST}:"
echo "  control(A')   = ${j5}"
echo "  sweep         = ${j6}  (afterok ${j5})"
echo "  analyze       = ${j7}  (afterok ${j6})  -> results/${DEST}/summary.json"
echo "  weightspace   = ${j8}  (afterok ${j5})"
echo "  mechanism8b   = ${j8b} (afterok ${j8}) -> results/${DEST}/cir_weightspace.json"
echo "watch: squeue -u \$USER ; results land under results/${DEST}/"
