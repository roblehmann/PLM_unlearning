#!/bin/bash
#SBATCH --job-name=tb-cir-frontier
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# CIR efficacy-vs-disruption frontier: one array task per (unlearn_rate, terminate_ratio)
# operating point. Each trains a B into checkpoints/cir_frontier/<tag> + results/
# cir_frontier/<tag> (with its forget-deletion + drift in B_compute.json). Collect the
# whole frontier with scripts/06b_cir_frontier.py afterwards.
#   sbatch slurm/03c_cir_frontier.sbatch model=esm2_8m          # debug grid on 8M
#   sbatch slurm/03c_cir_frontier.sbatch model=esm2_35m         # headline
# Keep --array=0-(#RATES*#RATIOS - 1) in sync with the grid below (default 2*5=10 -> 0-9).
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"

RATES=(0.1 0.2)
RATIOS=(1.001 1.002 1.005 1.01 1.02)
CELLS=()
for r in "${RATES[@]}"; do for t in "${RATIOS[@]}"; do CELLS+=("$r $t"); done; done

i="${SLURM_ARRAY_TASK_ID:-0}"
read -r RATE RATIO <<< "${CELLS[$i]}"
TAG="r${RATE}_t${RATIO}"
echo "frontier cell ${i}/${#CELLS[@]}: unlearn_rate=${RATE} terminate_ratio=${RATIO} tag=${TAG}"

python scripts/03b_train_cir.py blocking=cir \
  blocking.unlearn_rate="${RATE}" blocking.terminate_retain_ppl_ratio="${RATIO}" \
  blocking.steps_per_epoch=20 blocking.max_epochs=80 \
  paths.checkpoints="checkpoints/cir_frontier/${TAG}" paths.results="results/cir_frontier/${TAG}" \
  tracking.backend=none "$@"
