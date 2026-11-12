#!/bin/bash
#SBATCH --job-name=tb-cir-frontier-fine
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-14
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# FINE CIR frontier — resolves the 0.1% disruption region the coarse grid overshot.
# Lower rates + small steps_per_epoch (=4) so termination is checked often enough to LAND
# at 0.05-0.2% drift instead of jumping past it. Writes to checkpoints/cir_frontier_fine/
# <tag> + results/cir_frontier_fine/<tag>; collect with:
#   sbatch slurm/06b_cir_frontier.sbatch model=esm2_8m +frontier_dir=cir_frontier_fine
# Run:
#   sbatch slurm/03c_cir_frontier_fine.sbatch model=esm2_8m         # or model=esm2_35m
# --array is sized to the grid (3*5=15 -> 0-14); over-provisioned tasks self-skip.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"

RATES=(0.02 0.05 0.1)
RATIOS=(1.0005 1.001 1.002 1.005 1.01)
STEPS=4
MAXEP=200
CELLS=()
for r in "${RATES[@]}"; do for t in "${RATIOS[@]}"; do CELLS+=("$r $t"); done; done

N=${#CELLS[@]}
i="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$i" -ge "$N" ]; then echo "task $i >= ${N} cells; nothing to do"; exit 0; fi
read -r RATE RATIO <<< "${CELLS[$i]}"
TAG="r${RATE}_t${RATIO}"
echo "fine frontier cell ${i}/${N}: unlearn_rate=${RATE} terminate_ratio=${RATIO} steps_per_epoch=${STEPS} tag=${TAG}"

python scripts/03b_train_cir.py blocking=cir \
  blocking.unlearn_rate="${RATE}" blocking.terminate_retain_ppl_ratio="${RATIO}" \
  blocking.steps_per_epoch="${STEPS}" blocking.max_epochs="${MAXEP}" \
  paths.checkpoints="checkpoints/cir_frontier_fine/${TAG}" paths.results="results/cir_frontier_fine/${TAG}" \
  tracking.backend=none "$@"
