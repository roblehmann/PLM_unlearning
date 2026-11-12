#!/bin/bash
#SBATCH --job-name=tb-cir-frontier-mg
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# CIR efficacy-vs-disruption frontier for the MGnify run. The default CIR op-point
# (unlearn_rate=0.01, terminate_retain_ppl_ratio=1.001 = 0.1% budget) DELETED NOTHING on
# this family (forget ratio 1.0x). This grid spans more aggressive rates x looser
# disruption budgets to find a point that ACTUALLY blocks F (forget ratio >=2x) while
# keeping benign drift acceptable (utility gate <=2%). Each array task trains one B into
# runs/MGYF18690_metagenomic/{checkpoints,results}/cir_frontier/<tag> with its
# forget-deletion + drift in B_compute.json.
#   sbatch slurm/03c_cir_frontier_mgnify.sbatch model=esm2_35m
# Then read the frontier: for each cell, forget_ppl_ratio_B_over_base vs benign_drift_rel.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"

RUN=runs/MGYF18690_metagenomic
RATES=(0.1 0.2 0.4)
RATIOS=(1.005 1.01 1.02 1.05)
CELLS=()
for r in "${RATES[@]}"; do for t in "${RATIOS[@]}"; do CELLS+=("$r $t"); done; done

i="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$i" -ge "${#CELLS[@]}" ]; then echo "task $i >= ${#CELLS[@]} cells; nothing to do"; exit 0; fi
read -r RATE RATIO <<< "${CELLS[$i]}"
TAG="r${RATE}_t${RATIO}"
echo "MGnify CIR frontier cell ${i}/${#CELLS[@]}: unlearn_rate=${RATE} terminate_ratio=${RATIO} tag=${TAG}"

python scripts/03b_train_cir.py blocking=cir \
  blocking.unlearn_rate="${RATE}" blocking.terminate_retain_ppl_ratio="${RATIO}" \
  blocking.steps_per_epoch=20 blocking.max_epochs=80 \
  paths.root="${RUN}" \
  paths.checkpoints="${RUN}/checkpoints/cir_frontier/${TAG}" \
  paths.results="${RUN}/results/cir_frontier/${TAG}" \
  tracking.backend=none "$@"
