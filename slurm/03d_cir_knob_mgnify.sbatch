#!/bin/bash
#SBATCH --job-name=tb-cir-knob-mg
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# CIR KNOB sweep for MGnify. The frontier (rate x terminate_ratio) never blocked (max 2.8x),
# but O(F,R) removable_topk=0.649 (~Pfam 0.672) proves a LARGE F-distinctive subspace EXISTS.
# This sweeps the subspace knobs the frontier held FIXED -- subspace_mode x n_act_pcs x
# layer_band -- at the best frontier operating point (unlearn_rate=0.4, terminate_ratio=1.02),
# to test whether ANY config reaches the removable subspace and blocks F within the util budget.
# Each cell -> B into runs/MGYF18690_metagenomic/{checkpoints,results}/cir_knob/<tag>; read
# forget_ppl_ratio_B_over_base + benign_drift_rel from each B_compute.json.
#   sbatch slurm/03d_cir_knob_mgnify.sbatch model=esm2_35m
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"

RUN=runs/MGYF18690_metagenomic
MODES=(retain_pca forget_pca)
NPCS=(24 48)
BANDS=("[3,4,5,6,7,8]" "[1,2,3,4,5,6]" "[5,6,7,8,9,10]")
CELLS=()
for m in "${MODES[@]}"; do for n in "${NPCS[@]}"; do for b in "${BANDS[@]}"; do CELLS+=("$m|$n|$b"); done; done; done

i="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$i" -ge "${#CELLS[@]}" ]; then echo "task $i >= ${#CELLS[@]} cells; nothing to do"; exit 0; fi
IFS='|' read -r MODE NPC BAND <<< "${CELLS[$i]}"
BTAG=$(echo "$BAND" | tr -d '[] ' | awk -F, '{print $1"-"$NF}')
TAG="${MODE}_n${NPC}_b${BTAG}"
echo "CIR knob cell ${i}/${#CELLS[@]}: subspace_mode=${MODE} n_act_pcs=${NPC} layer_band=${BAND} tag=${TAG}"

python scripts/03b_train_cir.py blocking=cir \
  blocking.subspace_mode="${MODE}" blocking.n_act_pcs="${NPC}" "blocking.layer_band=${BAND}" \
  blocking.unlearn_rate=0.4 blocking.terminate_retain_ppl_ratio=1.02 \
  blocking.steps_per_epoch=20 blocking.max_epochs=80 \
  paths.root="${RUN}" \
  paths.checkpoints="${RUN}/checkpoints/cir_knob/${TAG}" \
  paths.results="${RUN}/results/cir_knob/${TAG}" \
  tracking.backend=none "$@"
