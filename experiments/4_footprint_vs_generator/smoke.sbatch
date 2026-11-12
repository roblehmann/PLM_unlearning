#!/bin/bash
#SBATCH --job-name=tb-fvg-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/4_footprint_vs_generator/outputs/%x-%j.out
#SBATCH --mail-type=FAIL
# End-to-end toy-size run of t1 -> t2 -> t3 -> t4 -> aggregate on real hardware, with
# TB_FVG_TAG=smoke so nothing lands on top of the real artifacts. Validates shapes, hooks,
# checkpoint loading and the aggregator; the NUMBERS ARE MEANINGLESS at these sizes.
# t0 is skipped here on purpose — it exercises the "no decontaminated set" fallback that
# every downstream script must survive.
#   sbatch experiments/4_footprint_vs_generator/smoke.sbatch
#   sbatch experiments/4_footprint_vs_generator/smoke.sbatch --arms pfam mgnify
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
set -e
export TB_FVG_TAG=smoke
E=experiments/4_footprint_vs_generator
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(--arms pfam mgnify)

python $E/t1_acquire_gap.py  --device cuda --n-eval 8 --n-train 16 --epochs 2 --batch-size 4 "${ARMS[@]}"
python $E/t2_delta_rank.py   --device cuda --n-eval 8 --ks 1 4 "${ARMS[@]}"
python $E/t3_subspaces.py    --device cuda --n-eval 8 --n-pcs 6 --corpus-seqs 16 --grad-seqs 8 \
                             --pc-batch 4 --grad-batch 4 "${ARMS[@]}"
python $E/t4_causal_ablation.py --device cuda --n-eval 8 --n-random 1 "${ARMS[@]}"
python $E/aggregate.py "${ARMS[@]}"
echo "SMOKE OK"
