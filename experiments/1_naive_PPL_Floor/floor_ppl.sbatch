#!/bin/bash
#SBATCH --job-name=tb-Fcand-floor
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=experiments/1_naive_PPL_Floor/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Preliminary probe: PPL_floor (full-FT to convergence on F_train) for each
# candidate pair, combined with experiments/0_F_candidate_PPL/results/naive_ppl.csv
# into rel_headroom / tau / headroom_ok. Actually trains (up to 5 sequential
# full fine-tunes, <=8192 seqs / <=10 epochs each) — needs a real GPU slot, unlike
# the inference-only naive_ppl.sbatch. Submit from repo root:
#   sbatch experiments/1_naive_PPL_Floor/floor_ppl.sbatch
#   sbatch experiments/1_naive_PPL_Floor/floor_ppl.sbatch --max-epochs 20
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/1_naive_PPL_Floor/floor_ppl.py --device cuda "$@"
