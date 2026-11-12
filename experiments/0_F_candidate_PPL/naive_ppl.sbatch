#!/bin/bash
#SBATCH --job-name=tb-Fcand-ppl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/0_F_candidate_PPL/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Preliminary probe: PPL_naive (base ESM-2, no fine-tuning) on every candidate
# test set under experiments/0_F_candidate_PPL/data/pairN/*_test.fasta.
# Reuses the pipeline's own scoring code (src/eval, src/models) unmodified;
# writes experiments/0_F_candidate_PPL/results/naive_ppl.csv. Submit from repo root:
#   sbatch experiments/0_F_candidate_PPL/naive_ppl.sbatch
#   sbatch experiments/0_F_candidate_PPL/naive_ppl.sbatch --model facebook/esm2_t12_35M_UR50D
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/0_F_candidate_PPL/naive_ppl.py --device cuda "$@"
