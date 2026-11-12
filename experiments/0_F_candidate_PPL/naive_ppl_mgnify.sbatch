#!/bin/bash
#SBATCH --job-name=tb-mgnify-ppl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/0_F_candidate_PPL/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Preliminary probe: PPL_naive (base ESM-2, no fine-tuning) on the MGnify metagenomic
# family candidates listed in experiments/0_F_candidate_PPL/data/mgnify_seed_seqs/manifest.tsv.
# Reuses the pipeline's own scoring code (src/eval, src/models) unmodified;
# writes experiments/0_F_candidate_PPL/results/naive_ppl_mgnify.csv. Submit from repo root:
#   sbatch experiments/0_F_candidate_PPL/naive_ppl_mgnify.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/0_F_candidate_PPL/naive_ppl_mgnify.py --device cuda "$@"
