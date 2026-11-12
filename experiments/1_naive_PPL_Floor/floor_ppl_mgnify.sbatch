#!/bin/bash
#SBATCH --job-name=tb-mgnify-floor
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=experiments/1_naive_PPL_Floor/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Preliminary probe: PPL_floor (full-FT to convergence) for each MGnify F candidate
# (experiments/0_F_candidate_PPL/data/mgnify_seed_seqs/manifest.tsv), combined with
# experiments/0_F_candidate_PPL/results/naive_ppl_mgnify.csv into rel_headroom / tau /
# headroom_ok. Actually trains (5 sequential full fine-tunes) — needs a real GPU slot.
# Pools here (67-1319 seed seqs) are far smaller than the Pfam pairs' (52k-81k), but
# these are full-length metagenomic proteins (mean ~500aa, vs short Pfam domain
# fragments), so each step is much slower per-sequence — pair0 alone (199 train seqs,
# max_epochs=10) took ~50min. Budgeted generously since patience=3 may not fire on
# every pair even with eval_every=1. Results checkpoint to CSV after every pair, so a
# timeout doesn't lose earlier pairs' results. Submit from repo root:
#   sbatch experiments/1_naive_PPL_Floor/floor_ppl_mgnify.sbatch --max-epochs 25
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/1_naive_PPL_Floor/floor_ppl_mgnify.py --device cuda "$@"
