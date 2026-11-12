#!/bin/bash
#SBATCH --job-name=tb-mgnify-floor-cl
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=experiments/1_naive_PPL_Floor/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# PPL_floor for the 4 MGnify F candidates (pair0,1,3,4 — pair2 dropped, see script
# docstring) using a CLUSTER-AWARE train/test split instead of floor_ppl_mgnify.py's
# plain random_split, which was found to leak near-duplicate seed sequences across
# train/test (mmseqs2 check: 67-100% of test seqs >=50% identity to train). Single-
# linkage clusters each family's seed pool via mmseqs2 (same identity threshold the
# pipeline's own leakage decontamination uses) and assigns whole clusters to
# train/test so no near-duplicate can appear on both sides. Needs a real GPU slot
# (5 full fine-tunes) plus CPU time for the mmseqs clustering pass. Submit from repo
# root:
#   sbatch experiments/1_naive_PPL_Floor/floor_ppl_mgnify_clustered.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/1_naive_PPL_Floor/floor_ppl_mgnify_clustered.py --device cuda "$@"
