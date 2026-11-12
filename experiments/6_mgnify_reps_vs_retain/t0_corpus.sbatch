#!/bin/bash
#SBATCH --job-name=tb-mgreps-t0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/6_mgnify_reps_vs_retain/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# T0 — the gate. Parses both corpora, verifies that the cached retain embeddings line up
# with retain_train.fasta id-for-id, builds the length-matched retain arm and the
# cost-balanced shard plan. CPU only: it is pure parsing over 230k sequences, ~2 min. Every
# later phase depends on it via afterok, so a failure here costs no GPU time at all.
#   sbatch experiments/6_mgnify_reps_vs_retain/t0_corpus.sbatch
#   sbatch experiments/6_mgnify_reps_vs_retain/t0_corpus.sbatch --n-shards 48
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/6_mgnify_reps_vs_retain/t0_corpus.py "$@"
