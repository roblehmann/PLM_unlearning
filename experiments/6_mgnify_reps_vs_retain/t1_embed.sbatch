#!/bin/bash
#SBATCH --job-name=tb-mgreps-t1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/6_mgnify_reps_vs_retain/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# T1 — embed the 35,459 MGnify representatives (~1 min at the measured 667 seq/s; the wall
# is dominated by model load) and verify that the cached retain embeddings still reproduce
# under today's code. That verification is the gate which makes reusing the cache
# legitimate: without it, a changed numerical regime could masquerade as biology.
#   sbatch experiments/6_mgnify_reps_vs_retain/t1_embed.sbatch
#   sbatch experiments/6_mgnify_reps_vs_retain/t1_embed.sbatch --reembed-retain
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/6_mgnify_reps_vs_retain/t1_embed.py --device cuda "$@"
