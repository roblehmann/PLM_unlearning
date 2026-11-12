#!/bin/bash
#SBATCH --job-name=tb-mgreps-t2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/6_mgnify_reps_vs_retain/outputs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# T2 — the expensive phase: per-sequence pseudo-perplexity for all 229,523 sequences.
# Total work is ~3.9e9 token-forwards; at a conservative 120k tok/s that is ~9 GPU-h, so 24
# cost-balanced shards land at ~15-25 min each and the 4 h wall is a ~10x margin that
# absorbs a 3x throughput miss. Shards are balanced on predicted cost (L*ceil(L/8)) because
# PPL cost grows ~L^2 — an index split would time out on the long tail alone. Resumable:
# resubmit a failed task, do not restart the array.
#   sbatch --array=0-23 experiments/6_mgnify_reps_vs_retain/t2_ppl.sbatch
#   sbatch --array=7 experiments/6_mgnify_reps_vs_retain/t2_ppl.sbatch    # one shard again
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"/experiments/6_mgnify_reps_vs_retain/shard_from_array.sh
python experiments/6_mgnify_reps_vs_retain/t2_ppl.py --device cuda \
  "${SHARD_FLAG[@]}" "$@"
