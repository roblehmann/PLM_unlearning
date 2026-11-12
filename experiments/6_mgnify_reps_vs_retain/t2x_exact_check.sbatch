#!/bin/bash
#SBATCH --job-name=tb-mgreps-t2x
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/6_mgnify_reps_vs_retain/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# T2x — scores 300 sequences per corpus at masks_per_pass=1 as well as 8, to check that the
# speedup is not biased DIFFERENTLY between the two corpora. The headline is a difference,
# so a differential approximation bias would be indistinguishable from the finding. ~15 min.
#   sbatch experiments/6_mgnify_reps_vs_retain/t2x_exact_check.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/6_mgnify_reps_vs_retain/t2x_exact_check.py --device cuda "$@"
