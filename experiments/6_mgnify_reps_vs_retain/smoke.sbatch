#!/bin/bash
#SBATCH --job-name=tb-mgreps-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/6_mgnify_reps_vs_retain/outputs/%x-%j.out
#SBATCH --mail-type=FAIL
# Toy end-to-end of every phase into results_smoke/. Any GPU will do (--gres=gpu:1, not
# a100) since it scores a few hundred sequences. Submit this FIRST, then chain the real DAG
# behind it with `run_all.sh --after <this jobid>` so a failing smoke cancels the chain and
# a passing one starts it with no human round-trip.
#   sbatch experiments/6_mgnify_reps_vs_retain/smoke.sbatch
#   SMOKE_HOMOLOGY=1 sbatch experiments/6_mgnify_reps_vs_retain/smoke.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
bash experiments/6_mgnify_reps_vs_retain/smoke.sh "$@"
