#!/bin/bash
#SBATCH --job-name=tb-ladder-report
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=experiments/3_scale_ladder/outputs/%x-%j.out
# Phase F: figure + table + results paragraph, regenerated from the three manifests.
# CPU-only and cheap — re-run it any time a manifest changes rather than editing outputs.
#   sbatch experiments/3_scale_ladder/phaseF_aggregate.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/3_scale_ladder/phaseF_aggregate.py "$@"
