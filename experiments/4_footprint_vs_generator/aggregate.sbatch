#!/bin/bash
#SBATCH --job-name=tb-fvg-agg
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/4_footprint_vs_generator/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# Aggregate tests 1-4 into results/{summary.json, RESULTS.md, fig_*.png}. CPU only, and
# tolerant of missing phases (it reports which ones are absent), so it is safe to re-run
# at any point to see the state of the experiment.
#   sbatch experiments/4_footprint_vs_generator/aggregate.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/4_footprint_vs_generator/aggregate.py "$@"
