#!/bin/bash
#SBATCH --job-name=tb-rvr-agg
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/5_readout_vs_representation/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# Aggregate T0-T3 into results/{summary.json, RESULTS.md, fig_*.png}. CPU only, and
# tolerant of missing phases (it reports which are absent), so it is safe to re-run at any
# point to see the state of the experiment. Exits non-zero if a cross-phase consistency
# check fails (T2 k=0 must equal T1's hybrids; T3's pre-attack PPPL must equal B).
#   sbatch experiments/5_readout_vs_representation/aggregate.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/5_readout_vs_representation/aggregate.py "$@"
