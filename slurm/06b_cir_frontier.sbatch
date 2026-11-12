#!/bin/bash
#SBATCH --job-name=tb-cir-frontier-collect
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# Collect the CIR frontier (results/cir_frontier/*) into frontier.csv + a plot, with the
# kept-vs-collapsed mechanism readout per operating point. CPU only (weight-space math +
# checkpoint loads, no GPU). Run after slurm/03c_cir_frontier.sbatch.
#   sbatch slurm/06b_cir_frontier.sbatch model=esm2_8m
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/06b_cir_frontier.py "$@"
