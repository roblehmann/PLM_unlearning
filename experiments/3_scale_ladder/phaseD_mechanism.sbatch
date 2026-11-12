#!/bin/bash
#SBATCH --job-name=tb-ladder-mech
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/3_scale_ladder/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# Phase D: the priority readout — displacement + per-layer profile + 21-alpha LMC walk
# per scale. Displacement is seconds; the LMC walk is 21 pseudo-PPL evaluations on two
# eval sets. High --mem because three 650M state dicts are held on the CPU side.
#   sbatch experiments/3_scale_ladder/phaseD_mechanism.sbatch --scales 150M 650M
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/3_scale_ladder/phaseD_mechanism.py --device cuda "$@"
