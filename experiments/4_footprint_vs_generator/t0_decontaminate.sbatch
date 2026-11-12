#!/bin/bash
#SBATCH --job-name=tb-fvg-t0
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=experiments/4_footprint_vs_generator/outputs/%x-%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT_80
# Test 0 — put both arms' F_test on the SAME homology-decontamination footing (mmseqs).
# CPU only. Must run before t1/t3/t4, which score on the surviving subset.
#   sbatch experiments/4_footprint_vs_generator/t0_decontaminate.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/4_footprint_vs_generator/t0_decontaminate.py --threads "${SLURM_CPUS_PER_TASK:-8}" "$@"
