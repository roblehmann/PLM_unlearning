#!/bin/bash
#SBATCH --job-name=tb-fvg-t3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/4_footprint_vs_generator/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Test 3 — build every subspace in ONE space (the output.dense input space, CIR's band) and
# cross-overlap them: CIR's realised edit, Delta_acquire, the F-distinctive footprint, the
# retain-shared basis, and the F_test / F_train generating directions.
# Writes results/<arm>/bases.npz, which t4 consumes — run this before t4. Needs t1's theta_acq.
#   sbatch experiments/4_footprint_vs_generator/t3_subspaces.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/4_footprint_vs_generator/t3_subspaces.py --device cuda "$@"
