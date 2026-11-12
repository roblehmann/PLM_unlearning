#!/bin/bash
#SBATCH --job-name=tb-rvr-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/5_readout_vs_representation/outputs/%x-%j.out
#SBATCH --mail-type=FAIL
# End-to-end smoke: every phase at toy size on the primary arm only, writing to
# results_smoke/ via TB_RVR_TAG so it can never land on top of real artifacts. Proves the
# splice constructs, the freeze-and-train path runs, and each phase writes its JSON.
# The NUMBERS ARE NOT MEANINGFUL at this size (16 seqs, exact masking, 2 epochs) — in
# particular T0's reproduction gate is expected to FAIL, because it compares against
# 256-sequence pipeline numbers. Run this first, through SLURM.
#   sbatch experiments/5_readout_vs_representation/smoke.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
bash experiments/5_readout_vs_representation/smoke.sh
