#!/bin/bash
#SBATCH --job-name=tb-cir-repair
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# Phase 8d (CIR): attacker repair-delta localization — the mechanistic payoff figure.
# Reproduces ONE full-FT clean-F relearn on a CIR B (saving θ_attacked), then localizes
# Δ_attack = θ_attacked − θ_B in CIR's own kept/collapsed basis AND the independent
# F-distinctive-vs-R-shared basis, plus its alignment with the CIR update. See H1 vs H2.
#   sbatch slurm/08d_cir_repair.sbatch model=esm2_35m blocking=cir +op=cir35m_peak +budget_n=4096
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env_setup.sh"
python scripts/08d_cir_repair.py "$@"
