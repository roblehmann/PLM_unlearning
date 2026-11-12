#!/bin/bash
#SBATCH --job-name=tb-Fcand-overlap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/2_FR_overlap/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Preliminary probe: O(F,R) activation-subspace overlap (src.analysis.cir_fvsr.subspace_overlap,
# the same computation as scripts/08e_cir_overlap.py / Phase 8e) between each candidate F and
# the fixed canonical retain set R (data/splits/retain_train.fasta), for all 5 pairs under
# experiments/0_F_candidate_PPL/data. Forward-pass-only (no backprop, no checkpoints) — same
# inference tier as naive_ppl.sbatch, just 5x a single 08e-style anchor run. Submit from repo root:
#   sbatch experiments/2_FR_overlap/overlap.sbatch
#   sbatch experiments/2_FR_overlap/overlap.sbatch --n-pcs 12
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/2_FR_overlap/overlap.py --device cuda "$@"
