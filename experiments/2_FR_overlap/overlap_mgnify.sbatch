#!/bin/bash
#SBATCH --job-name=tb-mgnify-overlap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/2_FR_overlap/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# Preliminary probe: O(F,R) activation-subspace overlap (src.analysis.cir_fvsr.subspace_overlap,
# same computation as scripts/08e_cir_overlap.py / Phase 8e) for the MGnify F candidates that
# survived the leakage check (pair1, pair3 by default), against the fixed canonical retain set R
# (data/splits/retain_train.fasta). Uses the CLUSTER-AWARE train split
# (experiments/0_F_candidate_PPL/data/mgnify_seed_seqs/cluster_split/), not the plain random one.
# Forward-pass-only (no backprop, no checkpoints) — same inference tier as overlap.sbatch. Submit
# from repo root:
#   sbatch experiments/2_FR_overlap/overlap_mgnify.sbatch
#   sbatch experiments/2_FR_overlap/overlap_mgnify.sbatch --pairs pair1 pair3 pair4
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/2_FR_overlap/overlap_mgnify.py --device cuda "$@"
