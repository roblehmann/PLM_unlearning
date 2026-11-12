#!/bin/bash
#SBATCH --job-name=tb-mgreps-t2b
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/6_mgnify_reps_vs_retain/outputs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
# T2b — mmseqs2 search of the 35k representatives against the 194k retain sequences, to
# split them into "has a UniRef50 homolog" and "novel". CPU only; mmseqs comes from the
# module load in env_setup.sh. Expect 20-60 min at -s 7.5 on 16 cores; the 4 h wall is the
# margin, and -s 5.7 is the ~3-4x faster fallback if it ever overruns. Depends only on T0,
# so it runs alongside the GPU phases.
#   sbatch experiments/6_mgnify_reps_vs_retain/t2b_homology.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/6_mgnify_reps_vs_retain/t2b_homology.py \
  --threads "${SLURM_CPUS_PER_TASK:-16}" "$@"
