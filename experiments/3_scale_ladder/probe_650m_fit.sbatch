#!/bin/bash
#SBATCH --job-name=tb-ladder-probe
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:25:00
#SBATCH --output=experiments/3_scale_ladder/outputs/%x-%j.out
# Resource probe, not an experiment: confirms the a100 this project actually gets has
# enough VRAM for the heaviest 650M step in the ladder, and records the peak. Assuming it
# "fits comfortably on 40-80 GB" is how a 24h Phase C job dies at hour six, so this
# measures the real peak for:
#   (1) full-FT attacker/floor step  - params + grads + AdamW states + activations
#   (2) task-blocking outer step     - the tent-pole: K inner adversary steps then an
#                                      outer update, i.e. two optimizers live at once
#   (3) pseudo-perplexity eval       - inference at the configured eval batch size
# Short enough to land in the 4h partition.
#   sbatch experiments/3_scale_ladder/probe_650m_fit.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/slurm/env_setup.sh"
python experiments/3_scale_ladder/probe_650m_fit.py "$@"
