#!/bin/bash
# Source this from the sbatch scripts to set up modules + the conda env on a SLURM cluster.
#
# Bio tools come from environment modules (no bioconda install needed); adjust the module
# names below to your site. The Python env is a conda env created from environment.yml
# MINUS the bioconda tools.

# --- bio tool modules (these bundle their OWN Miniconda and prepend it to PATH; we
#     load them first and activate the project env last so its python ends up first) ---
# Reduce CUDA fragmentation (helps the sweep on tighter-memory GPUs).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

module purge 2>/dev/null || true
module load mmseqs2/14.7   2>/dev/null || module load mmseqs2/14.7e284 2>/dev/null || echo "WARN: mmseqs2 module not loaded"
module load hmmer/3.3.2    2>/dev/null || echo "WARN: hmmer module not loaded"
module load cdhit/4.8.1    2>/dev/null || true

# --- activate the project's conda env ---
# Locate conda by FILE PATH, not via `command -v conda`: the modules above put a
# bundled Miniconda on PATH that would otherwise be picked and lacks our env.
_conda_sh=""
if [ -n "${TASKBLOCK_CONDA_SH:-}" ] && [ -f "${TASKBLOCK_CONDA_SH}" ]; then
  _conda_sh="${TASKBLOCK_CONDA_SH}"
else
  for _base in "$HOME/lib/miniconda3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge"; do
    if [ -f "${_base}/etc/profile.d/conda.sh" ]; then _conda_sh="${_base}/etc/profile.d/conda.sh"; break; fi
  done
fi
if [ -z "${_conda_sh}" ] && command -v conda >/dev/null 2>&1; then
  _conda_sh="$(conda info --base)/etc/profile.d/conda.sh"
fi

if [ -n "${_conda_sh}" ] && [ -f "${_conda_sh}" ]; then
  # shellcheck disable=SC1090
  source "${_conda_sh}"
  conda activate "${TASKBLOCK_ENV:-plm_unlearning}" \
    || echo "WARN: could not activate '${TASKBLOCK_ENV:-plm_unlearning}' (create it, or set TASKBLOCK_ENV)"
else
  echo "WARN: conda.sh not found; set TASKBLOCK_CONDA_SH to your conda's etc/profile.d/conda.sh"
fi

# --- repo root + logs ---
export TASKBLOCK_ROOT="${TASKBLOCK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "${TASKBLOCK_ROOT}/slurm/logs"
cd "${TASKBLOCK_ROOT}" || exit 1

echo "env ready: python=$(command -v python) | mmseqs=$(command -v mmseqs || echo none) | hmmemit=$(command -v hmmemit || echo none)"
if python -c "import transformers" 2>/dev/null; then
  echo "  transformers: OK ($(python -c 'import transformers; print(transformers.__version__)'))"
else
  echo "  WARN: 'transformers' NOT importable with $(command -v python) — the project env is not active."
fi
