#!/bin/bash
# End-to-end smoke of the anchor screen on a 512-family slice, writing to
# results_smoke/anchors/ — never on top of the real artifacts. Runs the offline tests
# first, then every stage including one small live InterPro fetch and a GPU embed.
#SBATCH --job-name=tb-anchor-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=experiments/7_pfam_reps_vs_retain/outputs/%x-%j.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/env_setup.sh"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TB_ANCHOR_TAG=smoke
EXP=experiments/7_pfam_reps_vs_retain

echo "== offline tests =="
python -u -m pytest -q "$EXP/test_anchors.py"

echo "== t1 similarity, pass 1 (block 256, centered space only) =="
python -u "$EXP/t1_similarity.py" --limit-families 512 --limit-retain 4000 \
  --block 256 --topk 20 --spaces cen --store-matrix cen --calibration-size 512
cp "$EXP/results_smoke/anchors/matrix/family_metrics.tsv" "${TMPDIR:-/tmp}/metrics_block256.tsv"

echo "== t1 similarity, pass 2 (block 128, both spaces — this is the run t3/t4 consume) =="
python -u "$EXP/t1_similarity.py" --limit-families 512 --limit-retain 4000 \
  --block 128 --topk 20 --spaces raw,cen --store-matrix raw,cen --calibration-size 512

echo "== t2 sequence statistics =="
python -u "$EXP/t2_seqstats.py" --limit-families 512 --limit-retain 4000 --workers "${SLURM_CPUS_PER_TASK}"

echo "== t3 selection =="
python -u "$EXP/t3_select.py" --top 3

echo "== t4 figures =="
python -u "$EXP/t4_figures.py" --block-factor 2 --exact-subsample 200 --profile-sample 200 --top 3

echo "== t5 members (3 families, 50 members each) =="
python -u "$EXP/t5_members.py" fetch --top 3 --max-members 50 --page-size 50
python -u "$EXP/t5_members.py" profile --top 3 --threads "${SLURM_CPUS_PER_TASK}"
python -u "$EXP/t5_members.py" cluster --top 3 --threads "${SLURM_CPUS_PER_TASK}"
python -u "$EXP/t5_members.py" embed --top 3 --sample 40
python -u "$EXP/t5_members.py" report --top 3

echo "== assertions =="
python -u - <<'PY'
import json, sys
from pathlib import Path
base = Path('experiments/7_pfam_reps_vs_retain/results_smoke/anchors')
problems = []
for stage in ('matrix', 'seqstats', 'selection'):
    v = json.loads((base / stage / 'validation.json').read_text())
    if v.get('status') != 'passed':
        problems.append(f'{stage} validation: {v}')
required = ['matrix/family_metrics.tsv', 'matrix/sim_cen.f16.npy', 'matrix/summary_similarity.json',
            'seqstats/rep_sequence_stats.tsv', 'seqstats/retain_calibration.npz',
            'selection/families_scored.tsv', 'selection/shortlist.tsv',
            'selection/gate_waterfall.tsv', 'selection/RESULTS.md',
            'plots/similarity_distributions.png', 'plots/gate_waterfall.png',
            'members/verified_shortlist.tsv', 'members/RESULTS.md']
for rel in required:
    p = base / rel
    if not p.exists() or p.stat().st_size == 0:
        problems.append(f'missing or empty: {rel}')
water = (base / 'selection/gate_waterfall.tsv').read_text().strip().splitlines()[1:]
counts = [int(line.split('\t')[1]) for line in water]
if any(b > a for a, b in zip(counts, counts[1:])):
    problems.append(f'gate waterfall is not monotone: {counts}')
import csv, os
old_path = Path(os.environ.get('TMPDIR', '/tmp')) / 'metrics_block256.tsv'
if old_path.exists():
    old = {r['pfam_acc']: r for r in csv.DictReader(old_path.open(), delimiter='\t')}
    new = list(csv.DictReader((base / 'matrix/family_metrics.tsv').open(), delimiter='\t'))
    worst = max(abs(float(r['cen_iso_nn']) - float(old[r['pfam_acc']]['cen_iso_nn']))
                for r in new if r['pfam_acc'] in old)
    print(f'block-size invariance: max |d cen_iso_nn| between --block 256 and 128 = {worst:.2e}')
    if worst > 1e-6:
        problems.append(f'isolation depends on the block size ({worst:.2e})')
else:
    problems.append('block-size invariance copy missing')
summary = json.loads((base / 'matrix/summary_similarity.json').read_text())
raw = summary['spaces']['raw']['anisotropy']['mean_resultant_length']
cen = summary['spaces']['cen']['anisotropy']['mean_resultant_length']
print(f'anisotropy: raw |mean unit vector| = {raw:.4f}, centered = {cen:.4f}')
if not cen < raw:
    problems.append('centering did not reduce the shared direction')
scored = (base / 'selection/families_scored.tsv').read_text().splitlines()
print(f'{len(scored)-1} families scored; gate waterfall {counts}')
if problems:
    print('SMOKE FAILED:'); [print(' -', p) for p in problems]; sys.exit(1)
print('SMOKE PASSED')
PY
