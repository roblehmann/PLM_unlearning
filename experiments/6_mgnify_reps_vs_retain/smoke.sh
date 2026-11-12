#!/usr/bin/env bash
# Toy end-to-end run of every phase, into results_smoke/ (TB_MGREPS_TAG), on a few hundred
# sequences. The numbers it produces are meaningless; what it proves is that every phase
# runs, every artifact appears, the resume path is a no-op, and — the point — it PRINTS THE
# MEASURED tok/s FROM THE REAL PPL PATH, which is what the t2 array width should be sized
# from before committing ~9 GPU-h.
#
# Run it through SLURM:
#   sbatch experiments/6_mgnify_reps_vs_retain/smoke.sbatch
# then chain the real DAG behind it without waiting:
#   experiments/6_mgnify_reps_vs_retain/run_all.sh --after <smoke_jobid>
#
# SMOKE_HOMOLOGY=1 also exercises t2b against a small retain slice.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
EXP=experiments/6_mgnify_reps_vs_retain
export TB_MGREPS_TAG="${TB_MGREPS_TAG:-smoke}"
R="$EXP/results_${TB_MGREPS_TAG}"
DEV="${SMOKE_DEVICE:-cuda}"
N_REPS="${SMOKE_N_REPS:-200}"
N_RETAIN="${SMOKE_N_RETAIN:-200}"

rm -rf "$R"
mkdir -p "$EXP/outputs"

echo "=== t0 (gate, with the FULL cached-retain id check — it is a one-second check and it "
echo "===      is the assumption that matters, so the smoke does not skip it)"
python "$EXP/t0_corpus.py" --n-shards 2 --max-reps "$N_REPS" --max-retain "$N_RETAIN"

echo "=== t1 (embed + retain-cache cosine gate)"
python "$EXP/t1_embed.py" --device "$DEV" --limit "$N_REPS" --verify-retain 32

echo "=== t2 (both shards)"
python "$EXP/t2_ppl.py" --device "$DEV" --shard 0 --n-shards 2
python "$EXP/t2_ppl.py" --device "$DEV" --shard 1 --n-shards 2

echo "=== t2 shard 0 again — must be a no-op (the resume path is what saves a timed-out task)"
before=$(wc -l < "$R/ppl/shard_000_of_002.tsv")
python "$EXP/t2_ppl.py" --device "$DEV" --shard 0 --n-shards 2 | tee /dev/stderr | \
  grep -q "nothing to do" || { echo "SMOKE FAIL: resume path rescored work"; exit 1; }
after=$(wc -l < "$R/ppl/shard_000_of_002.tsv")
[ "$before" = "$after" ] || { echo "SMOKE FAIL: shard grew on resume ($before -> $after)"; exit 1; }

echo "=== t2x (masks_per_pass fidelity, tiny)"
python "$EXP/t2x_exact_check.py" --device "$DEV" --n 8 --max-len-cap 200

if [ "${SMOKE_HOMOLOGY:-0}" = "1" ]; then
  echo "=== t2b (mmseqs, small slice of both sides)"
  head -n 4000 data/splits/retain_train.fasta > "$R/retain_slice.fasta"
  head -n 400 data/MGnify/family_representative_seq_corpus/mgnifams_reps.fasta > "$R/reps_slice.fasta"
  python "$EXP/t2b_homology.py" --threads "${SLURM_CPUS_PER_TASK:-4}" -s 5.7 \
    --query "$R/reps_slice.fasta" --target "$R/retain_slice.fasta"
fi

echo "=== t3 (UMAP; n_neighbors clamps to the tiny point count)"
python "$EXP/t3_umap.py" --n-neighbors 15 --knn-k 5

echo "=== t4 (report)"
python "$EXP/t4_report.py"

echo "=== artifact assertions"
must_exist() { [ -s "$1" ] || { echo "SMOKE FAIL: missing or empty $1"; exit 1; }; }
must_exist "$R/corpus_manifest.json"
must_exist "$R/work_table.tsv"
must_exist "$R/meta/reps_meta.tsv"
must_exist "$R/embeddings/reps.npy"
must_exist "$R/embeddings/manifest.json"
must_exist "$R/ppl/shard_000_of_002.tsv"
must_exist "$R/ppl/shard_001_of_002.tsv"
must_exist "$R/ppl/shard_000_of_002.done.json"
must_exist "$R/ppl_exact_check.json"
must_exist "$R/umap/coords.npy"
must_exist "$R/umap/knn_mixing.json"
must_exist "$R/per_sequence.tsv.gz"
must_exist "$R/summary.json"
must_exist "$R/RESULTS.md"
for f in fig1_umap_reps_vs_retain fig3_ppl_distributions fig4_ppl_vs_length; do
  must_exist "$R/plots/$f.png"
done
python3 - "$R/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
need = ["protocol", "corpora", "ppl_contrasts", "length_control", "truncation",
        "composition", "verdicts", "checks"]
missing = [k for k in need if k not in s]
if missing:
    raise SystemExit(f"SMOKE FAIL: summary.json missing keys {missing}")
print("  summary.json keys ok")
PY

echo
echo "=== measured throughput (SIZE THE t2 ARRAY FROM THIS) ==="
python3 - "$R" <<'PY'
import glob, json, sys
tps = []
for p in sorted(glob.glob(f"{sys.argv[1]}/ppl/*.done.json")):
    d = json.load(open(p))
    if d.get("tokens_per_sec"):
        tps.append(d["tokens_per_sec"])
        print(f"  {p.split('/')[-1]}: {d['seqs_per_sec']:.1f} seq/s, {d['tokens_per_sec']:,.0f} tok/s")
if tps:
    m = sum(tps) / len(tps)
    total = 3.9e9   # token-forwards for the full 229,523-sequence work table
    print(f"  mean {m:,.0f} tok/s -> full run ~{total / m / 3600:.1f} GPU-h; "
          f"at 24 shards ~{total / m / 24 / 60:.0f} min/shard")
    if m < 60_000:
        print("  WARNING: below ~60k tok/s — widen the t2 array (run_all.sh --shards 48)")
PY

echo
echo "=== smoke OK — artifacts in $R"
