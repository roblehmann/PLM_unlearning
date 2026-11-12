"""Cluster-aware train/test split + PPL_floor test for the MGnify F candidates that
survived the leakage check (pair0, pair1, pair3, pair4 — pair2 excluded by default:
its whole 94-seq pool was found to be internally redundant, with every held-out
sequence at >=50% identity to something in train under a plain random split, leaving
zero clean test sequences after filtering).

floor_ppl_mgnify.py's plain random_split let near-duplicate seed sequences leak across
train/test (verified via mmseqs2: 67-100% of test sequences had >=50% identity to
something in train, up to exact 1.000 duplicates) — the same mechanism already
diagnosed as the root cause of the old PF00106 run's inflated headroom
(headroom-collapse-cause). This script fixes that by single-linkage clustering each
family's seed pool via mmseqs2 all-vs-self search at the SAME 0.50 identity threshold
the pipeline's own leakage decontamination uses, then assigning whole clusters to
train or test so no near-duplicate pair can land on both sides. Reuses
src.train.attack.run_attack for the floor fit and the same mmseqs2 command shape as
src.data.leakage_filter.mmseqs_search (which can't be reused directly here — it
discards the target id, which a same-pool clustering pass needs).

    python experiments/1_naive_PPL_Floor/floor_ppl_mgnify_clustered.py

Writes derived splits to
experiments/0_F_candidate_PPL/data/mgnify_seed_seqs/cluster_split/<pair>/{train,test}.fasta
(kept for inspection/reuse) and results/floor_ppl_mgnify_clustered.csv, plus the same
live per-epoch history logging as floor_ppl_mgnify.py.
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANDIDATES_DIR = ROOT / "experiments" / "0_F_candidate_PPL"
DEFAULT_PAIRS = ["pair0", "pair1", "pair3", "pair4"]

from src.eval.pseudo_perplexity import acquisition_threshold  # noqa: E402
from src.models.load import load_tokenizer  # noqa: E402
from src.train.attack import run_attack  # noqa: E402
from src.utils.io import ensure_dir, read_fasta, save_json, write_fasta  # noqa: E402
from src.utils.tracking import JSONLTracker  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=str(CANDIDATES_DIR / "data" / "mgnify_seed_seqs" / "manifest.tsv"))
    p.add_argument("--naive-csv", default=str(CANDIDATES_DIR / "results" / "naive_ppl_mgnify.csv"),
                   help="experiment-0 output, used for PPL_naive per pair's F family")
    p.add_argument("--split-dir", default=str(CANDIDATES_DIR / "data" / "mgnify_seed_seqs" / "cluster_split"),
                   help="where derived cluster-aware train/test fastas are written")
    p.add_argument("--out", default=str(HERE / "results" / "floor_ppl_mgnify_clustered.csv"))
    p.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, help="pair2 dropped by default; see module docstring")
    p.add_argument("--model", default="facebook/esm2_t12_35M_UR50D", help="HF id (esm2_35m headline model)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--identity-threshold", type=float, default=0.50,
                    help="mmseqs fident >= this -> same cluster; matches the pipeline's own decontamination ceiling")
    p.add_argument("--coverage", type=float, default=0.50)
    p.add_argument("--train-frac", type=float, default=0.7, help="target sequence-count fraction, subject to whole-cluster assignment")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--min-test", type=int, default=5, help="skip a pair if the cluster-aware split leaves fewer test seqs than this")
    # mirrors floor_ppl_mgnify.py / configs/attack/full_ft.yaml / configs/metrics/thresholds.yaml
    p.add_argument("--max-train-seqs", type=int, default=8192)
    p.add_argument("--max-epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=1.0e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mlm-probability", type=float, default=0.15)
    p.add_argument("--tau-rho", type=float, default=0.80)
    p.add_argument("--max-eval-seqs", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--masks-per-pass", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_ppl_naive(naive_csv: Path) -> dict:
    import pandas as pd

    df = pd.read_csv(naive_csv)
    f_rows = df[df["role"] == "F"]
    return dict(zip(f_rows["pair"], f_rows["ppl_naive"]))


def all_vs_self_edges(fasta_path: Path, coverage: float, workdir: Path, threads: int = 8):
    """mmseqs easy-search fasta vs itself, keeping BOTH query and target ids (unlike
    src.data.leakage_filter.mmseqs_search, which only needs/keeps the query id)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    res = workdir / "result.m8"
    tmp = workdir / "tmp"
    cmd = [
        "mmseqs", "easy-search", str(fasta_path), str(fasta_path), str(res), str(tmp),
        "--min-seq-id", "0.0", "-c", str(coverage), "--cov-mode", "0",
        "--format-output", "query,target,fident", "--threads", str(threads), "-v", "1",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    edges = []
    with open(res) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            q, t, fident = parts[0], parts[1], float(parts[2])
            if q != t:
                edges.append((q, t, fident))
    return edges


def build_clusters(record_ids, edges, threshold: float):
    """Single-linkage clustering (union-find) over edges with fident >= threshold."""
    parent = {rid: rid for rid in record_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for q, t, fident in edges:
        if fident >= threshold and q in parent and t in parent:
            union(q, t)

    clusters = {}
    for rid in record_ids:
        clusters.setdefault(find(rid), []).append(rid)
    return list(clusters.values())


def assign_clusters(clusters, train_frac: float, seed: int):
    """Greedily assign whole clusters to train/test aiming for train_frac by sequence
    count (largest-first bin-packing after a seeded shuffle for tie-breaking)."""
    rng = random.Random(seed)
    clusters = list(clusters)
    rng.shuffle(clusters)
    clusters.sort(key=len, reverse=True)
    total = sum(len(c) for c in clusters)
    target_train = round(total * train_frac)
    train_ids, test_ids = set(), set()
    train_count = 0
    for c in clusters:
        if train_count < target_train:
            train_ids.update(c)
            train_count += len(c)
        else:
            test_ids.update(c)
    return train_ids, test_ids


def main():
    args = parse_args()
    import pandas as pd

    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path, sep="\t")
    manifest_dir = manifest_path.parent
    f_rows = manifest[manifest["role"] == "F"]
    f_rows = f_rows[f_rows["pair_idx"].apply(lambda i: f"pair{int(i)}" in args.pairs)]

    ppl_naive_by_pair = load_ppl_naive(Path(args.naive_csv))

    tok = load_tokenizer(args.model)
    attack_cfg = {
        "mode": "full_ft", "lr": args.lr, "batch_size": args.batch_size,
        "mlm_probability": args.mlm_probability, "max_epochs": args.max_epochs,
        "weight_decay": 0.0, "patience": args.patience,
    }

    rows = []
    for _, r in f_rows.iterrows():
        pair = f"pair{int(r['pair_idx'])}"
        if pair not in ppl_naive_by_pair:
            print(f"[{pair}] skipped: no PPL_naive row in {args.naive_csv}")
            continue
        ppl_naive = float(ppl_naive_by_pair[pair])

        records = read_fasta(manifest_dir / str(r["fasta"]))
        record_ids = [x.id for x in records]

        pair_split_dir = Path(args.split_dir) / pair
        ensure_dir(pair_split_dir)
        all_fa = pair_split_dir / "all.fasta"
        write_fasta(records, str(all_fa))
        edges = all_vs_self_edges(all_fa, args.coverage, pair_split_dir / "search")
        clusters = build_clusters(record_ids, edges, args.identity_threshold)
        train_ids, test_ids = assign_clusters(clusters, args.train_frac, args.split_seed)

        by_id = {x.id: x for x in records}
        train_recs = [by_id[i] for i in record_ids if i in train_ids]
        test_recs = [by_id[i] for i in record_ids if i in test_ids]
        write_fasta(train_recs, str(pair_split_dir / "train.fasta"))
        write_fasta(test_recs, str(pair_split_dir / "test.fasta"))

        print(f"[{pair}] {r['mgnifam_id']}: {len(clusters)} clusters (id>={args.identity_threshold}) "
              f"of {len(records)} seed seqs -> train={len(train_recs)} test={len(test_recs)}")

        if len(test_recs) < args.min_test:
            print(f"  SKIP: only {len(test_recs)} test seqs after cluster-aware split "
                  f"(< --min-test {args.min_test}); too few to trust a PPL estimate")
            continue

        train_seqs = [x.seq for x in train_recs][: args.max_train_seqs]
        test_seqs = [x.seq for x in test_recs]
        print(f"  full-FT on {len(train_seqs)} train / {len(test_seqs)} test seqs "
              f"(max_epochs={args.max_epochs}) ...")

        results_dir = str(Path(args.out).parent)
        tracker = JSONLTracker(run_name=f"{pair}_clustered", results_dir=results_dir)
        res = run_attack(
            args.model, tok, train_seqs, test_seqs, attack_cfg, args.device,
            seed=args.seed, max_length=args.max_length, background_seqs=None,
            ppl_batch_size=args.eval_batch_size, eval_masks_per_pass=args.masks_per_pass,
            max_eval_seqs=args.max_eval_seqs, eval_every=args.eval_every, tracker=tracker,
        )
        tracker.finish()
        save_json(res["history"], Path(results_dir) / "history" / f"{pair}_clustered.json")
        ppl_floor = res["best_target_ppl"]
        tau = acquisition_threshold(ppl_naive, ppl_floor, args.tau_rho)
        rel_headroom = (ppl_naive - ppl_floor) / ppl_naive if ppl_naive else 0.0
        headroom_ok = rel_headroom > 0.05 and ppl_naive > tau

        print(f"  -> PPL_naive={ppl_naive:.4f} PPL_floor={ppl_floor:.4f} "
              f"rel_headroom={rel_headroom:.3f} tau={tau:.4f} headroom_ok={headroom_ok} "
              f"(best_epoch={res['best_epoch']}/{res['epochs_run']})")

        rows.append({
            "pair": pair, "mgnifam_id": r["mgnifam_id"], "n_seed_total": len(records),
            "n_clusters": len(clusters), "n_train_used": len(train_seqs), "n_test": len(test_seqs),
            "ppl_naive": ppl_naive, "ppl_floor": ppl_floor,
            "rel_headroom": rel_headroom, "tau": tau, "headroom_ok": headroom_ok,
            "epochs_run": res["epochs_run"], "best_epoch": res["best_epoch"],
        })

        # Flush after every pair so a timeout doesn't lose already-finished pairs.
        ensure_dir(Path(args.out).parent)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"  (checkpointed {len(rows)} pairs to {args.out})")

    if not rows:
        raise SystemExit("No pairs produced a usable cluster-aware split; nothing to report.")

    df = pd.DataFrame(rows).sort_values("rel_headroom", ascending=False)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
