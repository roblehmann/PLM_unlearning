#!/usr/bin/env python3
"""T0 — corpus preparation, validation gate, and the cost-balanced work plan.

Everything expensive downstream depends on assumptions that are cheap to check here, on a
CPU node, in about two minutes:

  * the reps and retain FASTAs exist, parse, and have unique ids (the mmseqs join in t2b
    and every merge in t4 assume uniqueness);
  * the CACHED retain embeddings really do correspond, id for id and in order, to
    retain_train.fasta. The whole UMAP half of this experiment reuses that array; if the
    order were off the figure would be confidently wrong;
  * the HF snapshot is in the local cache, so an offline miss fails in seconds on CPU
    rather than after a GPU has been allocated;
  * both anchor families (F, F') are present and flagged.

It then writes the work plan the GPU phases consume: per-sequence metadata, the
length-matched retain subset (built against the FULL retain pool — matching from a 35k
subsample fails in the 100-150 aa bin), and a cost-balanced shard assignment.

Exits non-zero on any failure, so `afterok` stops the DAG before any GPU time is spent.

  python experiments/6_mgnify_reps_vs_retain/t0_corpus.py
  python experiments/6_mgnify_reps_vs_retain/t0_corpus.py --n-shards 24
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

import numpy as np

import common as C


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps-fasta", default=str(C.REPS_FASTA))
    ap.add_argument("--retain-fasta", default=str(C.RETAIN_FASTA))
    ap.add_argument("--n-shards", type=int, default=24,
                    help="t2 array width; shards are balanced on predicted token-forwards.")
    ap.add_argument("--max-reps", type=int, default=0, help="smoke only: cap the reps corpus")
    ap.add_argument("--max-retain", type=int, default=0, help="smoke only: cap the retain corpus")
    ap.add_argument("--match-bin-width", type=int, default=C.MATCH_BIN_WIDTH)
    ap.add_argument("--match-max-delta", type=int, default=C.MATCH_MAX_DELTA)
    ap.add_argument("--skip-cache-check", action="store_true",
                    help="skip the cached-retain-embedding verification (smoke on a subset)")
    C.common_args(ap)
    return ap.parse_args()


def _fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def verify_cached_retain(retain_ids) -> dict:
    """The single assumption the embedding reuse rests on — checked id for id, in order."""
    if not C.CACHED_RETAIN_NPY.exists():
        _fail(f"cached retain embeddings missing: {C.CACHED_RETAIN_NPY}\n"
              "  regenerate with `sbatch slurm/12_embed_corpora.sbatch --anchors retain`, "
              "or run t1 with --reembed-retain (~5 GPU-min).")
    arr = np.load(C.CACHED_RETAIN_NPY, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != C.HIDDEN_DIM:
        _fail(f"cached retain embeddings have shape {arr.shape}, expected (n, {C.HIDDEN_DIM})")
    if arr.shape[0] != len(retain_ids):
        _fail(f"cached retain embeddings have {arr.shape[0]} rows but the FASTA has "
              f"{len(retain_ids)} sequences")

    with open(C.CACHED_RETAIN_TSV) as fh:
        rows = [ln.strip() for ln in fh if ln.strip()]
    if not rows or rows[0] != "id":
        _fail(f"{C.CACHED_RETAIN_TSV}: expected an 'id' header line")
    cached_ids = rows[1:]
    if len(cached_ids) != len(retain_ids):
        _fail(f"cached id list has {len(cached_ids)} entries, FASTA has {len(retain_ids)}")
    mismatch = [i for i, (a, b) in enumerate(zip(cached_ids, retain_ids)) if a != b]
    if mismatch:
        i = mismatch[0]
        _fail(f"cached retain ids do not match the FASTA in order: {len(mismatch)} rows differ, "
              f"first at row {i} ({cached_ids[i]!r} vs {retain_ids[i]!r})")

    manifest = {}
    if C.CACHED_RETAIN_MANIFEST.exists():
        manifest = C.load_json(C.CACHED_RETAIN_MANIFEST).get("shared/retain", {})
        if manifest.get("model") and manifest["model"] != C.MODEL_ID:
            _fail(f"cached retain embeddings were made with {manifest['model']}, "
                  f"this experiment uses {C.MODEL_ID}")
        if manifest.get("cap") not in (None, -1):
            _fail(f"cached retain embeddings were capped (cap={manifest['cap']}) — not the full corpus")
    return {
        "path": str(C.CACHED_RETAIN_NPY),
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "dtype": str(arr.dtype),
        "ids_verified_against_fasta": True,
        "source_manifest": manifest,
    }


def length_stats(lengths: np.ndarray) -> dict:
    q = lambda p: float(np.quantile(lengths, p))  # noqa: E731
    scored = np.minimum(lengths, C.SCORED_MAX)
    return {
        "n": int(lengths.size),
        "len_mean": float(lengths.mean()),
        "len_median": q(0.5), "len_p25": q(0.25), "len_p75": q(0.75),
        "len_p90": q(0.90), "len_p95": q(0.95),
        "len_min": int(lengths.min()), "len_max": int(lengths.max()),
        "n_truncated": int((lengths > C.SCORED_MAX).sum()),
        "frac_truncated": float((lengths > C.SCORED_MAX).mean()),
        "n_positions_scored": int(scored.sum()),
    }


def main() -> int:
    args = parse_args()
    C.apply_results_override(args.results)
    t_start = time.time()
    C.banner("T0 — corpus prep, validation gate, work plan")

    # ---------------------------------------------------------------- load
    from src.utils.io import read_fasta

    for p in (args.reps_fasta, args.retain_fasta):
        if not Path(p).exists() or Path(p).stat().st_size == 0:
            _fail(f"missing or empty FASTA: {p}")

    reps_records, reps_meta = C.load_reps(args.reps_fasta,
                                          limit=args.max_reps or None)
    retain_records = read_fasta(args.retain_fasta)
    retain_ids_full = [r.id for r in retain_records]
    if args.max_retain:
        retain_records = retain_records[:args.max_retain]
    print(f"  reps   {len(reps_records):>7,} sequences from {args.reps_fasta}")
    print(f"  retain {len(retain_records):>7,} sequences from {args.retain_fasta}", flush=True)

    # ---------------------------------------------------------------- gate
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}", flush=True)
        if not ok:
            _fail(f"{name}: {detail}")

    reps_ids = [r.id for r in reps_records]
    check("reps ids unique", len(set(reps_ids)) == len(reps_ids),
          f"{len(reps_ids) - len(set(reps_ids))} duplicates")
    check("retain ids unique", len(set(retain_ids_full)) == len(retain_ids_full),
          f"{len(retain_ids_full) - len(set(retain_ids_full))} duplicates")

    bad_len = [m.id for m in reps_meta
               if m.declared_len is not None and m.declared_len != m.length]
    check("reps len= header field matches the parsed sequence length", not bad_len,
          f"{len(bad_len)} mismatches, first {bad_len[:3]}")

    missing_anchors = [a for a in C.ANCHORS if a not in set(reps_ids)]
    if args.max_reps:
        print(f"  [skip] anchor presence (--max-reps {args.max_reps} truncates the corpus)")
    else:
        check("both anchor families present in the reps corpus", not missing_anchors,
              f"missing {missing_anchors}")

    snap = C.hf_snapshot_dir()
    check(f"HF snapshot for {C.MODEL_ID} is in the local cache", snap is not None,
          str(snap) if snap else "not found — the GPU phases would hang on an offline hub lookup")

    if args.skip_cache_check:
        cache_info = {"skipped": True}
        print("  [skip] cached retain embedding verification (--skip-cache-check)")
    else:
        cache_info = verify_cached_retain(retain_ids_full)
        check("cached retain embeddings align with the retain FASTA, id for id and in order",
              True, f"{cache_info['shape'][0]:,} x {cache_info['shape'][1]} {cache_info['dtype']}")

    # Exact cross-corpus duplicates: a trivial-leak catch, free at this point.
    reps_md5 = {hashlib.md5(r.seq.encode()).hexdigest() for r in reps_records}
    retain_md5 = {hashlib.md5(r.seq.encode()).hexdigest() for r in retain_records}
    n_dup = len(reps_md5 & retain_md5)
    print(f"  exact duplicate sequences across corpora: {n_dup}", flush=True)

    # ---------------------------------------------------------------- metadata
    reps_len = np.array([m.length for m in reps_meta], dtype=np.int64)
    retain_len = np.array([len(r.seq) for r in retain_records], dtype=np.int64)

    reps_rows = []
    for m, rec in zip(reps_meta, reps_records):
        row = m.as_row()
        row["entropy"] = C.seq_entropy(rec.seq)
        row["frac_X"] = rec.seq.count("X") / max(1, len(rec.seq))
        row["n_scored"] = C.scored_positions(m.length)
        reps_rows.append(row)

    meta_dir = C.results_dir("meta")
    with open(meta_dir / "reps_meta.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(reps_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(reps_rows)
    print(f"  wrote {meta_dir / 'reps_meta.tsv'}", flush=True)

    with open(meta_dir / "retain_meta.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "length", "n_scored", "entropy", "frac_X"])
        for rec in retain_records:
            w.writerow([rec.id, len(rec.seq), C.scored_positions(len(rec.seq)),
                        f"{C.seq_entropy(rec.seq):.6f}", f"{rec.seq.count('X') / max(1, len(rec.seq)):.8f}"])
    print(f"  wrote {meta_dir / 'retain_meta.tsv'}", flush=True)

    # ---------------------------------------------------------------- length matching
    # Analysis-side only: every retain sequence is scored anyway, so matching costs no GPU
    # time and can draw on the full pool. That is what makes it feasible — retain has
    # >=3.89x the reps count in every 50-aa bin, which a 35k subsample would not.
    match_idx, match_diag = C.length_match(
        reps_len, retain_len, seed=args.seed,
        bin_width=args.match_bin_width, max_delta=args.match_max_delta)
    matched_mask = np.zeros(retain_len.size, dtype=bool)
    for i in match_idx:
        if i >= 0:
            matched_mask[i] = True
    if matched_mask.sum():
        from scipy.stats import ks_2samp

        ks = ks_2samp(retain_len[matched_mask], reps_len)
        match_diag["ks_stat_len_vs_reps"] = float(ks.statistic)
        match_diag["ks_p"] = float(ks.pvalue)
    print(f"  length matching: {match_diag['n_matched']:,} matched, "
          f"{match_diag['n_unmatched']:,} unmatched, mean |dL| "
          f"{match_diag['mean_abs_dlen']:.2f}, KS vs reps "
          f"{match_diag.get('ks_stat_len_vs_reps', float('nan')):.4f}", flush=True)

    # ---------------------------------------------------------------- work table
    all_ids = reps_ids + [r.id for r in retain_records]
    all_corpus = ["reps"] * len(reps_ids) + ["retain"] * len(retain_records)
    all_len = np.concatenate([reps_len, retain_len])
    in_matched = np.concatenate([np.zeros(reps_len.size, dtype=bool), matched_mask])

    shards = C.shard_assign(all_len, args.n_shards)
    costs = np.array([C.predicted_cost(int(L)) for L in all_len], dtype=np.int64)

    with open(C.RESULTS / "work_table.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["row_id", "corpus", "id", "length", "n_scored", "in_matched",
                    "shard", "predicted_cost"])
        for i, (cid, corpus, L) in enumerate(zip(all_ids, all_corpus, all_len)):
            w.writerow([i, corpus, cid, int(L), C.scored_positions(int(L)),
                        int(in_matched[i]), int(shards[i]), int(costs[i])])
    print(f"  wrote {C.RESULTS / 'work_table.tsv'} ({len(all_ids):,} rows)", flush=True)

    # ---------------------------------------------------------------- cost report
    shard_rows = C.shard_cost_table(all_len, shards, args.n_shards)
    total_tf = int(costs.sum())
    per_shard = np.array([r["token_forwards"] for r in shard_rows], dtype=np.float64)
    imbalance = float(per_shard.max() / per_shard.mean()) if per_shard.mean() > 0 else float("nan")
    print(f"\n  total work {total_tf:,} token-forwards over {args.n_shards} shards "
          f"(max/mean imbalance {imbalance:.3f})")
    for tps, label in ((202_000, "optimistic"), (120_000, "conservative")):
        print(f"    @{tps:,} tok/s ({label}): total {total_tf / tps / 3600:.2f} GPU-h, "
              f"worst shard {per_shard.max() / tps / 60:.1f} min")
    print("  -> if the smoke's measured tok/s is below ~60k, widen --n-shards before the "
          "real launch.", flush=True)

    # ---------------------------------------------------------------- manifest
    reps_comp = C.aa_composition(r.seq for r in reps_records)
    retain_comp = C.aa_composition(r.seq for r in retain_records)
    manifest = {
        "protocol": C.protocol_block(n_shards=int(args.n_shards)),
        "n_shards": int(args.n_shards),
        "corpora": {
            "reps": {**length_stats(reps_len),
                     "source_counts": {s: sum(1 for m in reps_meta if m.source == s)
                                       for s in sorted({m.source for m in reps_meta})},
                     "aa_entropy_mean": float(np.mean([r["entropy"] for r in reps_rows])),
                     "frac_X": float(np.mean([r["frac_X"] for r in reps_rows]))},
            "retain": {**length_stats(retain_len)},
        },
        "composition": {
            "aa_freq": {"reps": reps_comp, "retain": retain_comp},
            "js_divergence_reps_vs_retain": C.js_divergence(reps_comp, retain_comp),
        },
        "duplicates": {"exact_seq_overlap_reps_vs_retain": int(n_dup)},
        "length_matching": match_diag,
        "cached_retain_embeddings": cache_info,
        "work": {
            "n_rows": int(len(all_ids)),
            "total_token_forwards": total_tf,
            "shard_imbalance_max_over_mean": imbalance,
            "shards": shard_rows,
            "hours_at_202k_tok_s": total_tf / 202_000 / 3600,
            "hours_at_120k_tok_s": total_tf / 120_000 / 3600,
        },
        "anchors": {a: {"role": role, "present": a in set(reps_ids)}
                    for a, role in C.ANCHORS.items()},
        "checks": checks,
        "seconds": time.time() - t_start,
    }
    C.save_json(manifest, C.RESULTS / "corpus_manifest.json")
    print(f"\nT0 OK in {time.time() - t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
