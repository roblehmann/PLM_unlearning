#!/usr/bin/env python3
"""T2 — per-sequence pseudo-perplexity for both corpora (SLURM array, cost-balanced).

Scores every sequence in t0's work table with the project's exact eval protocol
(`pseudo_perplexity`, batch 64, max_length 1024, masks_per_pass 8) — the same settings
behind tau, the headroom gate and every published PPL number here, so these values are
directly comparable to them.

Two properties make this phase safe to run as a 24-way array:

  * shards are balanced on PREDICTED COST (L * ceil(L/8)), not on index count. PPL cost
    grows ~L^2, so a contiguous split would hand one task the long tail and time it out
    while the rest idle;
  * it is RESUMABLE. Rows already present in this shard's TSV are skipped, and results are
    flushed every --flush-every sequences, so a preempted or timed-out task is fixed by
    resubmitting it rather than by redoing its work.

  python experiments/6_mgnify_reps_vs_retain/t2_ppl.py --shard 0 --n-shards 24 --device cuda
  sbatch --array=0-23 experiments/6_mgnify_reps_vs_retain/t2_ppl.sbatch
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np

import common as C


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-cpu-fallback", action="store_true")
    ap.add_argument("--batch-size", type=int, default=C.EVAL_BATCH)
    ap.add_argument("--masks-per-pass", type=int, default=C.MASKS_PER_PASS)
    ap.add_argument("--max-length", type=int, default=C.MAX_LENGTH)
    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--force", action="store_true", help="ignore existing rows and rescore")
    C.common_args(ap)
    return ap.parse_args()


def load_work(shard: int, n_shards: int):
    """This shard's rows from t0's work table, with the n-shards consistency assert."""
    manifest_path = C.RESULTS / "corpus_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"FAIL: {manifest_path} missing — run t0 first.")
    manifest = C.load_json(manifest_path)
    if int(manifest.get("n_shards", -1)) != int(n_shards):
        raise SystemExit(
            f"FAIL: t0 partitioned the work into {manifest.get('n_shards')} shards but this task "
            f"was given --n-shards {n_shards}. A mismatch would silently re-partition the work "
            "and leave sequences scored twice or not at all. Re-run t0 or fix the array width.")

    rows = []
    with open(C.RESULTS / "work_table.tsv") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if int(r["shard"]) == int(shard):
                rows.append({"row_id": int(r["row_id"]), "corpus": r["corpus"], "id": r["id"],
                             "length": int(r["length"])})
    return rows, manifest


def sequences_for(rows):
    """Resolve the shard's ids to sequences with one pass over each FASTA."""
    from src.utils.io import iter_fasta

    want = {}
    for r in rows:
        want.setdefault(r["corpus"], set()).add(r["id"])
    seqs = {}
    for corpus, path in (("reps", C.REPS_FASTA), ("retain", C.RETAIN_FASTA)):
        ids = want.get(corpus)
        if not ids:
            continue
        for rec in iter_fasta(str(path)):
            if rec.id in ids:
                seqs[(corpus, rec.id)] = rec.seq
        missing = ids - {k[1] for k in seqs if k[0] == corpus}
        if missing:
            raise SystemExit(f"FAIL: {len(missing)} ids from the work table are not in {path} "
                             f"(first: {sorted(missing)[:3]})")
    return seqs


def main() -> int:
    args = parse_args()
    C.apply_results_override(args.results)
    C.banner(f"T2 — pseudo-perplexity, shard {args.shard}/{args.n_shards}")
    device = C.device_arg(args.device, args.allow_cpu_fallback)

    rows, _ = load_work(args.shard, args.n_shards)
    out_dir = C.results_dir("ppl")
    stem = out_dir / f"shard_{args.shard:03d}_of_{args.n_shards:03d}"
    tsv, done_json = stem.with_suffix(".tsv"), Path(str(stem) + ".done.json")

    # ------------------------------------------------------------- resume
    have = set()
    if tsv.exists() and not args.force:
        with open(tsv) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                have.add(int(r["row_id"]))
    todo = [r for r in rows if r["row_id"] not in have]
    print(f"  {len(rows):,} rows in this shard, {len(have):,} already scored, "
          f"{len(todo):,} to do", flush=True)
    if not todo:
        print("  nothing to do — shard already complete", flush=True)
        if not done_json.exists():
            C.save_json({"n_rows": len(rows), "resumed_complete": True,
                         "protocol": C.protocol_block()}, done_json)
        return 0

    seqs = sequences_for(todo)

    from src.eval.pseudo_perplexity import pseudo_perplexity
    from src.models.load import load_mlm, load_tokenizer

    tok = load_tokenizer(C.MODEL_ID)
    model = load_mlm(C.MODEL_ID, device=device)
    model.eval()

    # Group by length so every call uses one batch size. `_score_one` batches equal-length
    # masked variants of a SINGLE sequence — no padding, no cross-sequence interaction — so
    # the batch size is numerically neutral and only caps peak activation memory. It is a
    # function of length alone, hence identical for reps and retain at the same length.
    todo.sort(key=lambda r: r["length"])
    write_header = not tsv.exists() or args.force
    mode = "w" if write_header else "a"
    fh = open(tsv, mode, newline="")
    w = csv.writer(fh, delimiter="\t")
    if write_header:
        w.writerow(["row_id", "corpus", "id", "length", "n_positions", "nll_sum", "ppl", "log_ppl"])

    t_start = time.time()
    n_done = 0
    tok_forwards = 0
    batch_sizes = {}
    chunk = 32                                    # sequences per pseudo_perplexity call
    try:
        for start in range(0, len(todo), chunk):
            part = todo[start:start + chunk]
            lmax = max(C.scored_positions(r["length"], args.max_length) for r in part)
            bs = max(1, min(args.batch_size, C.TOKEN_BUDGET // max(lmax, 1)))
            batch_sizes[str(lmax)] = int(bs)
            res = pseudo_perplexity(
                model, tok, [seqs[(r["corpus"], r["id"])] for r in part],
                device=device, batch_size=bs, max_length=args.max_length,
                masks_per_pass=args.masks_per_pass)
            for r, nll, pos in zip(part, res.per_seq_nll, res.per_seq_positions):
                log_ppl = nll / pos if pos else float("nan")
                w.writerow([r["row_id"], r["corpus"], r["id"], r["length"], pos,
                            f"{nll:.6f}", f"{math.exp(log_ppl) if pos else float('nan'):.6f}",
                            f"{log_ppl:.8f}"])
                tok_forwards += C.predicted_cost(r["length"], args.masks_per_pass, args.max_length)
            n_done += len(part)
            if n_done % max(args.flush_every, chunk) < chunk:
                fh.flush()
                os.fsync(fh.fileno())
                el = time.time() - t_start
                print(f"  {n_done:,}/{len(todo):,}  {n_done / el:.1f} seq/s  "
                      f"{tok_forwards / el:,.0f} tok/s  eta "
                      f"{(len(todo) - n_done) / max(n_done / el, 1e-9) / 60:.1f} min", flush=True)
    finally:
        fh.flush()
        os.fsync(fh.fileno())
        fh.close()

    dt = time.time() - t_start
    C.save_json({
        "shard": int(args.shard), "n_shards": int(args.n_shards),
        "n_rows": int(len(rows)), "n_scored_here": int(n_done),
        "seconds": dt, "seqs_per_sec": n_done / dt if dt else None,
        "tokens_per_sec": tok_forwards / dt if dt else None,
        "token_forwards": int(tok_forwards),
        "batch_size_by_max_len": batch_sizes,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "node": os.environ.get("SLURMD_NODENAME", ""),
        "protocol": C.protocol_block(eval_batch=args.batch_size,
                                     masks_per_pass=args.masks_per_pass),
    }, done_json)
    print(f"\nT2 shard {args.shard} OK — {n_done:,} sequences in {dt / 60:.1f} min "
          f"({tok_forwards / dt:,.0f} tok/s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
