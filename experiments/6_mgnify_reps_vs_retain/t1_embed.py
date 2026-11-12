#!/usr/bin/env python3
"""T1 — mean-pooled ESM-2 embeddings for the reps corpus, plus the retain-reuse gate.

The retain corpus was already embedded under this exact model, max_length and pooling path
by scripts/12_embed_corpora.py (194,064 x 480, float16, cap=-1). Recomputing it would cost
~5 GPU-minutes, so we adopt the cache — but only behind a gate. A silent change in
transformers, in the autocast dtype, or in the pooling mask between then and now would put
the two corpora in two different numerical regimes, and the entire UMAP separation this
experiment reports could be an artifact of that rather than of biology. So `--verify-retain`
re-embeds a deterministic sample with today's code and demands median cosine >= 0.999
against the cached rows.

  python experiments/6_mgnify_reps_vs_retain/t1_embed.py --device cuda
  python experiments/6_mgnify_reps_vs_retain/t1_embed.py --device cuda --reembed-retain
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import common as C


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-cpu-fallback", action="store_true")
    ap.add_argument("--batch-size", type=int, default=C.EMBED_BATCH)
    ap.add_argument("--max-length", type=int, default=C.MAX_LENGTH)
    ap.add_argument("--limit", type=int, default=0, help="smoke only: cap the reps corpus")
    ap.add_argument("--verify-retain", type=int, default=256,
                    help="re-embed this many retain sequences and compare to the cache (0 skips)")
    ap.add_argument("--verify-min-cosine", type=float, default=0.999)
    ap.add_argument("--reembed-retain", action="store_true",
                    help="ignore the cache and embed all of retain here (~5 GPU-min)")
    ap.add_argument("--force", action="store_true", help="re-embed reps even if cached")
    C.common_args(ap)
    return ap.parse_args()


def _load_model(device: str):
    from src.models.load import load_mlm, load_tokenizer

    tok = load_tokenizer(C.MODEL_ID)
    model = load_mlm(C.MODEL_ID, device=device)
    model.eval()
    return model, tok


def _write(feats: np.ndarray, ids, stem: Path) -> None:
    np.save(stem.with_suffix(".npy"), feats.astype(np.float16))
    with open(stem.with_suffix(".tsv"), "w") as fh:
        fh.write("id\n")
        for i in ids:
            fh.write(f"{i}\n")
    print(f"  wrote {stem.with_suffix('.npy')}  {feats.shape} float16", flush=True)


def main() -> int:
    args = parse_args()
    C.apply_results_override(args.results)
    C.banner("T1 — embeddings (reps) + cached-retain verification")
    device = C.device_arg(args.device, args.allow_cpu_fallback)
    print(f"  device {device}", flush=True)

    from src.utils.io import read_fasta

    emb_dir = C.results_dir("embeddings")
    reps_stem = emb_dir / "reps"

    reps_records, _ = C.load_reps(C.REPS_FASTA, limit=args.limit or None)
    reps_seqs = [r.seq for r in reps_records]
    reps_ids = [r.id for r in reps_records]
    reps_tokens = int(sum(min(len(s), args.max_length - 2) for s in reps_seqs))

    model = tok = None
    info = {"protocol": C.protocol_block(embed_batch=args.batch_size)}

    # ------------------------------------------------------------------ reps
    if reps_stem.with_suffix(".npy").exists() and not args.force:
        cached = np.load(reps_stem.with_suffix(".npy"), mmap_mode="r")
        print(f"  reps embeddings already cached {cached.shape} — pass --force to redo")
        info["reps"] = {"n": int(cached.shape[0]), "path": str(reps_stem.with_suffix(".npy")),
                        "reused_from_this_experiment": True}
    else:
        model, tok = _load_model(device)
        t0 = time.time()
        feats = C.embed_length_sorted(model, tok, reps_seqs, device, args.batch_size,
                                      args.max_length)
        dt = time.time() - t0
        _write(feats, reps_ids, reps_stem)
        info["reps"] = {
            "n": int(feats.shape[0]), "hidden_dim": int(feats.shape[1]),
            "path": str(reps_stem.with_suffix(".npy")),
            "seconds": dt, "seqs_per_sec": len(reps_seqs) / dt,
            "tokens_per_sec": reps_tokens / dt,
        }
        print(f"  reps: {len(reps_seqs):,} seqs in {dt:.1f}s = "
              f"{len(reps_seqs) / dt:.1f} seq/s, {reps_tokens / dt:,.0f} tok/s", flush=True)

    # ---------------------------------------------------------------- retain
    retain_stem = emb_dir / "retain"
    if args.reembed_retain:
        retain_records = read_fasta(str(C.RETAIN_FASTA))
        if model is None:
            model, tok = _load_model(device)
        t0 = time.time()
        feats = C.embed_length_sorted(model, tok, [r.seq for r in retain_records], device,
                                      args.batch_size, args.max_length)
        dt = time.time() - t0
        _write(feats, [r.id for r in retain_records], retain_stem)
        info["retain"] = {"n": int(feats.shape[0]), "path": str(retain_stem.with_suffix(".npy")),
                          "reused": False, "seconds": dt,
                          "seqs_per_sec": len(retain_records) / dt}
    else:
        info["retain"] = {"n": int(np.load(C.CACHED_RETAIN_NPY, mmap_mode="r").shape[0]),
                          "path": str(C.CACHED_RETAIN_NPY), "reused": True,
                          "ids_verified_against_fasta": True,
                          "note": "verified id-for-id and in order by t0"}

        # ------------------------------------------------ the reuse gate
        if args.verify_retain > 0:
            cached = np.load(C.CACHED_RETAIN_NPY, mmap_mode="r")
            retain_records = read_fasta(str(C.RETAIN_FASTA))
            rng = np.random.default_rng(args.seed)
            idx = np.sort(rng.choice(len(retain_records),
                                     size=min(args.verify_retain, len(retain_records)),
                                     replace=False))
            if model is None:
                model, tok = _load_model(device)
            fresh = C.embed_length_sorted(model, tok, [retain_records[i].seq for i in idx],
                                          device, args.batch_size, args.max_length)
            old = np.asarray(cached[idx], dtype=np.float32)
            num = (fresh * old).sum(axis=1)
            den = np.linalg.norm(fresh, axis=1) * np.linalg.norm(old, axis=1)
            cos = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
            med = float(np.nanmedian(cos))
            verify = {"n": int(idx.size), "median_cosine": med,
                      "min_cosine": float(np.nanmin(cos)), "mean_cosine": float(np.nanmean(cos)),
                      "threshold": float(args.verify_min_cosine)}
            info["retain"]["verify_retain"] = verify
            print(f"  retain cache check on {idx.size} sequences: median cos {med:.6f} "
                  f"(min {verify['min_cosine']:.6f})", flush=True)
            if not (med >= args.verify_min_cosine):
                raise SystemExit(
                    f"FAIL: cached retain embeddings no longer reproduce (median cosine {med:.6f} "
                    f"< {args.verify_min_cosine}).\n"
                    "  The numerical regime has changed since the cache was written, so the two "
                    "corpora would not be comparable.\n"
                    "  Re-run this phase with --reembed-retain (~5 GPU-min) and let t3 use the "
                    "freshly written results/embeddings/retain.npy.")
        else:
            print("  [skip] retain cache verification (--verify-retain 0)")

    C.save_json(info, emb_dir / "manifest.json")
    print("\nT1 OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
