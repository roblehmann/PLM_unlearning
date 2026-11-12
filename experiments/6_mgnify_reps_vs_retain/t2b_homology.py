#!/usr/bin/env python3
"""T2b — split the MGnify representatives by whether they have a UniRef50 homolog.

"MGnify is modelled worse" is a weaker claim than it looks: it could just mean the corpora
differ in composition or length. The sharper claim is that the part of MGnify with NO
detectable homolog in the pretraining proxy is modelled worse — novelty, not domain shift.
This phase produces that split with an mmseqs2 search of the 35,459 reps against the
194,064 retain sequences, and t4 contrasts log-PPL between the two groups.

It also generalises, to the whole corpus, the single-family leakage check that found
0/30,000 retain sequences homologous to F.

CPU only; depends on t0 alone, so it runs alongside t1/t2.

  python experiments/6_mgnify_reps_vs_retain/t2b_homology.py --threads 16
  python experiments/6_mgnify_reps_vs_retain/t2b_homology.py -s 5.7   # faster fallback
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import common as C

FIELDS = "query,target,fident,alnlen,evalue,bits,qcov,tcov"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default=str(C.REPS_FASTA))
    ap.add_argument("--target", default=str(C.RETAIN_FASTA))
    ap.add_argument("--threads", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    ap.add_argument("-s", "--sensitivity", type=float, default=7.5,
                    help="mmseqs -s; 5.7 is the ~3-4x faster fallback")
    ap.add_argument("--evalue", type=float, default=1e-3)
    ap.add_argument("--coverage", type=float, default=0.5)
    ap.add_argument("--max-seqs", type=int, default=100)
    ap.add_argument("--bits-threshold", type=float, default=50.0,
                    help="headline homology call: E <= --evalue AND bits >= this")
    ap.add_argument("--bits-strict", type=float, default=100.0, help="sensitivity tier")
    ap.add_argument("--tmp", default=None, help="mmseqs tmp dir (default: node-local scratch)")
    ap.add_argument("--reuse-hits", action="store_true",
                    help="skip the search and re-reduce an existing hits.m8")
    C.common_args(ap)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    C.apply_results_override(args.results)
    C.banner("T2b — mmseqs homology split (reps vs retain)")
    out_dir = C.results_dir("homology")
    hits = out_dir / "hits.m8"

    if shutil.which("mmseqs") is None:
        raise SystemExit("FAIL: mmseqs not on PATH — slurm/env_setup.sh loads mmseqs2/14.7; "
                         "source it before running this phase.")

    t0 = time.time()
    if args.reuse_hits and hits.exists():
        print(f"  reusing {hits} (--reuse-hits)", flush=True)
        elapsed = 0.0
    else:
        tmp_root = args.tmp or os.environ.get("TMPDIR") or tempfile.gettempdir()
        with tempfile.TemporaryDirectory(dir=tmp_root, prefix="mgreps_mmseqs_") as tmp:
            cmd = [
                "mmseqs", "easy-search", args.query, args.target, str(hits), tmp,
                "-s", str(args.sensitivity), "-e", str(args.evalue),
                "-c", str(args.coverage), "--cov-mode", "0",
                "--max-seqs", str(args.max_seqs),
                "--format-output", FIELDS,
                "--threads", str(args.threads), "-v", "1",
            ]
            print("  " + " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        print(f"  search finished in {elapsed / 60:.1f} min", flush=True)

    # ------------------------------------------------------------- reduce
    # Both id sets were verified unique by t0, so the real ids can be kept and the join to
    # the per-sequence table is direct — no synthetic-id rewrite is needed here.
    best: dict = {}
    n_lines = 0
    with open(hits) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            n_lines += 1
            q, t, fident, alnlen, evalue, bits, qcov, tcov = parts[:8]
            bits_f = float(bits)
            row = best.get(q)
            if row is None or bits_f > row["best_bits"]:
                best[q] = {"id": q, "best_target": t, "best_fident": float(fident),
                           "best_alnlen": int(float(alnlen)), "best_evalue": float(evalue),
                           "best_bits": bits_f, "best_qcov": float(qcov),
                           "best_tcov": float(tcov), "n_hits": 0}
            best[q]["n_hits"] = best[q].get("n_hits", 0) + 1

    # Reduce over the QUERY file actually searched, not the constant — so a smoke run that
    # searched a slice reports on that slice rather than on 35k unsearched sequences.
    reps_records, _ = C.load_reps(args.query)
    rows = []
    n_hom = n_hom_strict = 0
    for rec in reps_records:
        b = best.get(rec.id)
        if b is None:
            rows.append({"id": rec.id, "n_hits": 0, "best_target": "", "best_evalue": "",
                         "best_fident": "", "best_bits": "", "best_qcov": "",
                         "hom_class": "novel", "hom_class_strict": "novel"})
            continue
        hom = (b["best_evalue"] <= args.evalue) and (b["best_bits"] >= args.bits_threshold)
        hom_strict = (b["best_evalue"] <= args.evalue) and (b["best_bits"] >= args.bits_strict)
        n_hom += int(hom)
        n_hom_strict += int(hom_strict)
        rows.append({"id": rec.id, "n_hits": b["n_hits"], "best_target": b["best_target"],
                     "best_evalue": b["best_evalue"], "best_fident": b["best_fident"],
                     "best_bits": b["best_bits"], "best_qcov": b["best_qcov"],
                     "hom_class": "homologous" if hom else "novel",
                     "hom_class_strict": "homologous" if hom_strict else "novel"})

    tsv = out_dir / "reps_homology.tsv"
    with open(tsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {tsv}", flush=True)

    n = len(rows)
    summary = {
        "protocol": C.protocol_block(),
        "enabled": True,
        "params": {"sensitivity": args.sensitivity, "evalue": args.evalue,
                   "coverage": args.coverage, "cov_mode": 0, "max_seqs": args.max_seqs,
                   "bits_threshold": args.bits_threshold, "bits_strict": args.bits_strict,
                   "threads": args.threads, "format": FIELDS},
        "n_reps": n,
        "n_hit_lines": n_lines,
        "n_reps_with_any_hit": len(best),
        "n_reps_homologous": n_hom,
        "n_reps_novel": n - n_hom,
        "frac_novel": (n - n_hom) / n if n else None,
        "n_reps_homologous_strict": n_hom_strict,
        "frac_novel_strict": (n - n_hom_strict) / n if n else None,
        "median_best_fident": (
            float(sorted(b["best_fident"] for b in best.values())[len(best) // 2])
            if best else None),
        "runtime_s": elapsed,
    }
    C.save_json(summary, out_dir / "summary.json")
    print(f"\n  {n_hom:,}/{n:,} reps have a UniRef50 homolog "
          f"(E<={args.evalue}, bits>={args.bits_threshold}); "
          f"{summary['frac_novel']:.1%} are novel", flush=True)
    print("T2b OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
