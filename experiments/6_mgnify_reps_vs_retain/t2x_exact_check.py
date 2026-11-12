#!/usr/bin/env python3
"""T2x — is the masks_per_pass=8 speedup biased DIFFERENTLY for the two corpora?

The headline of this experiment is a *difference* between two corpora. The project scores
pseudo-perplexity with `masks_per_pass=8`, which masks 8 strided positions per forward
instead of one — an approximation whose bias depends on how much information neighbouring
residues carry about each other, i.e. on sequence structure. If that bias were larger for
metagenomic sequence than for UniRef50, the headline would be an artifact of the speedup
rather than a fact about ESM-2.

So: score a deterministic, length-stratified subset of each corpus at both mpp=1 (exact)
and mpp=8, and report the number that matters — bias(reps) - bias(retain), with a
bootstrap CI. Costs ~15 GPU-min and closes the objection.

  python experiments/6_mgnify_reps_vs_retain/t2x_exact_check.py --device cuda --n 300
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import common as C


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-cpu-fallback", action="store_true")
    ap.add_argument("--n", type=int, default=300, help="sequences per corpus")
    ap.add_argument("--batch-size", type=int, default=C.EVAL_BATCH)
    ap.add_argument("--max-length", type=int, default=C.MAX_LENGTH)
    ap.add_argument("--max-len-cap", type=int, default=400,
                    help="exclude longer sequences: exact scoring costs L^2 and the bias "
                         "question does not need the tail.")
    C.common_args(ap)
    return ap.parse_args()


def stratified_sample(lengths, n, seed, cap):
    """Deterministic, roughly length-stratified pick so both corpora span the same range."""
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths)
    eligible = np.flatnonzero(lengths <= cap)
    if eligible.size <= n:
        return np.sort(eligible)
    bins = np.array([C.length_bin(int(lengths[i])) for i in eligible])
    picks = []
    for b in np.unique(bins):
        pool = eligible[bins == b]
        take = max(1, int(round(n * pool.size / eligible.size)))
        picks.append(rng.choice(pool, size=min(take, pool.size), replace=False))
    out = np.unique(np.concatenate(picks))
    if out.size > n:
        out = np.sort(rng.choice(out, size=n, replace=False))
    return np.sort(out)


def main() -> int:
    args = parse_args()
    C.apply_results_override(args.results)
    C.banner("T2x — masks_per_pass fidelity (mpp=1 vs mpp=8)")
    device = C.device_arg(args.device, args.allow_cpu_fallback)

    from src.eval.pseudo_perplexity import pseudo_perplexity
    from src.models.load import load_mlm, load_tokenizer
    from src.utils.io import read_fasta

    corpora = {}
    reps_records, _ = C.load_reps(C.REPS_FASTA)
    corpora["reps"] = reps_records
    corpora["retain"] = read_fasta(str(C.RETAIN_FASTA))

    tok = load_tokenizer(C.MODEL_ID)
    model = load_mlm(C.MODEL_ID, device=device)
    model.eval()

    out = {"protocol": C.protocol_block(), "n_per_corpus": int(args.n),
           "masks_per_pass_exact": 1, "masks_per_pass_fast": C.MASKS_PER_PASS,
           "max_len_cap": int(args.max_len_cap), "bias": {}, "per_corpus": {}}
    bias_samples = {}

    for name, records in corpora.items():
        lengths = [len(r.seq) for r in records]
        idx = stratified_sample(lengths, args.n, args.seed, args.max_len_cap)
        seqs = [records[i].seq for i in idx]
        print(f"  {name}: {len(seqs)} sequences, lengths "
              f"{min(len(s) for s in seqs)}-{max(len(s) for s in seqs)}", flush=True)

        t0 = time.time()
        fast = pseudo_perplexity(model, tok, seqs, device=device, batch_size=args.batch_size,
                                 max_length=args.max_length, masks_per_pass=C.MASKS_PER_PASS)
        exact = pseudo_perplexity(model, tok, seqs, device=device, batch_size=args.batch_size,
                                  max_length=args.max_length, masks_per_pass=1)
        dt = time.time() - t0

        lf = np.array(fast.per_seq_nll) / np.array(fast.per_seq_positions)
        le = np.array(exact.per_seq_nll) / np.array(exact.per_seq_positions)
        bias = lf - le                      # positive => the speedup over-states difficulty
        bias_samples[name] = bias
        out["per_corpus"][name] = {
            "n": int(len(seqs)),
            "micro_ppl_mpp8": fast.ppl, "micro_ppl_mpp1": exact.ppl,
            "log_ppl_mean_mpp8": float(lf.mean()), "log_ppl_mean_mpp1": float(le.mean()),
            "seconds": dt,
        }
        out["bias"][name] = {
            "mean": float(bias.mean()),
            "ci95": list(C.bootstrap_ci(lambda x: float(np.mean(x)), bias,
                                        n_boot=C.N_BOOT, seed=args.seed)),
            "median": float(np.median(bias)),
        }
        from scipy.stats import pearsonr, spearmanr

        out["per_corpus"][name]["pearson_log_ppl"] = float(pearsonr(lf, le).statistic)
        out["per_corpus"][name]["spearman_log_ppl"] = float(spearmanr(lf, le).statistic)
        print(f"    bias {bias.mean():+.5f} nats/residue   "
              f"micro PPL {fast.ppl:.4f} (mpp8) vs {exact.ppl:.4f} (mpp1)", flush=True)

    # The number that matters: does the approximation treat the two corpora differently?
    br, bt = bias_samples["reps"], bias_samples["retain"]
    rng = np.random.default_rng(args.seed)
    diffs = np.array([br[rng.integers(0, br.size, br.size)].mean()
                      - bt[rng.integers(0, bt.size, bt.size)].mean() for _ in range(C.N_BOOT)])
    out["bias_difference_reps_minus_retain"] = float(br.mean() - bt.mean())
    out["bias_difference_ci95"] = [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
    out["verdict_ci_excludes_zero"] = bool(
        out["bias_difference_ci95"][0] > 0 or out["bias_difference_ci95"][1] < 0)
    out["interpretation"] = (
        "If the CI excludes 0 the mpp=8 approximation is differentially biased between the "
        "corpora and the headline must be re-run at mpp=1 (~8x cost).")
    print(f"\n  bias(reps) - bias(retain) = {out['bias_difference_reps_minus_retain']:+.5f} "
          f"CI95 {out['bias_difference_ci95']}", flush=True)

    C.save_json(out, C.RESULTS / "ppl_exact_check.json")
    print("\nT2x OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
