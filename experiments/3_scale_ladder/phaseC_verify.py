"""Phase C acceptance check — did B and A' come out valid at this scale?

Phase C itself is just the existing trainers (`scripts/03_train_blocked.py`,
`scripts/04_train_control.py`) pointed at a per-scale run root. This script asserts the
four things required before those checkpoints may be used for anything:

  1. **Checkpoints exist** — {A = theta_0, B, A'} at this scale.
  2. **Static forget fired**  — B's n=0 forget PPL is >> the random-prediction ceiling of
     20, i.e. B is *confidently wrong* on F rather than merely uncertain. (A block that
     did not fire makes the downstream durability question meaningless.)
  3. **Benign utility retained** — |PPL_R(B) - PPL_R(A')| / PPL_R(A') < the configured
     tolerance (2%). A' is the right comparator, not base: it isolates the safeguard from
     "extra training".
  4. **FLOP match held** — A''s realized FLOPs match B's within a negligible gap. If they
     do not, A' is not compute-matched and every multiplier measured against it is a
     validity threat, not a result.

    python experiments/3_scale_ladder/phaseC_verify.py --scales 150M 650M

Writes ``<ladder_root>/manifests/train_ladder.json``. Cheap (two benign PPL passes per
scale) and non-destructive — run it before Phase D/E, and again after any retrain.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import common
from common import (DEFAULT_LADDER_ROOT, HERE, RANDOM_CEILING, banner, get_scale,
                    manifest_dir, run_root, sort_scales, upsert_row)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "scale_ladder.yaml"))
    p.add_argument("--scales", nargs="+", default=None, help="default: config train_scales")
    p.add_argument("--ladder-root", default=str(DEFAULT_LADDER_ROOT))
    p.add_argument("--device", default="cuda")
    p.add_argument("--utility-tolerance", type=float, default=0.02)
    p.add_argument("--flop-gap-tolerance", type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    from omegaconf import OmegaConf

    from src.data.datamodule import DataPaths, subsample, target_test_seqs
    from src.eval.pseudo_perplexity import pseudo_perplexity
    from src.eval.utility import benign_pseudo_perplexity, utility_retention
    from src.models.load import load_checkpoint, load_tokenizer
    from src.utils.io import fasta_seqs, load_json
    from src.utils.seeding import set_seed

    cfg = OmegaConf.load(args.config)
    m = cfg.metrics
    seed = int(cfg.seed)
    ladder_root = Path(args.ladder_root)
    manifest = manifest_dir(ladder_root) / "train_ladder.json"
    failures = []

    for scale_name in sort_scales(args.scales or list(cfg.train_scales)):
        scale = get_scale(scale_name)
        root = run_root(ladder_root, scale_name)
        ckpt_B, ckpt_Ap = root / "checkpoints" / "B", root / "checkpoints" / "A_prime"
        banner(f"PHASE C VERIFY @ {scale_name}")
        if not ckpt_B.exists() or not ckpt_Ap.exists():
            print(f"  MISSING checkpoints (B={ckpt_B.exists()}, A'={ckpt_Ap.exists()}) — run Phase C")
            failures.append(scale_name)
            continue

        set_seed(seed)
        paths = DataPaths(str(root / "data"))
        device, max_len = args.device, int(m.max_length)
        eval_bs, mpp, n_eval = int(m.eval_batch_size), int(m.masks_per_pass), int(m.max_eval_seqs)
        tok = load_tokenizer(scale.model_id)
        f_test = subsample(target_test_seqs(paths, "F"), n_eval, seed=seed)
        holdout = subsample(fasta_seqs(paths.retain_holdout), n_eval, seed=seed)

        base = load_checkpoint(scale.model_id, device=device)
        ppl_f_A = pseudo_perplexity(base, tok, f_test, device, eval_bs, max_len, masks_per_pass=mpp).ppl
        del base
        _free()
        B = load_checkpoint(str(ckpt_B), device=device)
        ppl_f_B = pseudo_perplexity(B, tok, f_test, device, eval_bs, max_len, masks_per_pass=mpp).ppl
        ppl_r_B = benign_pseudo_perplexity(B, tok, holdout, device, eval_bs, max_len, masks_per_pass=mpp).ppl
        del B
        _free()
        Ap = load_checkpoint(str(ckpt_Ap), device=device)
        ppl_r_Ap = benign_pseudo_perplexity(Ap, tok, holdout, device, eval_bs, max_len, masks_per_pass=mpp).ppl
        del Ap
        _free()

        b_rec = _maybe(load_json, root / "results" / "B_compute.json")
        ap_rec = _maybe(load_json, root / "results" / "Aprime_compute.json")
        flop_gap = float(ap_rec.get("flop_match_rel_gap", float("nan"))) if ap_rec else float("nan")

        static_x = ppl_f_B / ppl_f_A if ppl_f_A else float("nan")
        drift = utility_retention(ppl_r_B, ppl_r_Ap)
        reasons = []
        if not (ppl_f_B > RANDOM_CEILING):
            reasons.append(f"static forget did not fire: PPL_F(B) {ppl_f_B:.2f} <= ceiling {RANDOM_CEILING:.0f}")
        if not (drift < args.utility_tolerance):
            reasons.append(f"benign drift vs A' {drift * 100:.2f}% >= {args.utility_tolerance * 100:.0f}%")
        if not (math.isfinite(flop_gap) and abs(flop_gap) < args.flop_gap_tolerance):
            reasons.append(f"FLOP-match gap {flop_gap} outside +/-{args.flop_gap_tolerance}")

        row = {
            "scale": scale_name, "model_id": scale.model_id, "n_params": scale.n_params,
            "run_root": str(root), "checkpoint_B": str(ckpt_B), "checkpoint_Aprime": str(ckpt_Ap),
            "forget_ppl_A": ppl_f_A, "forget_ppl_B_n0": ppl_f_B, "static_forget_x": static_x,
            "random_ceiling": RANDOM_CEILING,
            "benign_ppl_B": ppl_r_B, "benign_ppl_Aprime": ppl_r_Ap,
            "utility_retention_rel": drift, "utility_tolerance": args.utility_tolerance,
            "B_flops_total": (b_rec or {}).get("flops_total"),
            "B_overhead_vs_plain_step": (b_rec or {}).get("overhead_factor_vs_plain_step"),
            "B_wall_clock_s": (b_rec or {}).get("wall_clock_s"),
            "Aprime_realized_flops": (ap_rec or {}).get("realized_flops"),
            "flop_match_rel_gap": flop_gap,
            "n_eval_seqs": len(f_test),
            "accept": not reasons, "accept_fail_reasons": reasons,
        }
        upsert_row(manifest, row, meta={"seed": seed, "phase": "C"})

        print(f"  static forget : PPL_F {ppl_f_A:.2f} (A) -> {ppl_f_B:.1f} (B) = {static_x:.0f}x "
              f"[ceiling {RANDOM_CEILING:.0f}]")
        print(f"  benign        : PPL_R {ppl_r_Ap:.3f} (A') vs {ppl_r_B:.3f} (B) -> drift {drift * 100:.2f}%")
        print(f"  compute       : B {_e((b_rec or {}).get('flops_total'))} FLOPs, "
              f"A' {_e((ap_rec or {}).get('realized_flops'))}, match gap {flop_gap:.2e}")
        print(f"  => {'ACCEPT' if not reasons else 'REJECT: ' + '; '.join(reasons)}")
        if reasons:
            failures.append(scale_name)

    banner("PHASE C VERIFY SUMMARY")
    for r in common.load_manifest(manifest).get("rows", []):
        print(f"  {r['scale']:>5}  static {_f(r.get('static_forget_x'), 0):>8}x  "
              f"drift {100 * r.get('utility_retention_rel', float('nan')):.2f}%  "
              f"flop gap {r.get('flop_match_rel_gap')}  {'ACCEPT' if r.get('accept') else 'REJECT'}")
    print(f"\nmanifest: {manifest}")
    if failures:
        print(f"  not accepted at: {', '.join(failures)} — fix before Phase D/E; a non-matched A' or an "
              "unfired block invalidates every downstream number.")
        raise SystemExit(1)


def _maybe(fn, path):
    p = Path(path)
    return fn(str(p)) if p.exists() else None


def _free():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _e(x):
    return "n/a" if x is None else f"{float(x):.3e}"


if __name__ == "__main__":
    main()
