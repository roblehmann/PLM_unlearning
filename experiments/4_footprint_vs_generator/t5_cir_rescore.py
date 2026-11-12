"""Test 5 (follow-up) — was the Pfam CIR "6.41x block" a leakage artifact?

t1-t4 showed that the distinctive footprint continued pretraining writes is memorisation,
not the generator of decontaminated F_test, for BOTH families. That predicts a specific,
cheap-to-check consequence about the safeguard itself:

    The original Pfam CIR block (forget_ppl_B/forget_ppl_base = 49.58/7.74 = 6.41x) was
    measured on the NATIVE F_test, which is 94% train-homologous. If the block is CIR
    collapsing the distinctive footprint (= memorisation of the F_train homology cloud),
    then on a decontaminated F_test — sequences OUTSIDE that cloud — the same B should
    show a much SMALLER block, because there was little transferable capability there for
    the collapse to remove in the first place.

This script re-scores the EXISTING checkpoints (no training) on both F_test variants:

    native   F_test  (256-subsample, seed 0)  — reproduces the published 6.41x
    decon    F_test  (the identical-criterion homology-clean set from t0)

for base theta_0, the CIR-B (cir35m_peak, the 6.41x operating point), its compute-matched
control A' (should be ~1x — no block), and the first-order mlac_fo B (the 1371x safeguard),
using the project's own scoring settings so the native numbers line up with thresholds.json.

    python experiments/4_footprint_vs_generator/t5_cir_rescore.py

One GPU, ~10 min. Writes results/cir_rescore.json (+ per-arm copy). aggregate.py folds it in.
"""
from __future__ import annotations

import argparse

import common as C

# The models to re-score, as (label, checkpoint path relative to repo root, note).
MODELS = [
    ("base", None, "theta_0 (ESM-2), never trained on F"),
    ("cir_B", "checkpoints/cir35m_peak/B", "CIR-B, the published 6.41x block (r0.2_t1.01)"),
    ("cir_A_prime", "checkpoints/cir35m_peak/A_prime", "CIR compute-matched control (expect ~1x)"),
    ("mlac_fo_B", "checkpoints/B", "first-order task-block B (the 1371x safeguard)"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-eval", type=int, default=C.N_EVAL)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    from src.data.datamodule import subsample
    from src.models.load import load_checkpoint
    from src.utils.io import fasta_seqs

    args = parse_args()
    device = C.device_arg(args.device)
    arm = C.get_arm("pfam")
    C.banner("T5 — CIR block re-scored on decontaminated F_test (Pfam)")

    tok = C.tokenizer()
    native = subsample(fasta_seqs(arm.data / "splits" / "F_test.fasta"), args.n_eval, seed=args.seed)
    decon_path = arm.results / "F_test_decon.fasta"
    if not decon_path.exists():
        raise SystemExit(f"missing {decon_path} — run t0_decontaminate.py first")
    decon_all = fasta_seqs(decon_path)
    decon = subsample(decon_all, args.n_eval, seed=args.seed)
    print(f"  native F_test: scoring {len(native)} seqs (seed {args.seed})")
    print(f"  decon  F_test: {len(decon_all)} available, scoring {len(decon)}")

    rows = {}
    for label, ckpt, note in MODELS:
        path = C.MODEL_ID if ckpt is None else str(C.ROOT / ckpt)
        if ckpt is not None and not (C.ROOT / ckpt / "config.json").exists():
            print(f"  {label}: checkpoint {ckpt} absent, skipping")
            continue
        model = load_checkpoint(path, device=device).eval()
        pn = C.ppl(model, tok, native, device)
        pd = C.ppl(model, tok, decon, device)
        rows[label] = {"ppl_native": pn, "ppl_decon": pd, "note": note, "checkpoint": ckpt}
        print(f"  {label:<14} native={pn:.3f}  decon={pd:.3f}   ({note})")
        del model
        if device.startswith("cuda"):
            import torch
            torch.cuda.empty_cache()

    base = rows.get("base", {})
    out = {"n_native": len(native), "n_decon": len(decon), "n_decon_available": len(decon_all),
           "seed": args.seed, "eval": {"masks_per_pass": C.MASKS_PER_PASS, "batch": C.EVAL_BATCH},
           "models": rows, "blocks": {}}

    C.banner("BLOCK RATIOS  (B F_test PPPL / base F_test PPPL)")
    print(f"  {'model':<14}{'native block':>14}{'decon block':>14}{'shrinkage':>12}")
    for label, r in rows.items():
        if label == "base" or not base:
            continue
        bn = r["ppl_native"] / base["ppl_native"] if base.get("ppl_native") else float("nan")
        bd = r["ppl_decon"] / base["ppl_decon"] if base.get("ppl_decon") else float("nan")
        shrink = bn / bd if bd else float("nan")
        out["blocks"][label] = {"block_native": bn, "block_decon": bd,
                                "block_shrinkage_native_over_decon": shrink}
        print(f"  {label:<14}{bn:>13.2f}x{bd:>13.2f}x{shrink:>11.2f}x")

    cir = out["blocks"].get("cir_B", {})
    if cir:
        print(f"\n  CIR block: {cir['block_native']:.2f}x on native (published ~6.41x) -> "
              f"{cir['block_decon']:.2f}x on decontaminated F_test.")
        print("  A large native/decon shrinkage => the block was collapse of the train-homologous "
              "memorisation, not of the decontaminated-F_test generator (the t1-t4 mechanism, "
              "now shown on the safeguard itself).")
    C.save_json(out, C.RESULTS / "cir_rescore.json")
    C.save_json(out, arm.results / "cir_rescore.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
