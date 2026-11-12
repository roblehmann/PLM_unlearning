"""T3 — does relearning ONLY the head reverse the block?

T1/T2 decompose the block statically. This closes the loop causally on the claim that
motivated the experiment: B's 6,202x static forget is undone by 16-64 real F examples. If
the block is a readout edit, then fine-tuning the 231,873 head parameters alone — 0.69% of
the model — should reverse it at those budgets.

Two modes, symmetric by construction:

    head_only   only the H1 head parameters are trainable (231,873)
    trunk_only  everything EXCEPT the H1 head is trainable (the control)

over budgets {16, 64, 256} x seeds {0, 1, 2}, on B, with the sweep's own full-FT recipe
read from configs/attack/full_ft.yaml. ``common.partial_attack`` replicates
``src/train/attack.py::run_attack`` step for step — same loader, clipping, eval cadence,
``best_target_ppl`` tracking and patience rule, and the same seed-dependent scoring
subsample — so these numbers are directly comparable to existing sweep rows.

The full fine-tuning baseline costs nothing: B's clean full-FT curve at the same budgets is
already on disk in each run's ``results/sweep/*shard*.csv`` and is read back for the
overlay. No checkpoints are written.

Runtime: ~1.5 h per arm on one A100 (90 runs across five arms). Requires t0_sanity.

    sbatch experiments/5_readout_vs_representation/t3_partial_attack.sbatch
    python experiments/5_readout_vs_representation/t3_partial_attack.py --arms mgnify
"""
from __future__ import annotations

import argparse
import time

import common as C


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budgets", nargs="+", type=int, default=C.T3_BUDGETS)
    p.add_argument("--seeds", nargs="+", type=int, default=C.T3_SEEDS)
    p.add_argument("--modes", nargs="+", default=C.T3_MODES, choices=C.T3_MODES)
    p.add_argument("--max-epochs", type=int, default=None,
                   help="override configs/attack/full_ft.yaml (smoke runs only)")
    return C.arm_args(p).parse_args()


def run_arm(arm, args, device) -> dict:
    from src.data.datamodule import attacker_trainset, target_test_seqs

    C.banner(f"T3 — {arm.label}")
    tok = C.tokenizer()
    paths = C.data_paths(arm)
    test_seqs = target_test_seqs(paths, "F")
    cfg = C.attack_cfg()
    if args.max_epochs is not None:
        cfg["max_epochs"] = int(args.max_epochs)
    print(f"  recipe: {cfg}", flush=True)

    baseline = C.sweep_baseline(arm)
    print(f"  full-FT baseline from sweep (median best_target_ppl): {baseline}", flush=True)

    rows = []
    for mode in args.modes:
        for n in args.budgets:
            for seed in args.seeds:
                t0 = time.time()
                seqs = attacker_trainset(paths, "F", "clean", int(n), seed=seed)
                res = C.partial_attack(str(arm.B), tok, seqs, test_seqs, mode, device,
                                       seed=seed, cfg=cfg, max_eval_seqs=args.n_eval,
                                       masks_per_pass=args.masks_per_pass)
                rec = {
                    "mode": mode, "budget_n": int(n), "seed": int(seed),
                    "best_target_ppl": res["best_target_ppl"],
                    "naive_target_ppl": res["naive_target_ppl"],
                    "final_target_ppl": res["final_target_ppl"],
                    "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
                    "trainable_params": res["trainable_params"],
                    "frac_params_trainable": res["trainable_params"]
                    / (res["trainable_params"] + res["frozen_params"]),
                    "recovery": res["naive_target_ppl"] / res["best_target_ppl"]
                    if res["best_target_ppl"] else None,
                    "wall_clock_s": time.time() - t0,
                    "history": res["history"],
                }
                rows.append(rec)
                print(f"    {mode:<11} n={n:<5} seed={seed}  "
                      f"{res['naive_target_ppl']:>12.2f} -> {res['best_target_ppl']:>9.4f}  "
                      f"({rec['frac_params_trainable']:.2%} of params, "
                      f"{rec['wall_clock_s'] / 60:.1f} min)", flush=True)

    res = {
        "arm": arm.key, "label": arm.label, "run_root": str(arm.run_root),
        "checkpoint": str(arm.B),
        "rows": rows,
        "full_ft_baseline_from_sweep": baseline,
        "recipe": cfg,
        "budgets": [int(b) for b in args.budgets],
        "seeds": [int(s) for s in args.seeds],
        "modes": list(args.modes),
        "head_set": C.HEADLINE_HEAD_SET,
        "protocol": C.protocol_block(args.n_eval, args.masks_per_pass, args.seed),
        "note": "target PPPL is scored on subsample(F_test, n_eval, seed=cell seed) — the "
                "same seed-dependent set run_attack uses, so rows line up with sweep rows.",
    }
    C.save_json(res, arm.results / "t3_partial_attack.json")
    return res


def main() -> int:
    args = parse_args()
    device = C.device_arg(args.device, args.allow_cpu_fallback)
    C.banner(f"T3 PARTIAL ATTACK — experiment 5 (device={device})")
    results = {k: run_arm(C.get_arm(k), args, device) for k in args.arms}

    C.banner("T3 SUMMARY — median best_target_ppl")
    import statistics

    for key, r in results.items():
        print(f"  {key}")
        for mode in r["modes"]:
            for n in r["budgets"]:
                v = [x["best_target_ppl"] for x in r["rows"]
                     if x["mode"] == mode and x["budget_n"] == n]
                base = r["full_ft_baseline_from_sweep"].get(str(n))
                if v:
                    print(f"    {mode:<11} n={n:<5} median={statistics.median(v):>9.4f}   "
                          f"full_ft(sweep)={'n/a' if base is None else f'{base:.4f}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
