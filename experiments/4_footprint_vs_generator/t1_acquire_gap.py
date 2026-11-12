"""Test 1 — the train/test PPPL gap of the ACQUIRED model, per family.

    theta_acq = theta_0 continued-pretrained on F_train (fixed budget, no F_test-based
                early stopping — see common.acquire)

and then PPPL of theta_0 and theta_acq on F_train_seen / F_train_unseen /
F_test(decontaminated) / retain_holdout.

PREDICTION. Large gap for the OOD metagenomic family (acquisition memorises F_train and
transfers weakly to a homology-decontaminated F_test) and a small gap for the
in-distribution Pfam family (acquisition sharpens a representation that already
generalises).

THE CEILING CHECK THIS ALSO SETTLES. If metagenomic F_test barely improves over the naive
base PPPL, then part of "CIR fails to block" is an artifact: there is little generalising
capability there to remove in the first place, and tau sits in a regime where no method
could show a block. The report therefore states, per arm:

    acquisition_realised_test = (PPPL_naive - PPPL_acq[F_test]) / (PPPL_naive - PPPL_floor)

i.e. how much of the run's own measured headroom the acquired model actually realises on
decontaminated held-out data, and the same quantity on F_train. A big train number with a
small test number IS the dissociation; a small test number in BOTH arms would instead mean
the substrate cannot support the measurement, and that is reported as a caveat rather than
hidden.

    python experiments/4_footprint_vs_generator/t1_acquire_gap.py --arms pfam mgnify

One GPU. ~1-1.5 h per arm at 35M (8192 seqs x 40 epochs) plus ~10 min of scoring.
Writes results/<arm>/t1_gap.json and the theta_acq checkpoint into the arm's run root.
"""
from __future__ import annotations

import argparse
import math

import common as C


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    C.arm_args(p)
    p.add_argument("--epochs", type=int, default=C.ACQUIRE_EPOCHS)
    p.add_argument("--n-train", type=int, default=C.N_ACQUIRE)
    p.add_argument("--lr", type=float, default=C.ACQUIRE_LR)
    p.add_argument("--batch-size", type=int, default=C.ACQUIRE_BATCH)
    p.add_argument("--reuse-checkpoint", action="store_true",
                   help="score an existing theta_acq instead of retraining it")
    return p.parse_args()


def run_arm(arm: C.Arm, args, device: str) -> dict:
    C.banner(f"T1 — {arm.label}")
    tok = C.tokenizer()
    sets = C.eval_sets(arm, args.n_eval, seed=args.seed)
    test_key = C.primary_test_key(sets)
    print(f"  eval sets: " + ", ".join(f"{k}({len(v)})" for k, v in sets.items()))
    print(f"  primary held-out key: {test_key}"
          + ("" if test_key == "F_test_decon" else "  [WARN: t0 decontamination missing]"))

    thr = arm.thresholds()
    print(f"  run thresholds: naive={thr.get('ppl_naive')} floor={thr.get('ppl_floor')} "
          f"tau={thr.get('tau')} rel_headroom={thr.get('rel_headroom')}")

    print("\n  theta_0 (base ESM-2):")
    base = C.load_base(device)
    ppl_base = C.ppl_all(base, tok, sets, device, prefix="base ")
    del base

    if args.reuse_checkpoint and (arm.acquired_ckpt / "config.json").exists():
        print(f"\n  reusing theta_acq at {arm.acquired_ckpt}")
        acq = C.load_acquired(arm, device)
        train_rec = {"reused": True, "checkpoint": str(arm.acquired_ckpt)}
    else:
        print(f"\n  acquiring: {args.n_train} F_train seqs x {args.epochs} epochs, "
              f"lr={args.lr}, full fine-tune (no F_test early stopping)")
        out = C.acquire(arm, device, epochs=args.epochs, n_train=args.n_train, lr=args.lr,
                        batch_size=args.batch_size, seed=args.seed,
                        probe_n=min(args.n_eval, C.PROBE_N))
        acq = out.pop("model").eval()
        train_rec = out

    print("\n  theta_acq:")
    ppl_acq = C.ppl_all(acq, tok, sets, device, prefix="acq  ")
    del acq

    naive = thr.get("ppl_naive")
    floor = thr.get("ppl_floor")

    def realised(key: str) -> float:
        """Fraction of the achievable headroom that theta_acq realises ON THAT CORPUS.

        The "naive" end is theta_0's PPPL measured on the SAME corpus (so the
        decontaminated held-out set is not scored against a naive value measured on the
        native split). The floor end is the run's own PPL_floor, which was measured on the
        native F_test — an optimistic reference for the decontaminated set, so a small
        realised-on-test number is if anything understated, never inflated.
        """
        if floor is None:
            return float("nan")
        span = ppl_base[key] - floor
        return (ppl_base[key] - ppl_acq[key]) / span if abs(span) > 1e-9 else float("nan")

    gap = {
        # the headline dissociation numbers
        "gap_ratio_test_over_train_seen": ppl_acq[test_key] / ppl_acq["F_train_seen"],
        "gap_ratio_test_over_train_unseen": ppl_acq[test_key] / ppl_acq["F_train_unseen"],
        "gap_ratio_train_unseen_over_seen": ppl_acq["F_train_unseen"] / ppl_acq["F_train_seen"],
        "gap_nats_test_minus_train_seen": (
            math.log(ppl_acq[test_key]) - math.log(ppl_acq["F_train_seen"])),
        # what fraction of the family's measured headroom acquisition actually realises
        "acquisition_realised_train_seen": realised("F_train_seen"),
        "acquisition_realised_train_unseen": realised("F_train_unseen"),
        "acquisition_realised_test": realised(test_key),
        # per-set improvement over base
        "improvement_over_base": {k: ppl_base[k] / ppl_acq[k] for k in ppl_acq},
    }
    for k, v in gap.items():
        if isinstance(v, float):
            print(f"  {k:<40} {v:.4f}")

    rec = {
        "arm": arm.key, "label": arm.label, "kind": arm.kind, "family": arm.family,
        "primary_test_key": test_key,
        "decontaminated": test_key == "F_test_decon",
        "n_eval": {k: len(v) for k, v in sets.items()},
        "ppl_base": ppl_base, "ppl_acq": ppl_acq, "gap": gap,
        "thresholds": {k: thr.get(k) for k in
                       ("ppl_naive", "ppl_floor", "tau", "rel_headroom", "gate_pass")},
        "acquire": train_rec,
        "protocol": {"n_train": args.n_train, "epochs": args.epochs, "lr": args.lr,
                     "batch_size": args.batch_size, "early_stopping": "none (fixed budget)"},
    }
    C.save_json(rec, arm.results / "t1_gap.json")
    return rec


def main() -> int:
    args = parse_args()
    device = C.device_arg(args.device)
    C.banner(f"TEST 1 — TRAIN/TEST PPPL GAP ON THE ACQUIRED MODEL   (device={device})")
    out = {k: run_arm(C.get_arm(k), args, device) for k in args.arms}

    C.banner("T1 SUMMARY")
    hdr = f"{'arm':<10}{'test key':<16}{'train_seen':>11}{'train_unseen':>14}{'test':>9}{'gap x':>8}{'realised test':>15}"
    print(hdr)
    for k, r in out.items():
        a = r["ppl_acq"]
        print(f"{k:<10}{r['primary_test_key']:<16}{a['F_train_seen']:>11.3f}"
              f"{a['F_train_unseen']:>14.3f}{a[r['primary_test_key']]:>9.3f}"
              f"{r['gap']['gap_ratio_test_over_train_seen']:>8.2f}"
              f"{r['gap']['acquisition_realised_test']:>15.2f}")
    C.save_json(out, C.RESULTS / "t1_gap.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
