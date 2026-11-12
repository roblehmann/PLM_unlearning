"""Test 4 — project the distinctive subspace out BY HAND and see whether F_test survives.

This is the decisive test, and it is deliberately independent of CIR. Instead of asking
"did the CIR optimiser fail?", we remove each candidate subspace exactly — a hard linear
projection of ``output.dense``'s input onto the orthogonal complement of the basis, at
every layer in CIR's band, on the frozen theta_acq — and score PPPL. No optimiser, no
retain objective, no termination rule, so whatever happens is a property of the SUBSPACE.

Ablated subspaces (from t3's bases.npz, 24 dims each):

  F_distinctive@acq  the distinctive footprint  <- THE TEST. If decontaminated F_test PPPL
                     barely moves while F_train_seen collapses, the footprint is not the
                     generator, and CIR's failure on the OOD family is explained without
                     reference to CIR at all.
  cir_edit           the directions CIR's optimiser actually moved
  delta_acquire      the directions continued pretraining wrote
  gen_test           POSITIVE CONTROL: the F_test-generating directions. F_test PPPL must
                     blow up here, otherwise the gradient-moment definition of "generator"
                     is not causally meaningful and nothing else in t3/t4 can be read.
  gen_train          the F_train-generating directions
  R_shared@base      the shared retain machinery — expected to damage EVERYTHING including
                     retain_holdout, which is exactly why a utility-preserving method may
                     not touch it
  random             NEGATIVE CONTROL: a Haar-random 24-frame in the same 1920-dim space.
                     Any effect smaller than this is noise.

Run on theta_acq (primary) and on theta_0 (context: what the projection costs a model that
never acquired the family).

    python experiments/4_footprint_vs_generator/t4_causal_ablation.py --arms pfam mgnify

Needs t3's bases.npz. One GPU, ~30 min per arm. Writes results/<arm>/t4_causal.json.
"""
from __future__ import annotations

import argparse

import common as C

ABLATIONS = ["F_distinctive@acq", "cir_edit", "delta_acquire", "gen_test", "gen_train",
             "F_distinctive@base", "R_shared@base", "random"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    C.arm_args(p)
    p.add_argument("--subspaces", nargs="+", default=ABLATIONS)
    p.add_argument("--models", nargs="+", default=["acq", "base"], choices=["acq", "base"])
    p.add_argument("--n-random", type=int, default=3, help="random-control draws to average")
    return p.parse_args()


def load_bases(arm: C.Arm, device: str):
    import numpy as np
    import torch

    path = arm.results / "bases.npz"
    if not path.exists():
        raise SystemExit(f"missing {path} — run t3_subspaces.py first")
    z = np.load(path)
    out = {}
    for key in z.files:
        name, li = key.split("|")
        out.setdefault(name, {})[int(li)] = torch.as_tensor(z[key], device=device)
    return out


def run_arm(arm: C.Arm, args, device: str) -> dict:
    import torch

    C.banner(f"T4 — {arm.label}")
    tok = C.tokenizer()
    sets = C.eval_sets(arm, args.n_eval, seed=args.seed)
    test_key = C.primary_test_key(sets)
    score_sets = {k: sets[k] for k in ("F_train_seen", "F_train_unseen", test_key, "retain_holdout")}
    bases = load_bases(arm, device)
    layers = sorted(next(iter(bases.values())).keys())
    dim = next(iter(next(iter(bases.values())).values())).shape[1]
    # Match the random control's dimension to the subspace this test is really about, so
    # "excess over random" compares like with like (F_distinctive can lose a row or two to
    # the QR rank check, and a k-mismatched control would flatter or penalise it).
    ref = bases.get("F_distinctive@acq") or next(iter(bases.values()))
    k_default = int(min(v.shape[0] for v in ref.values()))
    print(f"  layers={layers} dim={dim} k={k_default} primary held-out={test_key}")

    rec = {"arm": arm.key, "label": arm.label, "kind": arm.kind, "primary_test_key": test_key,
           "layers": layers, "dim": int(dim), "models": {}}

    for which in args.models:
        model = C.load_acquired(arm, device) if which == "acq" else C.load_base(device)
        print(f"\n  === theta_{which} ===")
        intact = C.ppl_all(model, tok, score_sets, device, prefix="intact ")
        rows = {}
        for name in args.subspaces:
            if name == "random":
                per_draw = []
                for d in range(args.n_random):
                    rb = {li: C.random_basis(dim, k_default, seed=1000 + d, device=device)
                          for li in layers}
                    with C.ActProjection(model, rb):
                        per_draw.append({s: C.ppl(model, tok, q, device) for s, q in score_sets.items()})
                row = {s: sum(p[s] for p in per_draw) / len(per_draw) for s in score_sets}
                row["n_draws"] = args.n_random
            elif name not in bases:
                print(f"  {name}: basis unavailable, skipped")
                continue
            else:
                with C.ActProjection(model, bases[name]):
                    row = {s: C.ppl(model, tok, q, device) for s, q in score_sets.items()}
            for s in score_sets:
                row[f"rel_{s}"] = row[s] / intact[s]
            rows[name] = row
            print(f"  ablate {name:<20}" + "  ".join(
                f"{s}={row[s]:.3f}({row['rel_' + s]:.2f}x)" for s in score_sets), flush=True)

        # The headline contrast: what the footprint ablation costs F_train vs F_test,
        # net of the random-subspace control (which sets the noise floor for "an effect").
        summary = {}
        rnd = rows.get("random", {})
        for name, row in rows.items():
            if name == "random":
                continue
            sel = {}
            for s in score_sets:
                base_rel = rnd.get(f"rel_{s}", 1.0)
                sel[s] = row[f"rel_{s}"] / base_rel if base_rel else float("nan")
            sel["selectivity_train_over_test"] = (
                sel["F_train_seen"] / sel[test_key] if sel[test_key] else float("nan"))
            summary[name] = sel
        rec["models"][which] = {"intact": intact, "ablated": rows,
                                "excess_over_random": summary}
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    acq = rec["models"].get("acq", {})
    if acq:
        s = acq["excess_over_random"].get("F_distinctive@acq", {})
        print(f"\n  FOOTPRINT ABLATION (excess over random control):"
              f"  F_train_seen x{s.get('F_train_seen', float('nan')):.2f}"
              f"  {test_key} x{s.get(test_key, float('nan')):.2f}"
              f"  selectivity={s.get('selectivity_train_over_test', float('nan')):.2f}")
    C.save_json(rec, arm.results / "t4_causal.json")
    return rec


def main() -> int:
    args = parse_args()
    device = C.device_arg(args.device)
    C.banner(f"TEST 4 — DIRECT CAUSAL PROJECTION-OUT OF THE DISTINCTIVE SUBSPACE   (device={device})")
    out = {k: run_arm(C.get_arm(k), args, device) for k in args.arms}

    C.banner("T4 SUMMARY — theta_acq, PPPL relative to intact (x), random control in brackets")
    for k, r in out.items():
        m = r["models"].get("acq")
        if not m:
            continue
        tk = r["primary_test_key"]
        print(f"\n{r['label']}")
        print(f"  {'ablated subspace':<22}{'F_train_seen':>14}{'F_train_unseen':>16}{tk:>18}{'retain':>10}")
        for name, row in m["ablated"].items():
            print(f"  {name:<22}{row['rel_F_train_seen']:>14.2f}{row['rel_F_train_unseen']:>16.2f}"
                  f"{row['rel_' + tk]:>18.2f}{row['rel_retain_holdout']:>10.2f}")
    C.save_json(out, C.RESULTS / "t4_causal.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
