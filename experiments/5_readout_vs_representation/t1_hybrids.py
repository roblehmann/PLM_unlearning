"""T1 — the two hybrids. The headline phase of experiment 5.

    H_head  = theta_0 trunk + B head    ("is the block a readout?")
    H_trunk = B trunk + theta_0 head    (the complement)

scored on F_test-256 and on the retain holdout, for all three head sets H1/H2/H3, plus
two controls and one dose-response curve:

* **A' splice controls** (``A_head``, ``A_trunk``). A' is the compute-matched do-nothing
  control and carries essentially no block, so splicing its head onto theta_0 must be a
  no-op. If it is not, splicing PER SE is damaging and every B number here is confounded —
  this is what makes T1 interpretable rather than suggestive, and it is why the phase
  reports "inconclusive" as a first-class outcome.
* **Benign PPPL on every row.** A hybrid with a huge F-PPPL and a wrecked benign PPPL is a
  broken model, not a readout block. B itself is selective (2.08% benign drift at 6,202x);
  a readout claim requires H_head to be comparably selective. This is the falsifier.
* **Head-only alpha path** (H1, theta_0 trunk, head interpolated theta_0 -> B). Gives a
  dose-response and reuses the project's LMC barrier convention, so the curve is directly
  comparable to the existing fig_lmc_barrier.

Read the result off ``frac_block`` (fraction of the block reproduced, in log space):
frac(H_head)~1 & frac(H_trunk)~0 => readout; the reverse => representational; both mid =>
distributed, and T2 localizes it.

Runtime: ~2-3 h for all five arms on one A100. Requires t0_sanity to have passed.

    python experiments/5_readout_vs_representation/t1_hybrids.py --device cuda
    python experiments/5_readout_vs_representation/t1_hybrids.py --arms mgnify
"""
from __future__ import annotations

import argparse
import math

import common as C

# A' carries no block, so its splice must not move F. Above this the splice itself is
# suspect and the arm is reported inconclusive rather than as evidence either way.
CONTROL_TOL = 0.05
# Benign drift a hybrid may show and still count as a selective (readout-like) edit.
# B itself sits at ~2%; 10% is deliberately loose so the verdict fails only on models
# that are actually broken.
BENIGN_TOL = 0.10
# frac_block bands for the verdict.
HIGH, LOW = 0.70, 0.30


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-alpha", action="store_true",
                   help="skip the head-only interpolation path (the 5 extra evals)")
    return C.arm_args(p).parse_args()


# The hybrid grid: (row name, trunk donor, head donor).
HYBRIDS = [
    ("H_head", "theta0", "B"),          # headline: does the head alone carry the block?
    ("H_trunk", "B", "theta0"),         # headline: does the trunk alone carry it?
    ("A_head", "theta0", "A_prime"),    # control: splicing per se must be a no-op
    ("A_trunk", "A_prime", "theta0"),   # control
]


def alpha_path(kit, tok, corpora, device, masks_per_pass) -> dict:
    """theta_0 trunk, head interpolated theta_0 -> B. Barrier via the project's own rule."""
    from src.analysis.weightspace import _barrier_nats

    curves = {"F": [], "benign": []}
    for a in C.ALPHAS:
        model, _ = kit.build_alpha("theta0", "B", a, C.HEADLINE_HEAD_SET)
        s = C.score(model, tok, corpora, device, masks_per_pass, label=f"head alpha={a:.2f}")
        for name in curves:
            curves[name].append(s[name])
    out = {"alphas": [float(a) for a in C.ALPHAS], "head_set": C.HEADLINE_HEAD_SET}
    for name, curve in curves.items():
        barrier, mono = _barrier_nats(out["alphas"], curve)
        out[f"{name}_ppl"] = curve
        out[f"{name}_barrier_nats"] = barrier
        out[f"{name}_monotonic"] = mono
    return out


def verdict(rows: dict, head_set: str) -> dict:
    """Turn one head set's numbers into the four-way reading."""
    hh = rows["H_head"]["frac_block"]
    ht = rows["H_trunk"]["frac_block"]
    ctrl = [rows["A_head"]["frac_block"], rows["A_trunk"]["frac_block"]]
    ctrl_ok = all(c is not None and abs(c) <= CONTROL_TOL for c in ctrl)
    benign_ok = all(rows[n]["benign_drift"] is not None
                    and abs(rows[n]["benign_drift"]) <= BENIGN_TOL
                    for n in ("H_head", "H_trunk"))

    if hh is None or ht is None:
        call, why = "inconclusive", "frac_block undefined (arm's B did not block)"
    elif not ctrl_ok:
        call, why = "inconclusive", (
            f"A' splice control moved F (frac_block {ctrl}); splicing per se is damaging, "
            f"so the B numbers are confounded")
    elif not benign_ok:
        call, why = "inconclusive", (
            "a hybrid's benign PPPL is wrecked — that is a broken model, not a selective "
            "readout block")
    elif hh >= HIGH and ht <= LOW:
        call, why = "readout", (
            f"{rows['H_head']['frac_params_from_head']:.2%} of parameters reproduces "
            f"{hh:.0%} of the block while the trunk alone reproduces {ht:.0%}")
    elif ht >= HIGH and hh <= LOW:
        call, why = "representational", (
            f"the trunk reproduces {ht:.0%} of the block and the head only {hh:.0%}")
    else:
        call, why = "distributed", (
            f"neither part dominates (head {hh:.0%}, trunk {ht:.0%}); see the T2 depth ladder")
    return {"head_set": head_set, "call": call, "reason": why,
            "frac_block_H_head": hh, "frac_block_H_trunk": ht,
            "frac_block_sum": (None if hh is None or ht is None else hh + ht),
            "control_ok": ctrl_ok, "benign_ok": benign_ok}


def run_arm(arm, args, device) -> dict:
    C.banner(f"T1 — {arm.label}")
    tok = C.tokenizer()
    corpora = C.eval_corpora(arm, args.n_eval, args.seed)
    kit = C.SpliceKit(arm, device)
    try:
        print("  donors", flush=True)
        donors = {name: C.score(kit.donor(name), tok, corpora, device, args.masks_per_pass,
                                label=name)
                  for name in ("theta0", "B", "A_prime")}
        base_F, B_F = donors["theta0"]["F"], donors["B"]["F"]
        base_benign = donors["theta0"]["benign"]

        by_head_set, verdicts = {}, {}
        for hs in C.HEAD_SET_ORDER:
            print(f"  head set {hs} — {C.HEAD_SET_DOC[hs]}", flush=True)
            rows = {}
            for name, trunk, head in HYBRIDS:
                model, prov = kit.build(trunk, head, hs)
                s = C.score(model, tok, corpora, device, args.masks_per_pass, label=name)
                rows[name] = {
                    "trunk": trunk, "head": head,
                    "pppl_F": s["F"], "pppl_benign": s["benign"],
                    "ratio_F_over_base": (s["F"] / base_F if base_F else None),
                    "frac_block": C.frac_block(s["F"], base_F, B_F),
                    "benign_drift": C.benign_drift(s["benign"], base_benign),
                    "n_params_from_head": prov["n_params_from_head"],
                    "frac_params_from_head": prov["frac_params_from_head"],
                    "n_keys_from_head": prov["n_keys_from_head"],
                    "decoder_untied": prov["decoder_untied"],
                }
                fb, bd = rows[name]["frac_block"], rows[name]["benign_drift"]
                print(f"      -> frac_block={'n/a' if fb is None else f'{fb:+.4f}'}  "
                      f"benign_drift={'n/a' if bd is None else f'{bd:+.4f}'}", flush=True)
            by_head_set[hs] = rows
            verdicts[hs] = verdict(rows, hs)
            print(f"    {hs} verdict: {verdicts[hs]['call'].upper()} — {verdicts[hs]['reason']}",
                  flush=True)

        alpha = None
        if not args.skip_alpha:
            print("  head-only alpha path (H1)", flush=True)
            alpha = alpha_path(kit, tok, corpora, device, args.masks_per_pass)

        head_counts = kit.head_param_counts()
    finally:
        kit.close()

    res = {
        "arm": arm.key, "label": arm.label, "family": arm.family,
        "run_root": str(arm.run_root),
        "donors": donors,
        "static_block_ratio": (B_F / base_F if base_F else None),
        "block_nats": (None if not (base_F and B_F) else math.log(B_F / base_F)),
        "hybrids": by_head_set,
        "verdicts": verdicts,
        "headline": verdicts[C.HEADLINE_HEAD_SET],
        "alpha_path": alpha,
        "head_param_counts": head_counts,
        "lm_head_displacement": arm.lm_head_rel_l2(),
        "protocol": C.protocol_block(args.n_eval, args.masks_per_pass, args.seed),
        "thresholds": {"control_tol": CONTROL_TOL, "benign_tol": BENIGN_TOL,
                       "high": HIGH, "low": LOW},
    }
    C.save_json(res, arm.results / "t1_hybrids.json")
    return res


def main() -> int:
    args = parse_args()
    device = C.device_arg(args.device, args.allow_cpu_fallback)
    C.banner(f"T1 HYBRIDS — experiment 5 (device={device})")
    results = {k: run_arm(C.get_arm(k), args, device) for k in args.arms}

    C.banner("T1 SUMMARY (head set H1)")
    print(f"  {'arm':<16} {'block':>10}  {'H_head':>8} {'H_trunk':>8}  "
          f"{'A_head':>8} {'A_trunk':>8}   verdict")
    for k, r in results.items():
        h = r["hybrids"][C.HEADLINE_HEAD_SET]

        def f(n):
            v = h[n]["frac_block"]
            return "n/a" if v is None else f"{v:+.3f}"

        print(f"  {k:<16} {r['static_block_ratio']:>9.0f}x  {f('H_head'):>8} {f('H_trunk'):>8}  "
              f"{f('A_head'):>8} {f('A_trunk'):>8}   {r['headline']['call']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
