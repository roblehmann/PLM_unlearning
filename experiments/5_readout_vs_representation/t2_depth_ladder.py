"""T2 — depth ladder. Where in the stack does the block actually start?

T1 answers "head or trunk" as a binary. This turns it into a localization curve by
walking the splice boundary downward from the head:

    forward ladder  theta_0 trunk + B's H1 head + B's LAST k encoder layers
    reverse ladder  B trunk + theta_0's H1 head + theta_0's LAST k encoder layers

for k in {0, 1, 2, 4, 8, 12}, scored on F and on the benign holdout. The forward ladder
asks how much of B you must paste onto base before the block appears; the reverse asks how
much of base you must paste onto B before it disappears. A block concentrated in the
readout is flat-then-nothing; one spread through the trunk climbs steadily with k.

Two cross-checks make the phase self-validating:
* k=0 on the forward ladder IS T1's ``H_head`` at H1, and k=0 on the reverse IS
  ``H_trunk``. ``aggregate.py`` asserts they agree to floating-point noise.
* k=12 transfers every encoder layer but NOT the input embedding, the final LayerNorm or
  the contact head, so it need not equal B exactly. The residual gap is reported rather
  than assumed away — it is precisely the part of the block that lives outside the
  encoder stack.

Runtime: ~40 min for all five arms on one A100. Requires t0_sanity to have passed.

    python experiments/5_readout_vs_representation/t2_depth_ladder.py --device cuda
    python experiments/5_readout_vs_representation/t2_depth_ladder.py --arms mgnify
"""
from __future__ import annotations

import argparse

import common as C


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", nargs="+", type=int, default=C.K_LADDER,
                   help="encoder layers transferred, counted from the top")
    p.add_argument("--directions", nargs="+", default=["forward", "reverse"],
                   choices=["forward", "reverse"])
    return C.arm_args(p).parse_args()


def ladder(kit, tok, corpora, device, args, trunk: str, head: str,
           base_F: float, B_F: float, base_benign: float) -> list:
    rows = []
    for k in args.k:
        extra = C.layer_keys(kit.keys, k)
        model, prov = kit.build(trunk, head, C.HEADLINE_HEAD_SET, extra_keys=extra)
        s = C.score(model, tok, corpora, device, args.masks_per_pass, label=f"k={k}")
        fb = C.frac_block(s["F"], base_F, B_F)
        rows.append({
            "k": int(k),
            "pppl_F": s["F"], "pppl_benign": s["benign"],
            "frac_block": fb,
            "benign_drift": C.benign_drift(s["benign"], base_benign),
            "n_params_from_head": prov["n_params_from_head"],
            "frac_params_from_head": prov["frac_params_from_head"],
            "n_keys_from_head": prov["n_keys_from_head"],
        })
        print(f"      -> frac_block={'n/a' if fb is None else f'{fb:+.4f}'}  "
              f"({prov['frac_params_from_head']:.2%} of params transferred)", flush=True)
    return rows


def run_arm(arm, args, device) -> dict:
    C.banner(f"T2 — {arm.label}")
    tok = C.tokenizer()
    corpora = C.eval_corpora(arm, args.n_eval, args.seed)
    kit = C.SpliceKit(arm, device)
    try:
        donors = {name: C.score(kit.donor(name), tok, corpora, device, args.masks_per_pass,
                                label=name)
                  for name in ("theta0", "B")}
        base_F, B_F, base_benign = donors["theta0"]["F"], donors["B"]["F"], donors["theta0"]["benign"]

        out = {}
        if "forward" in args.directions:
            print("  forward ladder: theta_0 trunk + B head + B's last k layers", flush=True)
            out["forward"] = ladder(kit, tok, corpora, device, args, "theta0", "B",
                                    base_F, B_F, base_benign)
        if "reverse" in args.directions:
            print("  reverse ladder: B trunk + theta_0 head + theta_0's last k layers", flush=True)
            out["reverse"] = ladder(kit, tok, corpora, device, args, "B", "theta0",
                                    base_F, B_F, base_benign)
    finally:
        kit.close()

    # The k=12 residual: what the encoder stack alone cannot account for.
    residual = None
    if "forward" in out:
        top = [r for r in out["forward"] if r["k"] == max(args.k)]
        if top and top[0]["frac_block"] is not None:
            residual = {
                "k": top[0]["k"],
                "frac_block": top[0]["frac_block"],
                "missing_frac_block": 1.0 - top[0]["frac_block"],
                "note": "k=k_max transfers every encoder layer + the H1 head, but NOT the "
                        "input embedding, the final LayerNorm or the contact head; this is "
                        "the share of the block living outside that set.",
            }

    res = {
        "arm": arm.key, "label": arm.label, "run_root": str(arm.run_root),
        "donors": donors,
        "static_block_ratio": (B_F / base_F if base_F else None),
        "ladders": out,
        "k_values": [int(k) for k in args.k],
        "encoder_residual": residual,
        "head_set": C.HEADLINE_HEAD_SET,
        "protocol": C.protocol_block(args.n_eval, args.masks_per_pass, args.seed),
    }
    C.save_json(res, arm.results / "t2_depth_ladder.json")
    return res


def main() -> int:
    args = parse_args()
    device = C.device_arg(args.device, args.allow_cpu_fallback)
    C.banner(f"T2 DEPTH LADDER — experiment 5 (device={device})")
    results = {k: run_arm(C.get_arm(k), args, device) for k in args.arms}

    C.banner("T2 SUMMARY — frac_block by k (forward ladder)")
    ks = [int(k) for k in args.k]
    print(f"  {'arm':<16} " + " ".join(f"k={k:<7}" for k in ks))
    for key, r in results.items():
        row = {x["k"]: x["frac_block"] for x in r["ladders"].get("forward", [])}
        cells = [("n/a" if row.get(k) is None else f"{row[k]:+.3f}") for k in ks]
        print(f"  {key:<16} " + " ".join(f"{c:<9}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
