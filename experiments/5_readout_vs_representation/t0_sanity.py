"""T0 — reproduction + splice sanity. Gates every other phase of experiment 5.

Three things must hold before a single hybrid number is worth reading:

1. **Reproduction.** Scoring theta_0 and B on F_test-256 with this module's settings must
   return each arm's stored ``forget_ppl_n0`` (mgnify: 13.1149 and 81,344.31 -> 6,202x).
   If it does not, the eval is not matched to the pipeline and no ratio in T1/T2 is
   comparable to the number this experiment exists to explain.
2. **Identity splice.** ``splice(theta_0 trunk, theta_0 head)`` must be BITWISE equal to
   theta_0 and score identically. A splice machine that perturbs anything on a no-op
   input would silently contaminate every hybrid.
3. **Key accounting.** The hybrid state dict must equal the head donor on exactly the
   intended keys and the trunk donor everywhere else — and for H3, the decoder must
   genuinely stop sharing storage with the input embedding.

Runtime: ~15 min for all five arms on one A100.

    python experiments/5_readout_vs_representation/t0_sanity.py --device cuda
    python experiments/5_readout_vs_representation/t0_sanity.py --arms mgnify
"""
from __future__ import annotations

import argparse

import common as C

# Reproduction tolerance on the stored forget_ppl_n0. Scoring is deterministic given the
# same corpus and settings, so this is a wiring check, not a statistical one; the slack
# only absorbs GPU non-determinism in the matmuls.
TOL = 0.01


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    return C.arm_args(p).parse_args()


def check_reproduction(kit, tok, corpora, device, arm, masks_per_pass):
    """Each donor's PPPL vs the pipeline's own stored number."""
    import math

    expected = arm.expected()
    out = {}
    for name in ("theta0", "B", "A_prime"):
        m = kit.donor(name)
        s = C.score(m, tok, corpora, device, masks_per_pass, label=f"{name} (donor)")
        exp = expected.get(name)
        rel = (abs(s["F"] - exp) / exp if exp and math.isfinite(s["F"]) else None)
        out[name] = {"pppl": s, "expected_F": exp, "rel_err": rel,
                     "ok": (rel is not None and rel <= TOL)}
        flag = "OK" if out[name]["ok"] else ("no stored value" if exp is None else "MISMATCH")
        print(f"    {name:<9} F={s['F']:>12.4f}  expected={exp}  rel_err="
              f"{'n/a' if rel is None else f'{rel:.2e}'}  [{flag}]", flush=True)
    return out


def check_identity(kit, tok, corpora, device, masks_per_pass, ref_pppl):
    """theta_0 trunk + theta_0 head must be bitwise theta_0."""
    import torch

    model, prov = kit.build("theta0", "theta0", C.HEADLINE_HEAD_SET)
    sd = model.state_dict()
    mismatched = [k for k, v in sd.items()
                  if not torch.equal(v.detach().cpu(), kit.sd["theta0"][k].cpu())]
    s = C.score(model, tok, corpora, device, masks_per_pass, label="identity splice")
    return {
        "bitwise_equal": not mismatched,
        "n_mismatched_keys": len(mismatched),
        "mismatched_keys": mismatched[:10],
        "pppl": s,
        "pppl_matches_theta0": abs(s["F"] - ref_pppl) < 1e-6,
        "provenance": prov,
    }


def check_accounting(kit, head_set: str):
    """The hybrid must take exactly the intended keys from the head donor."""
    import torch

    model, prov = kit.build("theta0", "B", head_set)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    transferred = set(prov["keys_from_head"])

    wrong_from_head, wrong_from_trunk = [], []
    for k, v in sd.items():
        if k == C.DECODER_KEY:
            continue                                   # checked separately below
        want = kit.sd["B"][k] if k in transferred else kit.sd["theta0"][k]
        if not torch.equal(v, want.cpu()):
            (wrong_from_head if k in transferred else wrong_from_trunk).append(k)

    decoder_ok = torch.equal(
        sd[C.DECODER_KEY],
        (kit.sd["B"] if head_set == "H3" else kit.sd["theta0"])[C.WORD_EMB_KEY].cpu())
    untied = (model.lm_head.decoder.weight.data_ptr()
              != model.esm.embeddings.word_embeddings.weight.data_ptr())

    return {
        "head_set": head_set,
        "n_keys_from_head": prov["n_keys_from_head"],
        "n_params_from_head": prov["n_params_from_head"],
        "frac_params_from_head": prov["frac_params_from_head"],
        "keys_from_head": prov["keys_from_head"],
        "decoder_untied": untied,
        "decoder_untied_as_intended": untied == (head_set == "H3"),
        "decoder_value_ok": bool(decoder_ok),
        "n_wrong_from_head": len(wrong_from_head),
        "n_wrong_from_trunk": len(wrong_from_trunk),
        "wrong_keys": (wrong_from_head + wrong_from_trunk)[:10],
        "ok": (not wrong_from_head and not wrong_from_trunk and decoder_ok
               and untied == (head_set == "H3")),
    }


def run_arm(arm, args, device) -> dict:
    C.banner(f"T0 — {arm.label}")
    tok = C.tokenizer()
    corpora = C.eval_corpora(arm, args.n_eval, args.seed)
    print(f"  corpora: F={len(corpora['F'])}  benign={len(corpora['benign'])}", flush=True)

    kit = C.SpliceKit(arm, device)
    try:
        print("  [1/3] reproduction", flush=True)
        repro = check_reproduction(kit, tok, corpora, device, arm, args.masks_per_pass)

        print("  [2/3] identity splice", flush=True)
        ident = check_identity(kit, tok, corpora, device, args.masks_per_pass,
                               repro["theta0"]["pppl"]["F"])
        print(f"    bitwise_equal={ident['bitwise_equal']}  "
              f"pppl_matches={ident['pppl_matches_theta0']}", flush=True)

        print("  [3/3] key accounting", flush=True)
        acct = {hs: check_accounting(kit, hs) for hs in C.HEAD_SET_ORDER}
        for hs, a in acct.items():
            print(f"    {hs}: {a['n_keys_from_head']} keys / {a['n_params_from_head']:,} params "
                  f"({a['frac_params_from_head']:.2%}) untied={a['decoder_untied']} "
                  f"[{'OK' if a['ok'] else 'FAIL'}]", flush=True)

        head_counts = kit.head_param_counts()
        n_total = kit.n_params_total
    finally:
        kit.close()

    gate = (all(v["ok"] for v in repro.values() if v["expected_F"] is not None)
            and ident["bitwise_equal"] and ident["pppl_matches_theta0"]
            and all(a["ok"] for a in acct.values()))

    res = {
        "arm": arm.key, "label": arm.label, "family": arm.family,
        "run_root": str(arm.run_root),
        "gate_pass": bool(gate),
        "reproduction": repro,
        "identity_splice": ident,
        "key_accounting": acct,
        "head_param_counts": head_counts,
        "n_params_total": n_total,
        "lm_head_displacement": arm.lm_head_rel_l2(),
        "protocol": C.protocol_block(args.n_eval, args.masks_per_pass, args.seed),
        "tolerance": TOL,
    }
    C.save_json(res, arm.results / "t0_sanity.json")
    print(f"  GATE: {'PASS' if gate else 'FAIL'}", flush=True)
    return res


def main() -> int:
    args = parse_args()
    device = C.device_arg(args.device, args.allow_cpu_fallback)
    C.banner(f"T0 SANITY — experiment 5 (device={device})")
    results = {k: run_arm(C.get_arm(k), args, device) for k in args.arms}

    C.banner("T0 SUMMARY")
    failed = [k for k, r in results.items() if not r["gate_pass"]]
    for k, r in results.items():
        f = r["reproduction"]["B"]["pppl"]["F"]
        b = r["reproduction"]["theta0"]["pppl"]["F"]
        print(f"  {k:<16} theta0={b:>9.4f}  B={f:>13.4f}  ratio={f / b:>10.1f}x  "
              f"[{'PASS' if r['gate_pass'] else 'FAIL'}]")
    if failed:
        print(f"\n  GATE FAILED for: {', '.join(failed)}")
        print("  Do NOT read T1/T2 for these arms — the eval is not matched to the pipeline.")
        return 1
    print("\n  All arms reproduce the pipeline's static block. T1 is safe to run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
