"""Test 3 — does CIR's collapse target overlap Delta_acquire, the F_test generator, or both?

All bases live in ONE space per layer: the input space of ``esm.encoder.layer[i].output.dense``
(the post-GELU FFN activations, dim 1920 at 35M), for CIR's band [3..8]. Six named bases,
each 24 orthonormal rows:

  R_shared@base     top-24 activation PCs of the retain set R on theta_0 — the general
                    covariation machinery CIR must spare.
  F_distinctive@base  F's top activation PCs with span(R_shared) removed, on theta_0 —
                    CIR's NOMINAL target: the distinctive footprint, and the object the
                    published O(F,R)=removable_topk number scores (0.672 pfam / 0.649 mgnify).
  F_distinctive@acq same, computed on theta_acq — the footprint of the model that actually
                    HAS the family.
  cir_edit          top-24 right singular vectors of (W_cirB - W_0) at output.dense — what
                    CIR's optimiser REALLY moved. The honest empirical target, independent
                    of what the method intended.
  delta_acquire     top-24 right singular vectors of (W_acq - W_0) at the same module —
                    the input-side directions continued pretraining writes.
  gen_test          top-24 eigenvectors of E[g g^T], g = dL_MLM/da on decontaminated F_test,
                    under theta_acq — the operational F_test-GENERATING directions.
  gen_train         the same on F_train_seen — the memorisation-side comparator.

PREDICTIONS.
  * O(cir_edit, delta_acquire) is high in BOTH arms — CIR does find the footprint
    (consistent with removable_topk being near-identical and with CIR's 7.4% displacement).
  * O(cir_edit, gen_test) and O(delta_acquire, gen_test) are high for pfam and LOW for
    mgnify. That is the dissociation, stated as a subspace fact.
  * O(delta_acquire, gen_train) >> O(delta_acquire, gen_test) for mgnify only: the update
    is coupled to the corpus it memorised, not to the held-out family.

Overlaps are energy fractions in [0,1] and are ALWAYS reported as a multiple of the
random-subspace chance level (24/1920 = 1.25%); a raw 0.10 means nothing on its own.

    python experiments/4_footprint_vs_generator/t3_subspaces.py --arms pfam mgnify

One GPU, ~40 min per arm. Writes results/<arm>/{t3_overlaps.json, bases.npz}; t4 consumes
the .npz, so run this before t4.
"""
from __future__ import annotations

import argparse
import itertools

import common as C

BASIS_ORDER = ["R_shared@base", "F_distinctive@base", "F_distinctive@acq",
               "cir_edit", "delta_acquire", "gen_test", "gen_train"]

# The pairs the mechanism actually makes a claim about (everything is computed, these are
# what the console summary and the aggregator lead with).
HEADLINE_PAIRS = [
    ("cir_edit", "delta_acquire"),
    ("cir_edit", "gen_test"),
    ("delta_acquire", "gen_test"),
    ("delta_acquire", "gen_train"),
    ("F_distinctive@acq", "gen_test"),
    ("F_distinctive@acq", "delta_acquire"),
    ("gen_train", "gen_test"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    C.arm_args(p)
    p.add_argument("--n-pcs", type=int, default=C.N_PCS)
    p.add_argument("--corpus-seqs", type=int, default=2048, help="sequences for the activation PCAs")
    p.add_argument("--grad-seqs", type=int, default=1024, help="sequences for the gradient moments")
    p.add_argument("--pc-batch", type=int, default=16)
    p.add_argument("--grad-batch", type=int, default=8)
    return p.parse_args()


def build_bases(arm: C.Arm, args, device: str):
    """{basis_name: {layer: tensor[k, dim]}} — every basis orthonormal, same space."""
    import numpy as np
    import torch

    from src.analysis.cir_fvsr import build_fvsr_bases
    from src.utils.io import fasta_seqs

    tok = C.tokenizer()
    sets = C.eval_sets(arm, args.n_eval, seed=args.seed)
    test_key = C.primary_test_key(sets)
    retain = fasta_seqs(arm.data / "splits" / "retain_train.fasta")
    # Activation PCAs use the FULL F_train, exactly as scripts/08e_cir_overlap.py did, so
    # F_distinctive@base is the same object the published removable_topk scored. The
    # gradient moments below instead use the acquisition subset, because gen_train is
    # deliberately the *memorisation* probe: the sequences theta_acq actually saw.
    f_family = fasta_seqs(arm.data / "splits" / "F_train.fasta")
    f_train = C.acquisition_seqs(arm)
    f_test_path = (arm.results / "F_test_decon.fasta") if test_key == "F_test_decon" \
        else (arm.data / "splits" / "F_test.fasta")
    f_test = fasta_seqs(f_test_path)
    print(f"  corpora: retain={len(retain)} F_train(all)={len(f_family)} "
          f"F_train_seen={len(f_train)} {test_key}={len(f_test)} (from {f_test_path})")

    base = C.load_base(device)
    layers = C.layers_for(base)
    dim = C.dense(base, layers[0]).in_features
    print(f"  layers={layers}  activation dim={dim}  k={args.n_pcs}  "
          f"chance overlap={args.n_pcs / dim:.4f}")

    bases = {n: {} for n in BASIS_ORDER}

    # --- activation PCAs on theta_0 (CIR's nominal target, as CIR saw it) --- #
    print("\n  F-vs-R activation bases on theta_0 ...", flush=True)
    fv0 = build_fvsr_bases(base, tok, layers, retain, f_family, n_pcs=args.n_pcs,
                           corpus_seqs=args.corpus_seqs, pc_batch=args.pc_batch,
                           max_length=C.MAX_LENGTH, seed=args.seed, device=device)
    for li in layers:
        bases["R_shared@base"][li] = torch.as_tensor(fv0[li]["R_shared"], device=device)
        bases["F_distinctive@base"][li] = torch.as_tensor(fv0[li]["F_distinctive"], device=device)

    base_dense = {li: C.dense(base, li).weight.detach().float().cpu().clone() for li in layers}
    del base
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # --- the same, on theta_acq (the model that has the family) ------------- #
    acq = C.load_acquired(arm, device)
    print("\n  F-vs-R activation bases on theta_acq ...", flush=True)
    fva = build_fvsr_bases(acq, tok, layers, retain, f_family, n_pcs=args.n_pcs,
                           corpus_seqs=args.corpus_seqs, pc_batch=args.pc_batch,
                           max_length=C.MAX_LENGTH, seed=args.seed, device=device)
    for li in layers:
        bases["F_distinctive@acq"][li] = torch.as_tensor(fva[li]["F_distinctive"], device=device)

    # --- weight-space row spaces ------------------------------------------- #
    print("\n  right singular vectors of the two weight updates ...", flush=True)
    sv_report = {"delta_acquire": {}, "cir_edit": {}}
    for li in layers:
        D = C.dense(acq, li).weight.detach().float().cpu() - base_dense[li]
        V, S = C.top_right_singular(D, args.n_pcs)
        bases["delta_acquire"][li] = V.to(device)
        sv_report["delta_acquire"][int(li)] = C.rank_stats(S)

    from src.models.load import load_checkpoint
    if (arm.cir_b / "config.json").exists():
        cirb = load_checkpoint(str(arm.cir_b), device="cpu")
        for li in layers:
            D = C.dense(cirb, li).weight.detach().float().cpu() - base_dense[li]
            V, S = C.top_right_singular(D, args.n_pcs)
            bases["cir_edit"][li] = V.to(device)
            sv_report["cir_edit"][int(li)] = C.rank_stats(S)
        del cirb
    else:
        print(f"  WARN: no CIR checkpoint at {arm.cir_b}; cir_edit bases skipped")
        bases.pop("cir_edit")

    # --- generator directions under theta_acq ------------------------------ #
    print(f"\n  gradient moments on {test_key} ...", flush=True)
    mom_test = C.act_grad_moments(acq, tok, layers, f_test, device=device,
                                  batch_size=args.grad_batch, seed=args.seed,
                                  max_seqs=args.grad_seqs)
    print(f"  gradient moments on F_train_seen ...", flush=True)
    mom_train = C.act_grad_moments(acq, tok, layers, f_train, device=device,
                                   batch_size=args.grad_batch, seed=args.seed,
                                   max_seqs=args.grad_seqs)
    for li in layers:
        bases["gen_test"][li], _ = C.top_eigvecs(mom_test[li]["grad_second_moment"], args.n_pcs)
        bases["gen_train"][li], _ = C.top_eigvecs(mom_train[li]["grad_second_moment"], args.n_pcs)
    del acq
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # orthonormalise defensively (QR-derived bases already are; eigh/svd ones too)
    for name in list(bases):
        for li in layers:
            bases[name][li] = C.orth_rows(bases[name][li].float())

    npz = {f"{name}|{li}": v.cpu().numpy() for name, d in bases.items() for li, v in d.items()}
    np.savez_compressed(arm.results / "bases.npz", **npz)
    print(f"  wrote {arm.results / 'bases.npz'} ({len(npz)} bases)")
    return bases, layers, dim, sv_report, test_key


def run_arm(arm: C.Arm, args, device: str) -> dict:
    C.banner(f"T3 — {arm.label}")
    bases, layers, dim, sv_report, test_key = build_bases(arm, args, device)
    names = [n for n in BASIS_ORDER if n in bases]

    per_layer, aggregate = {}, {}
    for a, b in itertools.combinations(names, 2):
        key = f"{a}|{b}"
        rows = {}
        for li in layers:
            rows[int(li)] = C.overlap(bases[a][li], bases[b][li])
        per_layer[key] = rows
        vals = list(rows.values())
        aggregate[key] = {
            "energy_X_in_Y": sum(v["energy_X_in_Y"] for v in vals) / len(vals),
            "energy_Y_in_X": sum(v["energy_Y_in_X"] for v in vals) / len(vals),
            "ratio_over_chance": sum(v["ratio_over_chance"] for v in vals) / len(vals),
            "mean_cos_principal_angle": sum(v["mean_cos_principal_angle"] for v in vals) / len(vals),
            "n_angles_ge_0.9": sum(v["n_angles_ge_0.9"] for v in vals) / len(vals),
        }

    print(f"\n  {'pair':<44}{'O(X in Y)':>11}{'x chance':>10}{'mean cos':>10}")
    for a, b in HEADLINE_PAIRS:
        key = f"{a}|{b}"
        if key not in aggregate:
            key = f"{b}|{a}"
        if key not in aggregate:
            continue
        v = aggregate[key]
        print(f"  {key:<44}{v['energy_X_in_Y']:>11.4f}{v['ratio_over_chance']:>10.2f}"
              f"{v['mean_cos_principal_angle']:>10.3f}")

    rec = {
        "arm": arm.key, "label": arm.label, "kind": arm.kind,
        "primary_test_key": test_key, "layers": [int(li) for li in layers],
        "dim": int(dim), "n_pcs": args.n_pcs, "chance_overlap": args.n_pcs / dim,
        "cir_checkpoint": str(arm.cir_b), "cir_label": arm.cir_label,
        "cir_forget_ratio": arm.cir_forget_ratio,
        "singular_spectra": sv_report,
        "aggregate": aggregate, "per_layer": per_layer,
        "published_removable_topk": arm.cir_overlap().get("aggregate", {}).get("removable_topk"),
    }
    C.save_json(rec, arm.results / "t3_overlaps.json")
    return rec


def main() -> int:
    args = parse_args()
    device = C.device_arg(args.device)
    C.banner(f"TEST 3 — CIR TARGET vs Delta_acquire vs F_test GENERATOR   (device={device})")
    out = {k: run_arm(C.get_arm(k), args, device) for k in args.arms}

    C.banner("T3 SUMMARY — overlap as a multiple of chance")
    print(f"{'pair':<44}" + "".join(f"{k:>12}" for k in out))
    for a, b in HEADLINE_PAIRS:
        cells = []
        for r in out.values():
            # keys are stored in BASIS_ORDER order, which is not the order HEADLINE_PAIRS
            # names them in — look both ways or the row silently prints as missing
            v = r["aggregate"].get(f"{a}|{b}") or r["aggregate"].get(f"{b}|{a}")
            cells.append(f"{v['ratio_over_chance']:>12.2f}" if v else f"{'-':>12}")
        print(f"{a}|{b}"[:43].ljust(44) + "".join(cells))
    C.save_json(out, C.RESULTS / "t3_overlaps.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
