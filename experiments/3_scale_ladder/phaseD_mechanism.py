"""Phase D of the 650M spot-check — MECHANISM DIAGNOSTICS ACROSS THE LADDER.

This is the *priority* readout of the whole spot-check. A natural objection is that
M~=1 is a 35M artifact — capability so diffuse at small scale that the block sits in a
shallow basin. The rebuttal is geometric, not a single multiplier: if the same in-basin
signature holds at 8M/35M/150M/650M, the mechanism is scale-robust and the objection
does not apply.

Per scale, from {A = theta_0, B, A'}:

  * **Displacement**  ||theta - theta_0|| and ||theta - theta_0|| / ||theta_0|| per
    parameter group, for B and A'. The headline is the **B/A' ratio** — B moving *less*
    than a compute-matched control that never saw a safeguard objective (35M reference:
    2.17% vs 5.00%, B/A' = 0.43x) means the giant static forget is a thin perturbation.
  * **Per-layer profile** stored against **relative depth in [0,1]**
    (layer_index / (n_layers - 1)), never raw layer index, so 6/12/30/33-layer models
    overlay on one axis. This is the only place scale differences are reconciled, and it
    happens at reporting time — no branching on layer count anywhere.
  * **LMC barrier** on theta_alpha = (1-alpha) theta_0 + alpha theta_B for 21 alphas,
    scoring forget-family and benign pseudo-perplexity. Reported as the max upward
    deviation of the NLL curve from the straight endpoint chord, in nats (35M reference:
    0.30 nats, monotone, ~6% of the endpoint span). A monotone path with ~no ridge means
    B is linearly connected to base: the attacker can walk back without climbing.
  * **static_forget_x** = PPL_F(B) / PPL_F(A) at n = 0, for context: a huge static forget
    coexisting with a flat barrier is exactly the dormant-not-deleted signature.

    python experiments/3_scale_ladder/phaseD_mechanism.py --scales 35M 150M 650M

Writes ``<ladder_root>/manifests/mechanism_ladder.json``, which alone must be sufficient
to draw the ladder figure (Phase F reads nothing else).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import common
from common import (DEFAULT_LADDER_ROOT, HERE, banner, get_scale, manifest_dir,
                    relative_depth, run_root, sort_scales, upsert_row)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "scale_ladder.yaml"))
    p.add_argument("--scales", nargs="+", default=None, help="default: config scales that have B + A'")
    p.add_argument("--ladder-root", default=str(DEFAULT_LADDER_ROOT))
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-lmc", action="store_true", help="displacement only (seconds); skips the 21 alpha evals")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from omegaconf import OmegaConf

    from src.analysis.weightspace import l2_by_group, lmc_path, model_state_dict_cpu
    from src.data.datamodule import DataPaths, subsample, target_test_seqs
    from src.eval.pseudo_perplexity import pseudo_perplexity
    from src.eval.utility import benign_pseudo_perplexity
    from src.models.load import load_checkpoint, load_tokenizer
    from src.utils.io import fasta_seqs, save_json
    from src.utils.seeding import set_seed

    cfg = OmegaConf.load(args.config)
    m, mech = cfg.metrics, cfg.mechanism
    seed = int(cfg.seed)
    ladder_root = Path(args.ladder_root)
    manifest = manifest_dir(ladder_root) / "mechanism_ladder.json"
    n_alphas = int(mech.lmc_alphas_n)
    alphas = [i / (n_alphas - 1) for i in range(n_alphas)]

    scales = sort_scales(args.scales or list(cfg.scales))
    existing = {r["scale"] for r in common.load_manifest(manifest).get("rows", [])}

    for scale_name in scales:
        scale = get_scale(scale_name)
        root = run_root(ladder_root, scale_name)
        ckpt_B, ckpt_Ap = root / "checkpoints" / "B", root / "checkpoints" / "A_prime"
        if not ckpt_B.exists():
            print(f"[{scale_name}] no B checkpoint at {ckpt_B}; run Phase C first — skipping")
            continue
        if scale_name in existing and not args.force:
            print(f"[{scale_name}] already in {manifest.name}; skipping (--force to recompute)")
            continue
        banner(f"PHASE D — MECHANISM @ {scale_name}  ({scale.model_id})")
        set_seed(seed)

        paths = DataPaths(str(root / "data"))
        device, max_len = args.device, int(m.max_length)
        eval_bs, mpp, n_eval = int(m.eval_batch_size), int(m.masks_per_pass), int(m.max_eval_seqs)
        tok = load_tokenizer(scale.model_id)
        f_test = subsample(target_test_seqs(paths, "F"), n_eval, seed=seed)
        holdout = subsample(fasta_seqs(paths.retain_holdout), n_eval, seed=seed)

        # ---- displacement -------------------------------------------------- #
        base = load_checkpoint(scale.model_id, device=device)
        sd0 = model_state_dict_cpu(base)
        B = load_checkpoint(str(ckpt_B), device=device)
        sdB = model_state_dict_cpu(B)
        ppl_f_B = pseudo_perplexity(B, tok, f_test, device, eval_bs, max_len, masks_per_pass=mpp).ppl
        del B
        _free(torch)
        l2_B = l2_by_group(sd0, sdB)

        l2_Ap = None
        if bool(mech.include_aprime) and ckpt_Ap.exists():
            Ap = load_checkpoint(str(ckpt_Ap), device=device)
            l2_Ap = l2_by_group(sd0, model_state_dict_cpu(Ap))
            del Ap
            _free(torch)
        elif bool(mech.include_aprime):
            print(f"  WARNING: no A' at {ckpt_Ap} — B/A' displacement ratio will be null")

        disp_B = float(l2_B["__total__"]["rel"])
        disp_Ap = float(l2_Ap["__total__"]["rel"]) if l2_Ap else float("nan")
        ratio = disp_B / disp_Ap if l2_Ap and disp_Ap else None
        print(f"  relative ||dtheta||: B = {disp_B:.4f}" +
              (f"   A' = {disp_Ap:.4f}   (B/A' = {ratio:.2f}x)" if ratio else ""))

        # ---- per-layer profile on relative depth ---------------------------- #
        per_layer = build_per_layer(l2_B, l2_Ap, scale.num_layers)

        # ---- static forget --------------------------------------------------- #
        # theta_0's F PPL is the LMC path's alpha=0 point, so it is only scored separately
        # when the LMC walk is skipped.
        ppl_f_base = None
        if args.skip_lmc:
            ppl_f_base = pseudo_perplexity(base, tok, f_test, device, eval_bs, max_len,
                                           masks_per_pass=mpp).ppl

        # ---- LMC barrier ------------------------------------------------------ #
        lmc = None
        if not args.skip_lmc:
            def f_fn(mo):
                return pseudo_perplexity(mo, tok, f_test, device, eval_bs, max_len, masks_per_pass=mpp).ppl

            def benign_fn(mo):
                return benign_pseudo_perplexity(mo, tok, holdout, device, eval_bs, max_len, masks_per_pass=mpp).ppl

            print(f"  LMC theta_0 -> theta_B over {len(alphas)} alphas (F-test + benign PPL) ...")
            lmc = lmc_path(base, sd0, sdB, {"f": f_fn, "benign": benign_fn}, alphas)
            ppl_f_base = lmc["f_ppl"][0]
            print(f"  F PPL {lmc['f_ppl'][0]:.2f} -> {lmc['f_ppl'][-1]:.1f}   "
                  f"barrier = {lmc['f_barrier_nats']:.3f} nats   monotonic = {lmc['f_monotonic']}")
            print(f"  benign  {lmc['benign_ppl'][0]:.2f} -> {lmc['benign_ppl'][-1]:.2f}   "
                  f"barrier = {lmc['benign_barrier_nats']:.3f} nats")
        del base, sd0, sdB
        _free(torch)

        static_forget_x = (ppl_f_B / ppl_f_base) if ppl_f_base else None
        span = abs(math.log(lmc["f_ppl"][-1]) - math.log(lmc["f_ppl"][0])) if lmc else None

        row = {
            "scale": scale_name, "model_id": scale.model_id, "n_params": scale.n_params,
            "num_layers": scale.num_layers, "run_root": str(root),
            "disp_B_rel": disp_B, "disp_Aprime_rel": disp_Ap if l2_Ap else None,
            "disp_B_abs": float(l2_B["__total__"]["l2"]),
            "disp_Aprime_abs": float(l2_Ap["__total__"]["l2"]) if l2_Ap else None,
            "disp_ratio": ratio,
            "lmc_barrier_nats": float(lmc["f_barrier_nats"]) if lmc else None,
            "lmc_monotone": bool(lmc["f_monotonic"]) if lmc else None,
            "lmc_benign_barrier_nats": float(lmc["benign_barrier_nats"]) if lmc else None,
            "lmc_endpoint_span_nats": span,
            "lmc_barrier_frac_of_span": (float(lmc["f_barrier_nats"]) / span) if (lmc and span) else None,
            "forget_ppl_base": ppl_f_base, "forget_ppl_B": ppl_f_B,
            "static_forget_x": static_forget_x,
            "random_ceiling": float(m.random_ceiling),
            "per_layer": per_layer,
            "alphas": alphas if lmc else None,
            "n_eval_seqs": len(f_test),
        }
        upsert_row(manifest, row, meta={"per_layer_axis": str(mech.per_layer_axis),
                                        "lmc_alphas_n": n_alphas, "seed": seed})
        if lmc:
            save_json(lmc, str(root / "results" / "scale_ladder_lmc.json"))
        print(f"  -> row written to {manifest}")

    banner("PHASE D SUMMARY — the ladder")
    print("  {:>6} {:>8} {:>8} {:>7} {:>9} {:>5} {:>11}".format(
        "scale", "disp B", "disp A'", "B/A'", "LMC nats", "mono", "static fx"))
    for r in common.load_manifest(manifest).get("rows", []):
        print(f"  {r['scale']:>6} {_f(r.get('disp_B_rel'),4):>8} {_f(r.get('disp_Aprime_rel'),4):>8} "
              f"{_f(r.get('disp_ratio'),2):>7} {_f(r.get('lmc_barrier_nats'),3):>9} "
              f"{str(r.get('lmc_monotone')):>5} {_f(r.get('static_forget_x'),1):>11}")
    print(f"\nmanifest: {manifest}")
    print("  Read: a flat-or-declining B/A' ratio and barrier with scale => the in-basin "
          "geometry that explains M~=1 is not a small-model artifact.")


def build_per_layer(l2_B: dict, l2_Ap, num_layers: int) -> list:
    """Per-encoder-layer displacement keyed by relative depth, plus the non-layer groups.

    Relative depth is the *only* cross-scale reconciliation in this experiment: layer i of
    an n-layer model maps to i/(n-1), so a 6-layer and a 33-layer profile share an x-axis.
    """
    out = []
    for group, entry in sorted(l2_B.items()):
        if group == "__total__":
            continue
        li = int(group.split("_")[1]) if group.startswith("layer_") else None
        ap = (l2_Ap or {}).get(group)
        out.append({
            "group": group,
            "layer_index": li,
            "relative_depth": relative_depth(li, num_layers) if li is not None else None,
            "B_rel": float(entry["rel"]), "B_l2": float(entry["l2"]),
            "Aprime_rel": float(ap["rel"]) if ap else None,
            "Aprime_l2": float(ap["l2"]) if ap else None,
            "ratio": (float(entry["rel"]) / float(ap["rel"])) if ap and float(ap["rel"]) else None,
            "n_params": int(entry["n_params"]),
        })
    return out


def _free(torch):
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


if __name__ == "__main__":
    main()
