"""Test 2 — what Delta_acquire = theta_acq - theta_0 is, and whether it carries F_test.

Two halves.

(a) DESCRIPTIVE. Per 2-D weight matrix, the singular spectrum of Delta_acquire and its
    rank descriptors (effective rank exp(H), participation ratio, stable rank, top-k
    energy fractions), plus the relative displacement ||Delta||_F / ||theta_0||_F. The
    prediction is that the OOD arm's update is both LARGER and HIGHER-rank: it has to
    build structure ESM-2 does not carry, whereas the in-distribution arm only has to
    re-weight structure that is already there.

(b) CAUSAL. Rank-truncate the update and re-score:

        remove-top-k :  W = W_0 + (Delta - sum_{i<k} s_i u_i v_i^T)
        keep-top-k   :  W = W_0 + sum_{i<k} s_i u_i v_i^T

    applied simultaneously to every 2-D weight, then PPPL on F_train_seen and on
    decontaminated F_test. The reported quantity is the *fraction of the acquisition gain
    destroyed*,

        destroyed(set, k) = (PPPL_k[set] - PPPL_acq[set]) / (PPPL_base[set] - PPPL_acq[set])

    so the two corpora are on a common 0..1 scale (0 = the edit cost nothing, 1 = the edit
    undid acquisition entirely on that corpus) regardless of their different absolute
    PPPLs. The dissociation shows up as destroyed(F_train) >> destroyed(F_test) for the
    OOD arm and destroyed(F_train) ~ destroyed(F_test) for the in-distribution arm: the
    top of the update is memorisation in the first case and the generator in the second.

    ``coupling`` = destroyed(F_test)/destroyed(F_train_seen) at each k is the single
    number this test contributes; a value near 1 means "the distinctive footprint IS the
    generator", near 0 means they dissociate.

    python experiments/4_footprint_vs_generator/t2_delta_rank.py --arms pfam mgnify

One GPU, ~45 min per arm (SVD is cheap; the PPPL evals dominate).
Writes results/<arm>/t2_delta_rank.json.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict

import common as C

DEFAULT_KS = [1, 2, 4, 8, 16, 32, 64]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    C.arm_args(p)
    p.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS,
                   help="rank-truncation depths (applied to every 2-D weight)")
    p.add_argument("--modes", nargs="+", default=["remove", "keep"], choices=["remove", "keep"])
    return p.parse_args()


def param_group(name: str) -> str:
    """Coarse bucket for reporting: which part of the network the update lands in."""
    if ".encoder.layer." not in name:
        return "embeddings/head"
    for tag in ("attention.self.query", "attention.self.key", "attention.self.value",
                "attention.output.dense", "intermediate.dense", "output.dense"):
        if tag in name:
            return tag
    return "other"


def layer_of(name: str):
    parts = name.split(".")
    if "layer" in parts:
        i = parts.index("layer")
        if i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def run_arm(arm: C.Arm, args, device: str) -> dict:
    import torch

    C.banner(f"T2 — {arm.label}")
    tok = C.tokenizer()
    sets = C.eval_sets(arm, args.n_eval, seed=args.seed)
    test_key = C.primary_test_key(sets)
    score_sets = {k: sets[k] for k in ("F_train_seen", test_key, "retain_holdout")}
    print(f"  scoring on {list(score_sets)} (primary held-out = {test_key})")

    base = C.load_base(device)
    print("\n  theta_0:")
    ppl_base = C.ppl_all(base, tok, score_sets, device, prefix="base ")

    acq = C.load_acquired(arm, device)
    print("\n  theta_acq:")
    ppl_acq = C.ppl_all(acq, tok, score_sets, device, prefix="acq  ")

    # ---- (a) spectra of Delta_acquire ------------------------------------- #
    print("\n  SVD of Delta_acquire over all 2-D weights ...", flush=True)
    base_params = {n: p.detach().float().cpu().clone() for n, p in base.named_parameters() if p.ndim == 2}
    del base
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    per_matrix, svd = {}, {}
    for name, p in acq.named_parameters():
        if p.ndim != 2 or name not in base_params:
            continue
        W0 = base_params[name]
        D = p.detach().float().cpu() - W0
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        svd[name] = (U, S, Vh)
        st = C.rank_stats(S)
        st["rel_displacement"] = float(D.norm() / (W0.norm() + 1e-12))
        st["group"] = param_group(name)
        st["layer"] = layer_of(name)
        per_matrix[name] = st

    def wmean(key, names=None):
        """||Delta||_F-weighted mean, skipping matrices with no update to describe."""
        names = names or list(per_matrix)
        pairs = [(per_matrix[n][key], per_matrix[n]["fro_norm"]) for n in names]
        pairs = [(v, w) for v, w in pairs if w > 0 and isinstance(v, float) and math.isfinite(v)]
        tot = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / tot if tot > 0 else float("nan")

    in_band = [n for n in per_matrix if per_matrix[n]["layer"] in C.LAYER_BAND
               and per_matrix[n]["group"] == "output.dense"]
    aggregate = {
        "n_matrices": len(per_matrix),
        "erank_frac_weighted": wmean("erank_frac"),
        "erank_weighted": wmean("erank"),
        "participation_ratio_weighted": wmean("participation_ratio"),
        "stable_rank_weighted": wmean("stable_rank"),
        "top24_energy_frac_weighted": wmean("top24_energy_frac"),
        "rel_displacement_total": float(
            torch.tensor([per_matrix[n]["fro_norm"] for n in per_matrix]).norm()
            / torch.tensor([base_params[n].norm() for n in per_matrix]).norm()),
        "cir_band_output_dense": {
            "erank_frac_weighted": wmean("erank_frac", in_band),
            "top24_energy_frac_weighted": wmean("top24_energy_frac", in_band),
        } if in_band else {},
    }
    by_group = defaultdict(list)
    for n, st in per_matrix.items():
        by_group[st["group"]].append(n)
    aggregate["by_group"] = {
        g: {"erank_frac_weighted": wmean("erank_frac", ns),
            "top24_energy_frac_weighted": wmean("top24_energy_frac", ns),
            "n": len(ns)}
        for g, ns in by_group.items()}

    print(f"  weighted effective-rank fraction = {aggregate['erank_frac_weighted']:.4f}"
          f"   top-24 energy = {aggregate['top24_energy_frac_weighted']:.4f}"
          f"   rel ||Delta|| = {aggregate['rel_displacement_total']:.4f}")

    # ---- (b) causal rank truncation --------------------------------------- #
    named = {n: p for n, p in acq.named_parameters() if n in svd}
    acq_state = {n: p.detach().clone() for n, p in named.items()}

    def set_truncated(mode: str, k: int):
        with torch.no_grad():
            for n, p in named.items():
                U, S, Vh = svd[n]
                kk = min(k, S.numel())
                top = (U[:, :kk] * S[:kk]) @ Vh[:kk]
                W0 = base_params[n]
                new = (W0 + top) if mode == "keep" else (acq_state[n].float().cpu() - top)
                p.data.copy_(new.to(p.dtype).to(p.device))

    def restore():
        with torch.no_grad():
            for n, p in named.items():
                p.data.copy_(acq_state[n])

    curves = {}
    for mode in args.modes:
        curves[mode] = []
        for k in args.ks:
            set_truncated(mode, k)
            row = {"k": k}
            for name, seqs in score_sets.items():
                row[name] = C.ppl(acq, tok, seqs, device)
            restore()
            for name in score_sets:
                denom = ppl_base[name] - ppl_acq[name]
                row[f"destroyed_{name}"] = ((row[name] - ppl_acq[name]) / denom
                                            if abs(denom) > 1e-9 else float("nan"))
            dt, dtr = row[f"destroyed_{test_key}"], row["destroyed_F_train_seen"]
            row["coupling_test_over_train"] = dt / dtr if abs(dtr) > 1e-6 else float("nan")
            curves[mode].append(row)
            print(f"  {mode}-top-{k:<4} " + "  ".join(
                f"{n}={row[n]:.3f}(d={row['destroyed_' + n]:+.2f})" for n in score_sets)
                + f"  coupling={row['coupling_test_over_train']:.2f}", flush=True)

    rec = {
        "arm": arm.key, "label": arm.label, "kind": arm.kind,
        "primary_test_key": test_key,
        "ppl_base": ppl_base, "ppl_acq": ppl_acq,
        "delta_acquire": {"aggregate": aggregate, "per_matrix": per_matrix},
        "truncation": curves, "ks": args.ks,
    }
    C.save_json(rec, arm.results / "t2_delta_rank.json")
    return rec


def main() -> int:
    args = parse_args()
    device = C.device_arg(args.device)
    C.banner(f"TEST 2 — Delta_acquire RANK AND ITS COUPLING TO F_test   (device={device})")
    out = {k: run_arm(C.get_arm(k), args, device) for k in args.arms}

    # Quote the deepest truncation actually run, so the summary is meaningful whatever
    # --ks was passed (a hard-coded k prints NaN on any non-default grid).
    k_quote = max(args.ks)
    C.banner("T2 SUMMARY")
    print(f"{'arm':<10}{'erank frac':>12}{'top24 energy':>14}{'rel |D|':>10}"
          f"{f'destroyed train@k{k_quote}':>21}{f'destroyed test@k{k_quote}':>20}{'coupling':>10}")
    for k, r in out.items():
        agg = r["delta_acquire"]["aggregate"]
        rows = [x for x in r["truncation"].get("remove", []) if x["k"] == k_quote]
        row = rows[0] if rows else {}
        tk = r["primary_test_key"]
        print(f"{k:<10}{agg['erank_frac_weighted']:>12.4f}{agg['top24_energy_frac_weighted']:>14.4f}"
              f"{agg['rel_displacement_total']:>10.4f}"
              f"{row.get('destroyed_F_train_seen', float('nan')):>21.3f}"
              f"{row.get('destroyed_' + tk, float('nan')):>20.3f}"
              f"{row.get('coupling_test_over_train', float('nan')):>10.2f}")
    C.save_json(out, C.RESULTS / "t2_delta_rank.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
