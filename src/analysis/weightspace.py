"""Weight-space mechanism analysis for the blocked model B.

Two probes that explain *why* a large static forget has ~no adaptation-cost durability:

1. **Per-layer L2** — ‖θ_B − θ₀‖ per layer (absolute and relative to ‖θ₀‖). If B sits
   close to base — and no farther than the compute-matched control A′ — the giant static
   forget is a thin perturbation, not a deep relocation in weight space.
2. **Linear-mode-connectivity (LMC) barrier** — interpolate θ₀ → θ_B and score F-test (and
   benign) pseudo-perplexity along the path. A ~flat / monotone path with **no barrier**
   means B is linearly connected to base: you can walk back toward base without crossing a
   loss ridge, so the block is trivially reversible.

The tensor ops import torch lazily so the module imports in a minimal environment; tests
use ``importorskip('torch')``.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Dict, List, Sequence

_LAYER_RE = re.compile(r"encoder\.layer\.(\d+)\.")


def param_layer_group(name: str) -> str:
    """Map a parameter name to a coarse group for per-layer aggregation."""
    m = _LAYER_RE.search(name)
    if m:
        return f"layer_{int(m.group(1)):02d}"
    if "embeddings" in name:
        return "embeddings"
    if "lm_head" in name:
        return "lm_head"
    if "contact_head" in name:
        return "contact_head"
    return "other"


def model_state_dict_cpu(model) -> Dict:
    """A detached CPU copy of a model's state dict (kept off-GPU to save memory)."""
    return {k: v.detach().to("cpu") for k, v in model.state_dict().items()}


def l2_by_group(sd0: Dict, sd1: Dict) -> Dict[str, dict]:
    """Per-group ‖θ1 − θ0‖₂ and relative ‖Δ‖/‖θ0‖ over shared float tensors, plus a
    ``__total__`` entry."""
    import torch

    acc: Dict[str, List[float]] = {}  # group -> [sq_delta, sq_base, n_params]
    for k, p0 in sd0.items():
        if k not in sd1 or not torch.is_floating_point(p0):
            continue
        p0f, p1f = p0.float(), sd1[k].float()
        d = p1f - p0f
        g = param_layer_group(k)
        a = acc.setdefault(g, [0.0, 0.0, 0])
        a[0] += float((d * d).sum().item())
        a[1] += float((p0f * p0f).sum().item())
        a[2] += int(p0.numel())

    out: Dict[str, dict] = {}
    tot_d = tot_b = 0.0
    tot_n = 0
    for g, (sd, sb, n) in acc.items():
        out[g] = {"l2": math.sqrt(sd), "rel": (math.sqrt(sd / sb) if sb > 0 else float("nan")), "n_params": n}
        tot_d += sd
        tot_b += sb
        tot_n += n
    out["__total__"] = {
        "l2": math.sqrt(tot_d),
        "rel": (math.sqrt(tot_d / tot_b) if tot_b > 0 else float("nan")),
        "n_params": tot_n,
    }
    return out


def projected_l2_by_group(
    sd0: Dict,
    sd1: Dict,
    bases: Dict[int, dict],
    weight_key_fmt: str = "esm.encoder.layer.{li}.output.dense.weight",
) -> Dict[str, dict]:
    """Split the realized weight delta ΔW = θ1 − θ0 of each CIR-targeted
    ``output.dense.weight`` into the **kept** vs **collapsed** subspaces and report each
    Frobenius norm (+ a ``__total__``).

    CIR's update is only allowed to use directions outside the collapsed common subspace,
    so genuine family-specific deletion should show up as a large **kept** displacement.
    A delta that is small in the kept subspace (no farther than the control A′ projected
    the same way) while LMC-connected to base = dormant-not-deleted under a new name.

    ``bases[li]`` provides the *collapsed* directions: ``act_basis`` [k_a, in_features]
    (input/column space) and ``grad_basis`` [k_g, out_features] (output/row space), with
    orthonormal rows. The kept component is ``P_g · ΔW · P_a`` with ``P = I − Vᵀ V``.
    """
    import torch

    out: Dict[str, dict] = {}
    tot_keep = tot_coll = tot_tot = 0.0
    for li, b in bases.items():
        key = weight_key_fmt.format(li=int(li))
        if key not in sd0 or key not in sd1:
            continue
        dW = (sd1[key].float() - sd0[key].float())                       # [out, in]
        Va = torch.as_tensor(b["act_basis"], dtype=torch.float32)        # [k_a, in]
        Vg = torch.as_tensor(b["grad_basis"], dtype=torch.float32)       # [k_g, out]
        keep = dW
        if Vg.numel() and Vg.shape[0] > 0:
            keep = keep - Vg.t() @ (Vg @ keep)                           # drop collapsed rows
        if Va.numel() and Va.shape[0] > 0:
            keep = keep - (keep @ Va.t()) @ Va                           # drop collapsed cols
        kept = float((keep * keep).sum().item())
        total = float((dW * dW).sum().item())
        coll = max(total - kept, 0.0)
        out[f"layer_{int(li):02d}"] = {
            "kept_l2": math.sqrt(kept),
            "collapsed_l2": math.sqrt(coll),
            "total_l2": math.sqrt(total),
        }
        tot_keep += kept
        tot_coll += coll
        tot_tot += total
    out["__total__"] = {
        "kept_l2": math.sqrt(tot_keep),
        "collapsed_l2": math.sqrt(tot_coll),
        "total_l2": math.sqrt(tot_tot),
    }
    return out


def fvsr_projected_l2(
    sd0: Dict,
    sd1: Dict,
    fvsr_bases: Dict[int, dict],
    weight_key_fmt: str = "esm.encoder.layer.{li}.output.dense.weight",
) -> Dict[str, dict]:
    """Decompose ΔW = θ1 − θ0 of each targeted ``output.dense.weight`` onto an
    **independent** F-distinctive-vs-R-shared *activation* basis (input/column space),
    NOT CIR's own collapse basis. This closes the CIR advocate's escape hatch: global L2
    is dismissible for a method claiming surgical movement, so we ask, in a referee's
    subspace, *how much of B's displacement lands in the directions the forget family uses
    that general proteins do not* (``F_distinctive``) vs the general/shared directions
    (``R_shared``) vs everything else (``rest``, low-variance for both).

    ``fvsr_bases[li]`` holds orthonormal rows in the down-proj input space:
    ``F_distinctive`` [k_f, in] (span of F's activation PCs orthogonalised against R's) and
    ``R_shared`` [k_r, in] (R's activation PCs). The two are mutually orthogonal by
    construction, so the three squared components add up to ‖ΔW‖². Projecting ΔW's columns
    onto an orthonormal basis U is ``‖ΔW · Uᵀ‖`` (U·Uᵀ = I)."""
    import torch

    out: Dict[str, dict] = {}
    tot_f = tot_r = tot_tot = 0.0
    for li, b in fvsr_bases.items():
        key = weight_key_fmt.format(li=int(li))
        if key not in sd0 or key not in sd1:
            continue
        dW = (sd1[key].float() - sd0[key].float())                       # [out, in]
        Uf = torch.as_tensor(b["F_distinctive"], dtype=torch.float32)    # [k_f, in]
        Vr = torch.as_tensor(b["R_shared"], dtype=torch.float32)         # [k_r, in]
        f_sq = float((dW @ Uf.t()).pow(2).sum().item()) if Uf.numel() and Uf.shape[0] > 0 else 0.0
        r_sq = float((dW @ Vr.t()).pow(2).sum().item()) if Vr.numel() and Vr.shape[0] > 0 else 0.0
        total = float((dW * dW).sum().item())
        rest = max(total - f_sq - r_sq, 0.0)
        out[f"layer_{int(li):02d}"] = {
            "f_distinctive_l2": math.sqrt(f_sq),
            "r_shared_l2": math.sqrt(r_sq),
            "rest_l2": math.sqrt(rest),
            "total_l2": math.sqrt(total),
        }
        tot_f += f_sq
        tot_r += r_sq
        tot_tot += total
    out["__total__"] = {
        "f_distinctive_l2": math.sqrt(tot_f),
        "r_shared_l2": math.sqrt(tot_r),
        "rest_l2": math.sqrt(max(tot_tot - tot_f - tot_r, 0.0)),
        "total_l2": math.sqrt(tot_tot),
    }
    return out


def interpolate_state_dicts(sd0: Dict, sd1: Dict, alpha: float) -> Dict:
    """(1−α)·θ0 + α·θ1 for shared float tensors; non-float keys (int buffers) copied
    from θ0. Result keys == ``sd0`` keys, so it loads strict into the base architecture."""
    import torch

    out = {}
    for k, v0 in sd0.items():
        if k in sd1 and torch.is_floating_point(v0):
            out[k] = ((1.0 - alpha) * v0 + alpha * sd1[k].to(v0.dtype)).to(v0.dtype)
        else:
            out[k] = v0.clone()
    return out


def _barrier_nats(alphas: Sequence[float], ppls: Sequence[float]):
    """Max upward deviation (in nats) of the NLL path from the straight line joining its
    endpoints, plus a monotonicity flag. Endpoints contribute 0, so barrier ≥ 0; barrier
    ≈ 0 ⇒ no ridge between θ0 and θ_B."""
    nll = [math.log(p) for p in ppls]
    a0, a1, n0, n1 = alphas[0], alphas[-1], nll[0], nll[-1]
    span = (a1 - a0) or 1.0
    barrier = 0.0
    for a, n in zip(alphas, nll):
        lin = n0 + (a - a0) / span * (n1 - n0)
        barrier = max(barrier, n - lin)
    inc = all(nll[i] <= nll[i + 1] + 1e-9 for i in range(len(nll) - 1))
    dec = all(nll[i] >= nll[i + 1] - 1e-9 for i in range(len(nll) - 1))
    return barrier, bool(inc or dec)


def lmc_path(
    base_model,
    sd0: Dict,
    sd1: Dict,
    eval_fns: Dict[str, Callable],
    alphas: Sequence[float],
) -> Dict:
    """Walk θ0 → θ1 over ``alphas``; at each step load the interpolated weights into
    ``base_model`` (overwritten in place) and run every ``eval_fns[name](model) -> ppl``.
    Returns the curves plus per-metric barrier (nats) and monotonicity."""
    metrics: Dict[str, List[float]] = {name: [] for name in eval_fns}
    for a in alphas:
        base_model.load_state_dict(interpolate_state_dicts(sd0, sd1, float(a)))
        base_model.eval()
        for name, fn in eval_fns.items():
            metrics[name].append(float(fn(base_model)))

    out: Dict = {"alphas": [float(a) for a in alphas]}
    for name, curve in metrics.items():
        barrier, mono = _barrier_nats(out["alphas"], curve)
        out[f"{name}_ppl"] = curve
        out[f"{name}_barrier_nats"] = barrier
        out[f"{name}_monotonic"] = mono
    return out
