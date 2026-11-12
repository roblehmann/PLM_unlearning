"""The independent F-distinctive-vs-R-shared *activation* basis, built on a frozen model
via a single GPU pass. Shared by Phase 8c (defender displacement θ_B − θ₀) and Phase 8d
(attacker repair-delta θ_attacked − θ_B), so both decompositions use the SAME referee
subspace — the one CIR did not get to choose.

Per targeted ``output.dense`` module we take the down-proj *input* activations over R and
over F, PCA each, and return two orthonormal row-bases in that input space:

    R_shared      : top PCs of the retain activations (the general/shared directions),
    F_distinctive : top PCs of the forget activations with span(R_shared) removed and
                    re-orthonormalised — the directions the family uses that general
                    proteins do not. Orthogonal to R_shared by construction.
"""
from __future__ import annotations

from typing import Dict, Sequence


def _act_covs(base, tokenizer, layers, corpus, *, pc_batch, max_length, seed, device):
    """Streaming covariance of the down-proj *input* activations at each targeted layer,
    over ``corpus`` (special/pad tokens excluded). Returns ``{layer_idx: _Cov}``."""
    import torch

    from src.train.cir import _Cov, _dense, make_plain_loader, special_id_set

    special = special_id_set(tokenizer)
    cap: dict = {}
    handles = [
        _dense(base, li).register_forward_hook(
            (lambda li_: (lambda _m, inp, _o: cap.__setitem__(li_, inp[0].detach())))(li)
        )
        for li in layers
    ]
    inter = _dense(base, layers[0]).in_features
    cov = {li: _Cov(inter, device, dtype=torch.float32) for li in layers}
    loader = make_plain_loader(corpus, tokenizer, pc_batch, max_length, shuffle=False, seed=seed)
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            m = attn.bool()
            for sid in special:
                m = m & (ids != sid)
            if m.sum() == 0:
                continue
            cap.clear()
            base(input_ids=ids, attention_mask=attn)
            for li in layers:
                cov[li].add(cap[li][m])
    for h in handles:
        h.remove()
    return cov


def subspace_overlap(
    base,
    tokenizer,
    layers: Sequence[int],
    retain_seqs: Sequence[str],
    forget_seqs: Sequence[str],
    *,
    n_pcs: int = 24,
    corpus_seqs: int = 2048,
    pc_batch: int = 16,
    max_length: int = 1024,
    seed: int = 0,
    device: str = None,
) -> Dict:
    """O(F,R): how much of F's representational variance lives in R's top-``n_pcs``
    activation subspace, per targeted layer, via principal angles on the two activation
    PCAs. High overlap ⇒ there is little family signal to remove that is not also shared
    with general proteins (leaving CIR little to act on), and predicts a shared-subspace attacker repair.

    Per layer returns:
      * ``cos_principal_angles`` — cos θ_i between span(V_F) and span(V_R) (1 = parallel),
      * ``O_topk`` — energy-weighted fraction of F's top-k PC variance captured by span(V_R),
      * ``O_var`` — fraction of F's *total* activation variance captured by span(V_R),
      * ``removable_topk`` = 1 − O_topk — the F-distinctive fraction there is *to* remove.
    Plus an aggregate mean over layers.
    """
    import torch

    from src.data.datamodule import subsample
    from src.train.cir import _top_k_basis

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    R = subsample(retain_seqs, corpus_seqs, seed=seed)
    F = subsample(forget_seqs, corpus_seqs, seed=seed + 7)
    covR = _act_covs(base, tokenizer, layers, R, pc_batch=pc_batch, max_length=max_length, seed=seed, device=device)
    covF = _act_covs(base, tokenizer, layers, F, pc_batch=pc_batch, max_length=max_length, seed=seed, device=device)

    per_layer: Dict[int, dict] = {}
    for li in layers:
        cR, _ = covR[li].cov_mean()
        cF, _ = covF[li].cov_mean()
        Vr, _ = _top_k_basis(cR, None, n_pcs)                 # [k, in]
        Vf, wf = _top_k_basis(cF, None, n_pcs)                # [k, in], weights λ^F_i
        cos = torch.linalg.svdvals(Vr @ Vf.t()).clamp(0, 1)   # principal-angle cosines
        proj_sq = (Vf @ Vr.t()).pow(2).sum(1)                 # ‖P_R v^F_i‖² per F PC
        w = wf.clamp(min=0)
        O_topk = float((w * proj_sq).sum() / (w.sum() + 1e-12))
        O_var = float(((Vr @ cF) * Vr).sum() / (torch.diagonal(cF).sum() + 1e-12))
        per_layer[int(li)] = {
            "cos_principal_angles": [round(float(c), 4) for c in cos],
            "O_topk": O_topk,
            "O_var": O_var,
            "removable_topk": 1.0 - O_topk,
            "n_angles_ge_0.9": int((cos >= 0.9).sum()),
        }
    ov = [per_layer[int(li)] for li in layers]
    agg = {
        "O_topk": sum(d["O_topk"] for d in ov) / len(ov),
        "O_var": sum(d["O_var"] for d in ov) / len(ov),
        "removable_topk": sum(d["removable_topk"] for d in ov) / len(ov),
        "mean_cos_principal_angle": sum(sum(d["cos_principal_angles"]) / len(d["cos_principal_angles"]) for d in ov) / len(ov),
    }
    return {"n_pcs": n_pcs, "per_layer": per_layer, "aggregate": agg}


def build_fvsr_bases(
    base,
    tokenizer,
    layers: Sequence[int],
    retain_seqs: Sequence[str],
    forget_seqs: Sequence[str],
    *,
    n_pcs: int = 24,
    corpus_seqs: int = 2048,
    pc_batch: int = 16,
    max_length: int = 1024,
    seed: int = 0,
    device: str = None,
    tol: float = 1e-4,
) -> Dict[int, dict]:
    """Return ``{layer_idx: {"F_distinctive": [k_f, in], "R_shared": [k_r, in]}}`` as CPU
    numpy arrays (orthonormal rows, mutually orthogonal across the two)."""
    import torch

    from src.data.datamodule import subsample
    from src.train.cir import _top_k_basis

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    R = subsample(retain_seqs, corpus_seqs, seed=seed)
    F = subsample(forget_seqs, corpus_seqs, seed=seed + 7)
    covR = _act_covs(base, tokenizer, layers, R, pc_batch=pc_batch, max_length=max_length, seed=seed, device=device)
    covF = _act_covs(base, tokenizer, layers, F, pc_batch=pc_batch, max_length=max_length, seed=seed, device=device)

    bases: Dict[int, dict] = {}
    for li in layers:
        cR, _ = covR[li].cov_mean()
        cF, _ = covF[li].cov_mean()
        Vr, _ = _top_k_basis(cR, None, n_pcs)                 # [k_r, in]
        Vf, _ = _top_k_basis(cF, None, n_pcs)                 # [k_f, in]
        M = Vf.t() - Vr.t() @ (Vr @ Vf.t())                   # F PCs, R-span removed  [in, k_f]
        Q, Rq = torch.linalg.qr(M)
        keep = torch.abs(torch.diagonal(Rq)) > tol            # drop dirs fully inside span(V_R)
        Uf = Q[:, keep].t().contiguous()                      # [k_f', in]  (⟂ V_R)
        bases[int(li)] = {"F_distinctive": Uf.cpu().numpy(), "R_shared": Vr.cpu().numpy()}
    return bases
