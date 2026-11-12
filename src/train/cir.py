"""CIR — Collapse of Irrelevant Representations (Sondej & Yang, 2025), ported as a
second safeguard alongside first-order MLAC/TAR ([src/train/taskblock.py]).

All arms still branch from the same base checkpoint θ0:
    B  = θ0 + this phase (CIR)
    A' = θ0 + plain MLM matched on FLOPs (src/train/mlm.py)        <- unchanged
    A  = θ0

CIR's theory of durable unlearning is *surgical selectivity*, not weight-space
displacement: an update is only allowed to use representational directions that are
**distinctive to the forget family**, so robustness comes from removing the *right*
directions rather than from moving far. Per targeted MLP module, per step:

  1. forward F + take module-output gradients of the MLP-breaking loss (Eq. 2);
  2. collapse a common subspace (mean + top-k PCs) out of BOTH the incoming
     activations and the module-output gradients (project onto the orthogonal
     complement; Eq. 1);
  3. weight update = einsum(pure_grads, pure_acts) — the activation×gradient outer
     product built from the *purified* vectors — normalised to a fixed update-norm
     (the effective unlearning rate);
  4. concurrently take a representation-preserving step on a retain batch.

THE ONE LOAD-BEARING ADAPTATION (do not change lightly). Stock
CIR estimates PCs on the (heterogeneous) forget corpus. For a *single-family* forget
set the family signal IS the top PCs of F, so collapsing them removes exactly what we
want to delete. We therefore estimate the collapse subspace on the **retain corpus R**
(``subspace_mode=retain_pca``, the headline Option A); the directions that survive in
the F vectors are the family-distinctive capability. ``discriminative`` (Option A′,
keep high-in-F/low-in-R) and ``forget_pca`` (Option B, stock CIR — an ablation /
negative control) are also implemented and pre-registered. R must be homology-disjoint
from F (the leakage filter already enforces this).

ESM-2 mapping (verified against transformers' modeling_esm.py): the targeted module is
``esm.encoder.layer[i].output.dense`` (the FFN down-projection). Its *input* is the
post-GELU activations; its *output* is the MLP output BEFORE the residual add — the
direct analog of CIR's MLP-output target. We never break at <cls> (BOS), <eos>, <pad>.
"""
from __future__ import annotations

import copy
import math
from typing import List, Optional, Sequence

from src.data.datamodule import SequenceDataset, cycle, subsample
from src.train.mlm import batch_tokens, cfg_get
from src.utils.flops import FlopCounter, train_flops
from src.utils.seeding import set_seed


# --------------------------------------------------------------------------- #
# Module / token helpers
# --------------------------------------------------------------------------- #
def _dense(model, layer_idx: int):
    """The FFN down-projection of block ``layer_idx`` (CIR's targeted MLP module)."""
    return model.esm.encoder.layer[layer_idx].output.dense


def resolve_layers(model, layer_band: Sequence[int]) -> List[int]:
    """Targeted layer indices, clamped to the model's depth. Falls back to a middle
    band (¼–¾) when ``layer_band`` is out of range (e.g. the 6-layer 8M debug model)."""
    n_layers = int(model.config.num_hidden_layers)
    layers = sorted({int(i) for i in layer_band if 0 <= int(i) < n_layers})
    if not layers:
        lo, hi = n_layers // 4, max(n_layers // 4 + 1, (3 * n_layers) // 4)
        layers = list(range(lo, hi))
    return layers


def special_id_set(tokenizer) -> set:
    """Token ids never broken / never used for PCA: <cls> (BOS), <eos>, <pad>, <mask>."""
    ids = {
        getattr(tokenizer, "cls_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "mask_token_id", None),
    }
    return {i for i in ids if i is not None}


def make_plain_loader(sequences, tokenizer, batch_size, max_length=1024, shuffle=False, seed=0):
    """Padding-only loader (NO MLM masking) — the 'all positions unmasked' forward path
    the breaking loss and the activation PCA want."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    ds = SequenceDataset(sequences, tokenizer, max_length=max_length)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collator, generator=g)


# --------------------------------------------------------------------------- #
# Streaming covariance + PCA (top-k eigenvectors of the centered covariance)
# --------------------------------------------------------------------------- #
class _Cov:
    """Streaming mean + uncentered second moment for PCA over many token vectors."""

    def __init__(self, dim: int, device, dtype):
        import torch

        self.dim = dim
        self.n = 0
        self.sum = torch.zeros(dim, device=device, dtype=dtype)
        self.xtx = torch.zeros(dim, dim, device=device, dtype=dtype)

    def add(self, X) -> None:  # X: [T, dim]
        if X.numel() == 0:
            return
        Xf = X.float()
        self.n += int(Xf.shape[0])
        self.sum += Xf.sum(0)
        self.xtx += Xf.t() @ Xf

    def cov_mean(self):
        import torch

        n = max(self.n, 1)
        mean = self.sum / n
        cov = self.xtx / n - torch.outer(mean, mean)
        return 0.5 * (cov + cov.t()), mean


def _top_k_basis(cov, mean, k: int):
    """Top-``k`` eigenvectors (largest eigenvalue) of ``cov`` as rows -> [k, dim],
    plus the eigenvalues. ``k==0`` -> empty basis (collapse the mean only)."""
    import torch

    dim = cov.shape[0]
    k = max(0, min(int(k), dim))
    if k == 0:
        return torch.zeros(0, dim, device=cov.device, dtype=cov.dtype), torch.zeros(0, device=cov.device)
    evals, evecs = torch.linalg.eigh(cov)          # ascending
    basis = evecs[:, -k:].t().contiguous()         # [k, dim], orthonormal rows
    return basis, evals[-k:]


def _discriminative_filter(basis, cov_R, cov_F, keep_ratio: float):
    """Option A′: from the R-derived candidate directions, COLLAPSE only the ones that
    are *also* high-variance under F (shared); KEEP directions that are much more
    variable in F than in R (family-distinctive). Returns the rows to collapse."""
    import torch

    if basis.shape[0] == 0:
        return basis
    keep_collapse = []
    for v in basis:
        var_R = float(v @ cov_R @ v)
        var_F = float(v @ cov_F @ v)
        if var_F <= keep_ratio * max(var_R, 1e-12):
            keep_collapse.append(v)                # shared -> collapse
    if not keep_collapse:
        return torch.zeros(0, basis.shape[1], device=basis.device, dtype=basis.dtype)
    return torch.stack(keep_collapse, 0)


def collapse(X, mean, basis, collapse_mean: bool):
    """Project ``X`` onto the orthogonal complement of (mean + span(basis))."""
    Xc = X - mean if collapse_mean else X
    if basis.shape[0] > 0:
        Xc = Xc - (Xc @ basis.t()) @ basis
    return Xc


# --------------------------------------------------------------------------- #
# Main trainer
# --------------------------------------------------------------------------- #
def train_cir(
    base_model,
    tokenizer,
    retain_seqs: Sequence[str],
    forget_seqs: Sequence[str],
    cfg,
    device: str,
    seed: int = 0,
    max_length: int = 1024,
    tracker=None,
    retain_eval_seqs: Optional[Sequence[str]] = None,
    eval_batch_size: int = 64,
    eval_masks_per_pass: int = 8,
    eval_max_seqs: int = 256,
) -> dict:
    """Run the CIR phase on ``base_model`` (θ0) -> B. Returns the model, a FLOP/
    wall-clock record (defender-side overhead, same schema as train_taskblock so
    scripts/04 + B_compute.json are unchanged), training history, and the per-layer
    collapse subspace bases (for the Phase-8b subspace-projected-displacement probe)."""
    import torch
    from torch.optim.lr_scheduler import LambdaLR

    from src.eval.pseudo_perplexity import pseudo_perplexity

    set_seed(seed)
    # ---- knobs ----
    subspace_mode = str(cfg_get(cfg, "subspace_mode", "retain_pca"))
    n_act_pcs = int(cfg_get(cfg, "n_act_pcs", 24))
    n_grad_pcs = int(cfg_get(cfg, "n_grad_pcs", 36))
    collapse_mean = bool(cfg_get(cfg, "collapse_mean", True))
    unlearn_rate = float(cfg_get(cfg, "unlearn_rate", 1e-2))
    layer_band = list(cfg_get(cfg, "layer_band", [3, 4, 5, 6, 7, 8]))
    eta_outer = float(cfg_get(cfg, "eta_outer", 5e-5))
    retain_batch = int(cfg_get(cfg, "retain_batch", 16))
    forget_batch = int(cfg_get(cfg, "forget_batch", 16))
    warmup_steps = int(cfg_get(cfg, "warmup_steps", 100))
    grad_clip = float(cfg_get(cfg, "grad_clip", 1.0))
    mlm_prob = float(cfg_get(cfg, "mlm_probability", 0.15))
    pc_corpus_seqs = int(cfg_get(cfg, "pc_corpus_seqs", 2048))
    pc_batch = int(cfg_get(cfg, "pc_batch", forget_batch))
    pc_act_policy = str(cfg_get(cfg, "pc_act_position_policy", "all_unmasked"))
    pc_grad_mask_draws = int(cfg_get(cfg, "pc_grad_mask_draws", 1))
    recompute_every_epoch = bool(cfg_get(cfg, "recompute_pcs_every_epoch", True))
    discriminative_keep_ratio = float(cfg_get(cfg, "discriminative_keep_ratio", 2.0))
    max_epochs = int(cfg_get(cfg, "max_epochs", 50))
    steps_per_epoch = int(cfg_get(cfg, "steps_per_epoch", 200))
    term_ratio = float(cfg_get(cfg, "terminate_retain_ppl_ratio", 1.001))
    log_every = int(cfg_get(cfg, "log_every", 25))
    eps = 1e-8

    theta = base_model.train().to(device)
    layers = resolve_layers(theta, layer_band)
    dtype = next(theta.parameters()).dtype
    theta_params = [p for p in theta.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in theta_params)

    # frozen reference θ0 (MLP_orig_out for the breaking loss; resid_orig for retain)
    theta_orig = copy.deepcopy(theta).to(device).eval()
    for p in theta_orig.parameters():
        p.requires_grad_(False)

    special = special_id_set(tokenizer)

    # ---- forward hooks on output.dense (acts = input, mlp_out = output) ----
    cap, cap_orig = {}, {}
    handles = []

    def _mk_hook(store, li, keep_graph):
        def hook(_mod, inp, out):
            store[li] = {"act": inp[0].detach(), "out": (out if keep_graph else out.detach())}
        return hook

    for li in layers:
        handles.append(_dense(theta, li).register_forward_hook(_mk_hook(cap, li, True)))
        handles.append(_dense(theta_orig, li).register_forward_hook(_mk_hook(cap_orig, li, False)))

    def _valid_mask(ids, attn):
        m = attn.bool()
        for sid in special:
            m = m & (ids != sid)
        return m

    def _breaking_loss(valid):
        """Eq. 2: mean over targeted layers of  sum_pos ReLU(MLP_out·MLP_orig_out) /
        (n_valid · avg‖MLP_out‖²), over non-special, non-BOS positions. Returns the
        scalar loss and the list of graph MLP-output tensors (for autograd.grad)."""
        vf = valid.float()
        cnt = vf.sum().clamp(min=1.0)
        total = 0.0
        mlp_outs = []
        for li in layers:
            out, out0 = cap[li]["out"], cap_orig[li]["out"]
            mlp_outs.append(out)
            dot = (out * out0).sum(-1)                              # [B, L]
            denom = (out.detach() ** 2).sum(-1)[valid].mean() + eps  # avg ‖MLP_out‖²
            total = total + (torch.relu(dot) * vf).sum() / (cnt * denom)
        return total / len(layers), mlp_outs

    counter = FlopCounter(n_params=n_params)
    plain_equiv_flops = 0.0

    # ----------------------------------------------------------------- #
    # Collapse-subspace estimation over a corpus (R and/or F)
    # ----------------------------------------------------------------- #
    def _accumulate(corpus):
        inter = _dense(theta, layers[0]).in_features
        hid = _dense(theta, layers[0]).out_features
        acc = {li: {"act": _Cov(inter, device, dtype=torch.float32),
                    "grad": _Cov(hid, device, dtype=torch.float32)} for li in layers}
        loader = make_plain_loader(corpus, tokenizer, pc_batch, max_length, shuffle=False, seed=seed)
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            valid = _valid_mask(ids, attn)
            if valid.sum() == 0:
                continue
            tok_n = batch_tokens(batch)
            # unmasked draw: activations (policy=all_unmasked) + gradients (draw 0)
            cap.clear(); cap_orig.clear()
            with torch.no_grad():
                theta_orig(input_ids=ids, attention_mask=attn)
            theta(input_ids=ids, attention_mask=attn)
            counter.record(tok_n, fwd_bwd=False)   # frozen ref forward
            loss, mlp_outs = _breaking_loss(valid)
            grads = torch.autograd.grad(loss, mlp_outs)
            counter.record(tok_n, fwd_bwd=True)    # θ forward + grad
            for j, li in enumerate(layers):
                if pc_act_policy == "all_unmasked":
                    acc[li]["act"].add(cap[li]["act"][valid])
                acc[li]["grad"].add(grads[j][valid])
            # extra masked draws stabilise the gradient PCA under sparse masking (MUDMAN)
            for _ in range(1, pc_grad_mask_draws):
                mids = ids.clone()
                pm = (torch.rand(mids.shape, device=device) < mlm_prob) & valid
                if tokenizer.mask_token_id is not None:
                    mids[pm] = tokenizer.mask_token_id
                cap.clear(); cap_orig.clear()
                with torch.no_grad():
                    theta_orig(input_ids=mids, attention_mask=attn)
                theta(input_ids=mids, attention_mask=attn)
                counter.record(tok_n, fwd_bwd=False)
                mloss, mouts = _breaking_loss(valid)
                mgrads = torch.autograd.grad(mloss, mouts)
                counter.record(tok_n, fwd_bwd=True)
                for j, li in enumerate(layers):
                    if pc_act_policy == "masked_only":
                        acc[li]["act"].add(cap[li]["act"][pm])
                    acc[li]["grad"].add(mgrads[j][valid])
        return acc

    def _estimate_bases():
        need_R = subspace_mode in ("retain_pca", "discriminative")
        need_F = subspace_mode in ("forget_pca", "discriminative")
        accR = _accumulate(subsample(retain_seqs, pc_corpus_seqs, seed=seed)) if need_R else None
        accF = _accumulate(subsample(forget_seqs, pc_corpus_seqs, seed=seed + 7)) if need_F else None
        src = accR if subspace_mode != "forget_pca" else accF
        bases = {}
        for li in layers:
            a_cov, a_mean = src[li]["act"].cov_mean()
            g_cov, g_mean = src[li]["grad"].cov_mean()
            a_basis, _ = _top_k_basis(a_cov, a_mean, n_act_pcs)
            g_basis, _ = _top_k_basis(g_cov, g_mean, n_grad_pcs)
            if subspace_mode == "discriminative":
                aR, _ = accR[li]["act"].cov_mean(); aF, _ = accF[li]["act"].cov_mean()
                gR, _ = accR[li]["grad"].cov_mean(); gF, _ = accF[li]["grad"].cov_mean()
                a_basis = _discriminative_filter(a_basis, aR, aF, discriminative_keep_ratio)
                g_basis = _discriminative_filter(g_basis, gR, gF, discriminative_keep_ratio)
            bases[li] = {"act_basis": a_basis, "act_mean": a_mean,
                         "grad_basis": g_basis, "grad_mean": g_mean}
        return bases

    # ----------------------------------------------------------------- #
    # The two per-step objectives
    # ----------------------------------------------------------------- #
    def _forget_update(batch, bases):
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        valid = _valid_mask(ids, attn)
        if valid.sum() == 0:
            return None
        tok_n = batch_tokens(batch)
        cap.clear(); cap_orig.clear()
        with torch.no_grad():
            theta_orig(input_ids=ids, attention_mask=attn)
        theta(input_ids=ids, attention_mask=attn)
        counter.record(tok_n, fwd_bwd=False)
        loss, mlp_outs = _breaking_loss(valid)
        grads = torch.autograd.grad(loss, mlp_outs)
        counter.record(tok_n, fwd_bwd=True)
        raws = {}
        total_sq = 0.0
        for j, li in enumerate(layers):
            A = cap[li]["act"][valid].float()                 # [T, inter]
            G = grads[j][valid].float()                       # [T, hid]
            pA = collapse(A, bases[li]["act_mean"], bases[li]["act_basis"], collapse_mean)
            pG = collapse(G, bases[li]["grad_mean"], bases[li]["grad_basis"], collapse_mean)
            dW = pG.t() @ pA                                   # [hid, inter] = dL/dW (purified)
            raws[li] = dW
            total_sq += float((dW * dW).sum())
        scale = unlearn_rate / (math.sqrt(total_sq) + eps)     # normalise total update-norm
        with torch.no_grad():
            for li in layers:
                _dense(theta, li).weight.add_(raws[li].to(dtype), alpha=-scale)
        return float(loss.item())

    optimizer = torch.optim.AdamW(theta_params, lr=eta_outer)
    scheduler = LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, warmup_steps)))

    def _retain_step(batch):
        nonlocal plain_equiv_flops
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        tok_n = batch_tokens(batch)
        vm = attn.bool().unsqueeze(-1).float()
        denom = vm.sum().clamp(min=1.0)
        optimizer.zero_grad(set_to_none=True)
        out = theta(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        with torch.no_grad():
            out0 = theta_orig(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        hs, hs0 = out.hidden_states, out0.hidden_states
        loss = 0.0
        for li in layers:
            h, h0 = hs[li + 1], hs0[li + 1]                    # residual-stream act after block li
            loss = loss + (((h - h0) ** 2) * vm).sum() / denom / h.shape[-1]
        loss = loss / len(layers)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(theta_params, grad_clip)
        optimizer.step()
        scheduler.step()
        counter.record(tok_n, fwd_bwd=True)                   # θ retain fwd+bwd
        counter.record(tok_n, fwd_bwd=False)                  # frozen ref forward
        counter.optimizer_steps += 1
        plain_equiv_flops += train_flops(n_params, tok_n)
        return float(loss.item())

    # ---- base retain pseudo-PPL (termination reference: 100.1% rule) ----
    eval_seqs = list(retain_eval_seqs) if retain_eval_seqs is not None \
        else subsample(retain_seqs, eval_max_seqs, seed=seed)
    base_retain_ppl = pseudo_perplexity(
        theta_orig, tokenizer, eval_seqs, device, eval_batch_size, max_length,
        masks_per_pass=eval_masks_per_pass).ppl
    theta.train()

    # Both objectives use natural (unmasked) sequences: the breaking loss operates on MLP
    # outputs and the retain loss preserves representations of real proteins. ``mlm_prob``
    # is only used for the optional masked gradient-PCA draws (pc_grad_mask_draws>1).
    forget_iter = cycle(make_plain_loader(forget_seqs, tokenizer, forget_batch, max_length, shuffle=True, seed=seed + 1))
    retain_iter = cycle(make_plain_loader(retain_seqs, tokenizer, retain_batch, max_length, shuffle=True, seed=seed))

    history: List[dict] = []
    bases = None
    stop = False
    global_step = 0
    for epoch in range(max_epochs):
        if bases is None or recompute_every_epoch:
            bases = _estimate_bases()
        last_break = last_retain = float("nan")
        for _ in range(steps_per_epoch):
            theta.train()
            lb = _forget_update(next(forget_iter), bases)
            lr_ = _retain_step(next(retain_iter))
            if lb is not None:
                last_break = lb
            last_retain = lr_
            global_step += 1
            if global_step % log_every == 0 and tracker:
                tracker.log({"step": global_step, "break_loss": last_break,
                             "retain_loss": last_retain, "flops": counter.total_flops}, step=global_step)
        cur_retain_ppl = pseudo_perplexity(
            theta.eval(), tokenizer, eval_seqs, device, eval_batch_size, max_length,
            masks_per_pass=eval_masks_per_pass).ppl
        theta.train()
        rec = {"epoch": epoch, "break_loss": last_break, "retain_loss": last_retain,
               "retain_ppl": cur_retain_ppl, "base_retain_ppl": base_retain_ppl,
               "retain_ppl_ratio": cur_retain_ppl / base_retain_ppl, "flops": counter.total_flops}
        history.append(rec)
        if tracker:
            tracker.log(rec, step=global_step)
        if cur_retain_ppl > term_ratio * base_retain_ppl:
            stop = True
        if stop:
            break

    for h in handles:
        h.remove()

    # bases -> CPU numpy for the Phase-8b subspace-projected-displacement probe
    subspace = {}
    for li in layers:
        b = bases[li]
        subspace[int(li)] = {
            "act_basis": b["act_basis"].detach().cpu().numpy(),
            "act_mean": b["act_mean"].detach().cpu().numpy(),
            "grad_basis": b["grad_basis"].detach().cpu().numpy(),
            "grad_mean": b["grad_mean"].detach().cpu().numpy(),
        }

    flops = counter.as_dict()
    flops["plain_equiv_flops"] = plain_equiv_flops
    flops["overhead_factor"] = (counter.total_flops / plain_equiv_flops) if plain_equiv_flops > 0 else float("nan")
    return {"model": theta, "flops": flops, "history": history, "subspace": subspace,
            "layers": layers, "base_retain_ppl": base_retain_ppl, "stopped_early": stop}
