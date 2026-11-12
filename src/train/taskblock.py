"""First-order MLAC/TAR-style task blocking — the ONLY safeguard-specific module.

All arms branch from the same base checkpoint θ0:
    B  = θ0 + this phase
    A' = θ0 + plain MLM matched on FLOPs (src/train/mlm.py)
    A  = θ0

Per outer step we simulate a K-step fine-tuning adversary on a clone φ of θ, then
push θ so that the adversary's *post-adaptation* forget loss stays high (capped at
`forget_margin`) while benign retain loss stays low. We use the first-order
coupling ∇_θ ≈ ∇_φ (FO-MAML / Reptile), avoiding second-order unrolling.

IMPLEMENTATION NOTE (pin against the references). There are several valid
first-order tamper-resistance formulations (where on the adversary trajectory the
loss is read, how the inner gradient folds into the outer update). The choice here
— penalise *low* post-adaptation forget loss via the FO identity — is one
defensible minimal version. Read the TAR (arXiv:2408.00761) / MLAC tamper-resistance
loss and confirm the exact coupling before the headline 35M run; document whatever
is chosen. A weak multiplier under this approximation is a LOWER BOUND on a careful
TAR implementation, not a verdict on task blocking. (Keep the Qi et al. durability
critique in mind for the Phase 6 variants.)
"""
from __future__ import annotations

import copy
from typing import List, Sequence

from src.data.datamodule import cycle, make_mlm_loader
from src.train.mlm import batch_tokens, cfg_get, mlm_loss
from src.utils.flops import FlopCounter, train_flops
from src.utils.seeding import set_seed


def train_taskblock(
    base_model,
    tokenizer,
    retain_seqs: Sequence[str],
    forget_seqs: Sequence[str],
    cfg,
    device: str,
    seed: int = 0,
    max_length: int = 1024,
    tracker=None,
) -> dict:
    """Run the task-blocking phase on ``base_model`` (θ0) -> B. Returns the model,
    a FLOP/wall-clock record (defender-side overhead), and training history."""
    import torch

    set_seed(seed)
    K = int(cfg_get(cfg, "inner_steps_K", 4))
    eta_inner = float(cfg_get(cfg, "eta_inner", 1e-4))
    eta_outer = float(cfg_get(cfg, "eta_outer", 5e-5))
    lambda_tr = float(cfg_get(cfg, "lambda_tr", 1.0))
    margin = float(cfg_get(cfg, "forget_margin", 6.0))
    outer_steps = int(cfg_get(cfg, "outer_steps", 3000))
    retain_batch = int(cfg_get(cfg, "retain_batch", 16))
    forget_batch = int(cfg_get(cfg, "forget_batch", 16))
    warmup_steps = int(cfg_get(cfg, "warmup_steps", 100))
    mlm_prob = float(cfg_get(cfg, "mlm_probability", 0.15))
    grad_clip = float(cfg_get(cfg, "grad_clip", 1.0))
    log_every = int(cfg_get(cfg, "log_every", 25))

    theta = base_model.train().to(device)
    theta_params: List = [p for p in theta.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in theta_params)

    optimizer = torch.optim.AdamW(theta_params, lr=eta_outer)
    from torch.optim.lr_scheduler import LambdaLR

    scheduler = LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, warmup_steps)))

    # φ: a persistent clone, refreshed from θ each outer step (the adversary starts at θ).
    phi = copy.deepcopy(theta).to(device)
    phi_params: List = [p for p in phi.parameters() if p.requires_grad]

    retain_iter = cycle(make_mlm_loader(retain_seqs, tokenizer, retain_batch, mlm_prob, max_length, shuffle=True, seed=seed))
    forget_iter = cycle(make_mlm_loader(forget_seqs, tokenizer, forget_batch, mlm_prob, max_length, shuffle=True, seed=seed + 1))

    counter = FlopCounter(n_params=n_params)
    plain_equiv_flops = 0.0  # FLOPs a plain MLM run would spend on the same retain batches
    history: List[dict] = []

    for step in range(outer_steps):
        R = next(retain_iter)
        F = next(forget_iter)
        f_tokens = batch_tokens(F)
        r_tokens = batch_tokens(R)

        # --- refresh φ ← θ ---
        with torch.no_grad():
            for pp, pt in zip(phi.parameters(), theta.parameters()):
                pp.copy_(pt)

        # --- inner adversary: K SGD steps minimising forget loss, from current θ ---
        for _ in range(K):
            f_loss = mlm_loss(phi, F, device)
            # allow_unused: ESM-2 uses rotary position embeddings, so
            # position_embeddings.weight (and contact_head) never enter the MLM graph
            # -> their grad is None; those params just don't update.
            grads = torch.autograd.grad(f_loss, phi_params, allow_unused=True)
            with torch.no_grad():
                for p, g in zip(phi_params, grads):
                    if g is not None:
                        p.add_(g, alpha=-eta_inner)
            counter.record(f_tokens, fwd_bwd=True)

        # --- outer (defender) objective ---
        L_retain = mlm_loss(theta, R, device)
        g_retain = torch.autograd.grad(L_retain, theta_params, allow_unused=True)
        counter.record(r_tokens, fwd_bwd=True)

        L_forget_adv = mlm_loss(phi, F, device)
        tr_term = torch.clamp(L_forget_adv, max=margin)  # min(L_forget_adv, margin); want LARGE
        g_tr = torch.autograd.grad(tr_term, phi_params, allow_unused=True)
        counter.record(f_tokens, fwd_bwd=True)

        # gradient to minimise = +g_retain (descend benign) − λ·g_tr (ascend capped forget).
        # First-order coupling: gradients evaluated at φ are applied to θ (∇_θ ≈ ∇_φ).
        # Params unused in the MLM graph (rotary position emb, contact head) get None grads.
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for pt, gr, gt in zip(theta_params, g_retain, g_tr):
                if gr is None and gt is None:
                    pt.grad = None
                    continue
                grad = torch.zeros_like(pt)
                if gr is not None:
                    grad = grad + gr
                if gt is not None:
                    grad = grad - lambda_tr * gt
                pt.grad = grad
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(theta_params, grad_clip)
        optimizer.step()
        scheduler.step()
        counter.optimizer_steps += 1
        plain_equiv_flops += train_flops(n_params, r_tokens)

        if step % log_every == 0 or step == outer_steps - 1:
            rec = {
                "step": step,
                "L_retain": float(L_retain.item()),
                "L_forget_adv": float(L_forget_adv.item()),
                "tr_term": float(tr_term.item()),
                "flops": counter.total_flops,
            }
            history.append(rec)
            if tracker:
                tracker.log(rec, step=step)

    flops = counter.as_dict()
    # Defender-side overhead vs a plain MLM step.
    flops["plain_equiv_flops"] = plain_equiv_flops
    flops["overhead_factor"] = (counter.total_flops / plain_equiv_flops) if plain_equiv_flops > 0 else float("nan")
    return {"model": theta, "flops": flops, "history": history}
