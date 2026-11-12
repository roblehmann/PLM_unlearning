"""Attacker harness: fine-tune a *fresh copy* of a given checkpoint (A / B / A')
toward a target family under a recipe (LoRA or full-FT), at a fixed budget n, with
the MLM objective. Returns the acquisition curve (held-out target-family
pseudo-perplexity per epoch) and membership AUROC.

Each call reloads the checkpoint from disk so budgets/seeds never contaminate one
another. The budget-n subsampling of the attacker pool is done by the caller
(sweep) — ``attacker_seqs`` here is exactly the n sequences the attacker gets.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from src.data.datamodule import make_mlm_loader, subsample
from src.eval.membership import membership_auroc
from src.eval.pseudo_perplexity import pseudo_perplexity
from src.models.load import LoraSpec, attach_lora, load_checkpoint
from src.train.mlm import batch_tokens, cfg_get, mlm_loss
from src.utils.flops import FlopCounter
from src.utils.seeding import set_seed


def run_attack(
    checkpoint_path: str,
    tokenizer,
    attacker_seqs: Sequence[str],
    target_test_seqs: Sequence[str],
    attack_cfg,
    device: str,
    seed: int = 0,
    max_length: int = 1024,
    background_seqs: Optional[Sequence[str]] = None,
    ppl_batch_size: int = 64,
    eval_masks_per_pass: int = 8,
    max_eval_seqs: Optional[int] = None,
    eval_every: int = 1,
    tracker=None,
    save_model_path: Optional[str] = None,
    amp_dtype: Optional[str] = None,
) -> dict:
    """Run one attack cell. Returns a dict including:

    ``amp_dtype`` (None | "bfloat16" | "float16") wraps only the TRAINING forward/backward
    in ``torch.autocast``; params/grads/optimizer stay fp32 (autocast keeps fp32 master
    weights) and evaluation stays fp32, so the reported PPL is measured identically to a
    pure-fp32 run — the mixed precision only speeds/shrinks the fit. Default None = exact
    prior behaviour, so every existing caller is untouched. Used by the scale-ladder gate to
    make the 650M floor fit feasible; bf16 needs no GradScaler.
        best_target_ppl   — min held-out target-family PPL over epochs (adversary
                            early-stops at its best; this is what the multiplier uses)
        final_target_ppl  — PPL at the last epoch
        membership_auroc  — held-out target vs background (if background given)
        history           — per-epoch acquisition curve
    """
    import torch

    set_seed(seed)
    # Cap held-out sets used only for scoring (not training) so PPL/AUROC stay cheap on
    # very large families. Deterministic by seed.
    if max_eval_seqs:
        target_test_seqs = subsample(target_test_seqs, max_eval_seqs, seed=seed)
        if background_seqs:
            background_seqs = subsample(background_seqs, max_eval_seqs, seed=seed)
    mode = cfg_get(attack_cfg, "mode", "lora")
    lr = float(cfg_get(attack_cfg, "lr", 1e-4))
    batch_size = int(cfg_get(attack_cfg, "batch_size", 16))
    mlm_prob = float(cfg_get(attack_cfg, "mlm_probability", 0.15))
    max_epochs = int(cfg_get(attack_cfg, "max_epochs", 20))
    weight_decay = float(cfg_get(attack_cfg, "weight_decay", 0.0))
    patience = int(cfg_get(attack_cfg, "patience", 0))
    grad_clip = float(cfg_get(attack_cfg, "grad_clip", 1.0))

    model = load_checkpoint(checkpoint_path, device=device)
    if mode == "lora":
        model = attach_lora(model, LoraSpec.from_cfg(attack_cfg))
    # full_ft: all params already trainable
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    counter = FlopCounter(n_params=sum(p.numel() for p in params))

    # bf16/fp16 autocast wraps only the training step (below); default None -> fp32, so the
    # FLOP accounting and every non-AMP caller are unchanged. fp16 would need a GradScaler;
    # we only use bf16 here (dynamic-range-safe, no scaler), guarded by device == cuda.
    _amp = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(amp_dtype) if amp_dtype else None
    amp_on = _amp is not None and str(device).startswith("cuda")

    loader = make_mlm_loader(attacker_seqs, tokenizer, batch_size, mlm_prob, max_length, shuffle=True, seed=seed)

    def eval_target():
        return pseudo_perplexity(
            model, tokenizer, target_test_seqs, device, ppl_batch_size, max_length, eval_masks_per_pass
        )

    res0 = eval_target()
    history: List[dict] = [{"epoch": 0, "target_ppl": res0.ppl}]  # pre-attack PPL
    best_ppl = history[0]["target_ppl"]
    best_res = res0  # per-sequence decomposition at the best epoch (bootstrap SE upstream)
    best_epoch = 0
    bad = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            if amp_on:
                with torch.autocast(device_type="cuda", dtype=_amp):
                    loss = mlm_loss(model, batch, device)
            else:
                loss = mlm_loss(model, batch, device)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            counter.record(batch_tokens(batch), fwd_bwd=True, is_optimizer_step=True)
        # Score only every `eval_every` epochs (+ always the last) — held-out PPL is the
        # dominant cost; this does not change what is trained on.
        if not (epoch % max(1, eval_every) == 0 or epoch == max_epochs):
            continue
        res = eval_target()
        ppl = res.ppl
        history.append({"epoch": epoch, "target_ppl": ppl})
        if tracker:
            tracker.log({"epoch": epoch, "target_ppl": ppl}, step=epoch)
        if ppl < best_ppl - 1e-6:
            best_ppl, best_res, best_epoch, bad = ppl, res, epoch, 0
        else:
            bad += 1
            if patience and bad >= patience:
                break

    auroc = None
    if background_seqs is not None and len(background_seqs) > 0:
        try:
            auroc = membership_auroc(
                model, tokenizer, list(target_test_seqs), list(background_seqs),
                device=device, max_length=max_length, seed=seed,
            ).auroc
        except Exception as e:  # probe needs enough samples; never fail the run on it
            print(f"[attack] membership probe skipped: {e!r}")

    result = {
        "best_target_ppl": best_ppl,
        # Per-sequence NLL / scored-position counts behind `best_target_ppl`, so a caller
        # can bootstrap a corpus-level SE for it (the scale-ladder headroom gate needs the
        # floor's SE to check that tau is separable from the floor).
        "best_target_ppl_per_seq_nll": list(best_res.per_seq_nll),
        "best_target_ppl_per_seq_positions": list(best_res.per_seq_positions),
        "final_target_ppl": history[-1]["target_ppl"],
        "naive_target_ppl": history[0]["target_ppl"],
        "best_epoch": best_epoch,
        "epochs_run": len(history) - 1,
        "membership_auroc": auroc,
        "n_train": len(attacker_seqs),
        "trainable_params": sum(p.numel() for p in params),
        "flops": counter.as_dict(),
        "history": history,
    }
    # Optionally persist θ_attacked (the relearned model) before it is freed — used by the
    # Phase-8d repair-delta probe, NOT by the sweep (default None keeps sweep behavior).
    if save_model_path is not None:
        from src.models.load import save_model
        save_model(model, save_model_path)

    # Free the model/optimizer so GPU memory does not accumulate across sweep cells.
    import gc

    del model, optimizer, params, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
